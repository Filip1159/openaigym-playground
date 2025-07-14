import numpy as np
import warnings
from datetime import timedelta
from types import SimpleNamespace
from typing import Iterable, List

from r08.igniteutils.EndOfEpisodeHandler import EndOfEpisodeHandler
from r08.igniteutils.EpisodeFPSHandler import EpisodeFPSHandler
from r08.igniteutils.EpisodeEvents import EpisodeEvents
from r08.igniteutils.PeriodicEvents import PeriodicEvents
from ignite.engine import Engine
from ignite.metrics import RunningAverage

from r08.experience.ExperienceReplayBuffer import ExperienceReplayBuffer
from r08.experience.ExperienceSourceFirstLast import ExperienceFirstLast


SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    })
}


def unpack_batch(batch: List[ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # wynik i tak zostanie zamaskowany
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def batch_generator(buffer: ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, extra_metrics: Iterable[str] = ()):
    # pozbycie si� ostrze�enia o brakuj�cym wska�niku
    warnings.simplefilter("ignore", category=UserWarning)

    handler = EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Epizod %d: nagroda=%.0f, kroki=%s, "
              "predkosc=%.1f f/s, uplynelo=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Gra ukonczona w ci�gu %s sekund, po %d epizodach "
              "i %d iteracjach!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    # co 100 iteracji wysy�aj dane do tensorboard 
    PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
