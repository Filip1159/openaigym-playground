from typing import Optional

from ignite.engine import Engine, Events, State

from r08.experience.ExperienceSource import ExperienceSource
from r08.igniteutils.EpisodeEvents import EpisodeEvents


class EndOfEpisodeHandler:
    def __init__(self, exp_source: ExperienceSource, alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None,
                 subsample_end_of_episode: Optional[int] = None):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param subsample_end_of_episode: if given, end of episode event will be subsampled by this amount
        """
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._best_avg_reward = None
        self._subsample_end_of_episode = subsample_end_of_episode

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps
            self._update_smoothed_metrics(engine, reward, steps)
            if self._subsample_end_of_episode is None or engine.state.episode % self._subsample_end_of_episode == 0:
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and engine.state.metrics['avg_reward'] >= self._bound_avg_reward:
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics['avg_reward']
            elif self._best_avg_reward < engine.state.metrics['avg_reward']:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics['avg_reward']

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(('avg_reward', 'avg_steps'), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1-self._alpha) * val
