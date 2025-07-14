from typing import Generator, Tuple, Deque, Optional

import gymnasium as gym

from collections import deque, namedtuple  # double-ended queue

from gymnasium.core import ObsType, ActType

from r08.agent.DQNAgent import DQNAgent


# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])



class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env: gym.Env, agent: DQNAgent, max_steps_count: int = 2):
        """
        :param env: environment
        :param agent: callable to convert batch of states into actions to take
        :param max_steps_count: count of steps to track for every experience chain
        """
        assert max_steps_count >= 1
        self.env = env
        self.agent = agent
        self.steps_count = max_steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self) -> Generator[Tuple[Experience, ...], None, None]:
        history: Deque[Experience] = deque(maxlen=self.steps_count)
        current_reward: float = 0.0
        current_steps: int = 0
        state, _ = self.env.reset()

        while True:
            action: ActType = self._select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            current_reward += reward
            current_steps += 1

            if state is not None:
                history.append(Experience(state=state, action=action, reward=reward, done=is_done))

            # training probe ready to return
            if len(history) == self.steps_count:
                yield tuple(history)

            state = next_state

            if is_done:
                yield from self._flush_episode(history, current_reward, current_steps)
                state = self.env.reset()[0]
                current_reward = 0.0
                current_steps = 0


    def _select_action(self, state: Optional[ObsType]) -> ActType:
        if state is None:
            return self.env.action_space.sample()
        else:
            return self.agent(state)


    def _flush_episode(
            self,
            history: Deque[Experience],
            current_reward: float,
            current_steps: int
    ) -> Generator[Tuple[Experience, ...], None, None]:

        # in case of very short episode (shorter than our steps count), send gathered history
        if len(history) < self.steps_count:
            yield tuple(history)

        # generate tail of history
        while len(history) > 1:
            history.popleft()
            yield tuple(history)

        self.total_rewards.append(current_reward)
        self.total_steps.append(current_steps)
        history.clear()

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res
