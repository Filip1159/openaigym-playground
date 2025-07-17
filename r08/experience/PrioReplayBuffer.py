from typing import Generator, List, Tuple

import numpy as np

from r08.experience.ExperienceSourceFirstLast import ExperienceSourceFirstLast, ExperienceFirstLast


# parametry bufora
BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:
    def __init__(self, exp_source: ExperienceSourceFirstLast, buf_size: int, prob_alpha: float = 0.6):
        self.exp_source_iter: Generator[ExperienceFirstLast, None, None] = iter(exp_source)
        self.prob_alpha: float = prob_alpha
        self.capacity: int = buf_size
        self.pos: int = 0
        self.buffer: List[ExperienceFirstLast] = []
        self.priorities: np.ndarray = np.zeros((buf_size, ), dtype=np.float32)
        self.beta: float = BETA_START

    def update_beta(self, idx: int) -> float:
        v = BETA_START + idx * (1.0 - BETA_START) / BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self) -> int:
        return len(self.buffer)

    def populate(self, count: int) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[ExperienceFirstLast], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices: np.ndarray, batch_priorities: np.ndarray) -> None:
        for i, priority in zip(batch_indices, batch_priorities):
            self.priorities[i] = priority
