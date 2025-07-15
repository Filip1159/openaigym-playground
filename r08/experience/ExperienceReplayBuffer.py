from typing import Generator, List, Iterator

import numpy as np

from r08.experience.ExperienceSourceFirstLast import ExperienceSourceFirstLast, ExperienceFirstLast


class ExperienceReplayBuffer:
    def __init__(self, experience_source: ExperienceSourceFirstLast, buffer_size: int):
        self.experience_source_iter: Generator[ExperienceFirstLast, None, None] = iter(experience_source)
        self.buffer: List[ExperienceFirstLast] = []  # circular buffer
        self.capacity: int = buffer_size
        self.pos: int = 0  # circular index

    def __len__(self) -> int:
        return len(self.buffer)

    def __iter__(self) -> Iterator[ExperienceFirstLast]:
        return iter(self.buffer)

    def sample(self, batch_size) -> List[ExperienceFirstLast]:
        """
        Get one random batch from experience replay
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def populate(self, samples: int) -> None:
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

    def _add(self, sample: ExperienceFirstLast) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:  # buffer full, overwrite
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity
