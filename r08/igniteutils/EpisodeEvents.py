from ignite.engine import EventEnum


class EpisodeEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"
