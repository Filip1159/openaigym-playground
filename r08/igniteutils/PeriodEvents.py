from ignite.engine import EventEnum


class PeriodEvents(EventEnum):
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"
