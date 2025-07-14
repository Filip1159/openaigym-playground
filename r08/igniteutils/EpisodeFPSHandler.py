import time

from ignite.engine import Engine, Events
from ignite.handlers import Timer

from r08.igniteutils.EpisodeEvents import EpisodeEvents


class EpisodeFPSHandler:
    FPS_METRIC = 'fps'
    AVG_FPS_METRIC = 'avg_fps'
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self, fps_mul: float = 1.0, fps_smooth_alpha: float = 0.98):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()
        self._fps_smooth_alpha = fps_smooth_alpha

    def attach(self, engine: Engine, manual_step: bool = False):
        self._timer.attach(engine, step=None if manual_step else Events.ITERATION_COMPLETED)
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)
        engine.state.metrics[self.AVG_FPS_METRIC] = 0

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every time we've communicated with environment
        to get proper FPS
        :return:
        """
        self._timer.step()

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            fps = self._fps_mul / t_val
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None or avg_fps <= 0:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smooth_alpha
                avg_fps += (1-self._fps_smooth_alpha) * fps
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts
        self._timer.reset()
