"""Console progress driven by real engine emissions, including warmup/profile."""
from contextlib import contextmanager

from tqdm import tqdm


class BenchmarkProgress:
    def __init__(self, *, enabled, stages, tokens_per_batch, description):
        self.enabled = enabled
        self.tokens_per_batch = tokens_per_batch
        self.stages = tqdm(total=stages, desc=description, unit='stage',
                           disable=not enabled, dynamic_ncols=True,
                           position=0, mininterval=0.25)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stages.close()

    @contextmanager
    def phase(self, measurement, description):
        self.stages.set_postfix_str(description)
        with tqdm(total=self.tokens_per_batch, desc=description, unit='tok',
                  disable=not self.enabled, dynamic_ncols=True, position=1,
                  leave=False, mininterval=0.25) as tokens:
            measurement.progress_callback = tokens.update if self.enabled else None
            try:
                yield
            finally:
                measurement.progress_callback = None
        # A failed phase is not reported as completed.
        self.stages.update(1)
