import sys
import time
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02}m{secs:02}s"
    if minutes:
        return f"{minutes}m{secs:02}s"
    return f"{secs}s"


class ProgressBar:
    noninteractive_log_delay = 30.0
    noninteractive_log_interval = 30.0

    def __init__(self, text: str, count: int, transient: bool = True):
        self.text = text
        self.count = count
        self.transient = transient
        self.noninteractive = bool(self.text) and not sys.stdout.isatty()
        self.start_time = None
        self.last_log_time = None
        self.last_value = 0
        self.logged = False
        if self.text and not self.noninteractive:
            self.progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=transient,
                speed_estimate_period=600.0,
            )
            self.task_id = self.progress.add_task(text, total=count)

    def _reset_noninteractive(self):
        now = time.monotonic()
        self.start_time = now
        self.last_log_time = now
        self.last_value = 0
        self.logged = False

    def _log_noninteractive(self, value: int, force: bool = False):
        now = time.monotonic()
        elapsed = 0.0 if self.start_time is None else now - self.start_time
        self.last_value = value

        should_log = force
        if not should_log:
            should_log = (
                elapsed >= self.noninteractive_log_delay
                and now - self.last_log_time >= self.noninteractive_log_interval
            )
        if not should_log:
            return

        percent = 100.0 if self.count == 0 else (100.0 * value / self.count)
        speed = value / elapsed if elapsed > 0 and value > 0 else None
        remaining = (
            None
            if speed is None or value >= self.count
            else (self.count - value) / speed
        )
        print(
            f"{self.text}: {value}/{self.count} ({percent:5.1f}%) "
            f"elapsed {_format_duration(elapsed)} "
            f"eta {_format_duration(remaining)}"
        )
        sys.stdout.flush()
        self.last_log_time = now
        self.logged = True

    def __enter__(self):
        if self.text:
            if self.noninteractive:
                self._reset_noninteractive()
            else:
                self.progress.start()
            sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.text:
            if self.noninteractive:
                if self.logged and self.last_value < self.count:
                    self._log_noninteractive(self.last_value, force=True)
            else:
                if not self.transient:
                    self.progress.update(self.task_id, completed=self.count)
                self.progress.stop()

    def update(self, value: int):
        if self.text:
            if self.noninteractive:
                self._log_noninteractive(value)
            else:
                self.progress.update(self.task_id, completed=value)
            sys.stdout.flush()

    def new_task(self, text: str, count: int):
        self.text = text
        self.count = count
        if self.text:
            if self.noninteractive:
                self._reset_noninteractive()
            else:
                self.progress.update(
                    self.task_id, description=self.text, total=count, progress=0
                )
