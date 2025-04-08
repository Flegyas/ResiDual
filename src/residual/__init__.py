import logging
from datetime import datetime
from typing import Optional

import gin
import lovely_tensors as lt
from latentis import PROJECT_ROOT
from rich.console import ConsoleRenderable
from rich.logging import RichHandler
from rich.traceback import Traceback

from residual.nn.model_registry import get_text_encoder, get_vision_encoder  # noqa

from .tracing.tracer import ResidualTracer

lt.monkey_patch()


gin.constant("PROJECT_ROOT", PROJECT_ROOT)


class NNRichHandler(RichHandler):
    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Optional[Traceback],
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        # Hack to display the logger name instead of the filename in the rich logs
        path = record.name  # str(Path(record.pathname))
        level = self.get_level_text(record)
        time_format = None if self.formatter is None else self.formatter.datefmt
        log_time = datetime.fromtimestamp(record.created)

        log_renderable = self._log_render(
            self.console,
            [message_renderable] if not traceback else [message_renderable, traceback],
            log_time=log_time,
            time_format=time_format,
            level=level,
            path=path,
            line_no=record.lineno,
            link_path=record.pathname if self.enable_link_path else None,
        )
        return log_renderable


# Required workaround because PyTorch Lightning configures the logging on import,
# thus the logging configuration defined in the __init__.py must be called before
# the lightning import otherwise it has no effect.
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/1503
lightning_logger = logging.getLogger("lightning.pytorch")
# Remove all handlers associated with the lightning logger.
for handler in lightning_logger.handlers[:]:
    lightning_logger.removeHandler(handler)
lightning_logger.propagate = True

FORMAT = "%(message)s"
logging.basicConfig(
    format=FORMAT,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        NNRichHandler(
            rich_tracebacks=True,
            show_level=True,
            show_path=True,
            show_time=True,
            omit_repeated_times=True,
        )
    ],
)

__all__ = ["ResidualTracer"]
