# src/observability/logger.py
import os
import logging
from logging.handlers import RotatingFileHandler

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Misma carpeta para ambos logs
LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

BASE_LOG_FILE = os.path.join(LOG_DIR, os.getenv("LOG_FILE", "piloto.log"))
PROMPT_LOG_FILE = os.path.join(LOG_DIR, os.getenv("LOG_PROMPT_FILE", "piloto_prompts.log"))


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return True


class TraceAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.setdefault("trace_id", self.extra.get("trace_id", "-"))
        kwargs["extra"] = extra
        return msg, kwargs


# -------------------------
# Base logger (piloto.log)
# -------------------------
_base_logger = logging.getLogger("piloto")
_base_logger.setLevel(LOG_LEVEL)
_base_logger.propagate = False

if not _base_logger.handlers:
    h = RotatingFileHandler(
        BASE_LOG_FILE,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s trace_id=%(trace_id)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    h.setFormatter(fmt)
    h.addFilter(TraceIdFilter())
    _base_logger.addHandler(h)


# -------------------------
# Prompt logger (siempre ON)
# -------------------------
_prompt_logger = logging.getLogger("piloto.prompts")
_prompt_logger.setLevel(LOG_LEVEL)
_prompt_logger.propagate = False

if not _prompt_logger.handlers:
    ph = RotatingFileHandler(
        PROMPT_LOG_FILE,
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    pfmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s trace_id=%(trace_id)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ph.setFormatter(pfmt)
    ph.addFilter(TraceIdFilter())
    _prompt_logger.addHandler(ph)

    # Confirmación en el log base (ya no rompe)
    _base_logger.info(
        f"event=prompt_logger.initialized path={PROMPT_LOG_FILE}",
        extra={"trace_id": "-"},
    )


def get_logger(trace_id: str = "-") -> TraceAdapter:
    return TraceAdapter(_base_logger, {"trace_id": trace_id})


def get_prompt_logger(trace_id: str = "-") -> TraceAdapter:
    return TraceAdapter(_prompt_logger, {"trace_id": trace_id})


def prompts_enabled() -> bool:
    # Siempre habilitado para verificación
    return True
