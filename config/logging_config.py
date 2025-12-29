import builtins
import json
import logging
import logging.config
import os
import sys
import time
import functools
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union


_LOGGING_INITIALIZED = False
_PRINT_PATCHED = False
_ORIGINAL_PRINT = builtins.print


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter to make cloud log ingestion easier."""

    def format(self, record: logging.LogRecord) -> str:
        record_message = record.getMessage()
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record_message,
        }

        # Capture any custom contextual attributes that were added to the record.
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            }
        }
        if extra_fields:
            payload["extra"] = extra_fields

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


class CSVFormatter(logging.Formatter):
    """Formatter that writes log records as escaped CSV rows."""

    def __init__(self, fields: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.fields = fields or ["timestamp", "level", "logger", "message"]

    @staticmethod
    def _escape(value: str) -> str:
        if value is None:
            value = ""
        value = str(value).replace('"', '""')
        return f'"{value}"'

    def format(self, record: logging.LogRecord) -> str:
        field_values = []
        for field in self.fields:
            if field == "timestamp":
                value = self.formatTime(record, self.datefmt)
            elif field == "level":
                value = record.levelname
            elif field == "logger":
                value = record.name
            elif field == "message":
                value = record.getMessage()
            else:
                value = getattr(record, field, record.__dict__.get(field, ""))
            field_values.append(self._escape(value))
        if record.exc_info:
            field_values.append(self._escape(self.formatException(record.exc_info)))
        return ",".join(field_values)


def _build_logging_config(env: str, level: str) -> dict:
    console_formatter = {
        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    json_formatter = {
        "()": f"{__name__}.JsonFormatter",
        "datefmt": "%Y-%m-%dT%H:%M:%S%z",
    }

    log_file = os.getenv("APP_LOG_FILE", "logging.csv")

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "console"
            if env == "local"
            else "json",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": log_file,
            "mode": "a",
            "encoding": "utf-8",
            "formatter": "csv",
        },
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": console_formatter,
            "json": json_formatter,
            "csv": {
                "()": f"{__name__}.CSVFormatter",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": handlers,
        "root": {"handlers": ["console", "file"], "level": level},
    }


def setup_logging(env: Optional[str] = None, default_level: str = "INFO") -> None:
    """
    Configure the root logger once per process.

    Args:
        env: Explicit environment identifier ("local", "cloud", etc.). Falls back
            to APP_ENV environment variable or "local".
        default_level: Fallback log level when LOG_LEVEL is not set.
    """
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    resolved_env = (env or os.getenv("APP_ENV") or "local").lower()
    log_level = os.getenv("LOG_LEVEL", default_level).upper()
    config = _build_logging_config(resolved_env, log_level)

    logging.config.dictConfig(config)
    _LOGGING_INITIALIZED = True
    _patch_print()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Helper that ensures logging is initialized before returning a logger instance.
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


def _get_session_state_value(key: str, default: Any = None) -> Any:
    try:
        import streamlit as st

        return st.session_state.get(key, default)
    except Exception:
        return default


def _set_session_state_value(key: str, value: Any) -> None:
    try:
        import streamlit as st

        st.session_state[key] = value
    except Exception:
        pass


def _dynamic_context() -> Dict[str, Any]:
    return {
        "agent": _get_session_state_value("agent_name", os.getenv("DEFAULT_AGENT", "unknown_agent")),
        "step": _get_session_state_value("current_step", "unknown_step"),
        "triggered_by": _get_session_state_value(
            "user_email",
            _get_session_state_value("role", os.getenv("DEFAULT_USER", "unknown_user")),
        ),
        "page": _get_session_state_value(
            "current_page",
            _get_session_state_value("active_workflow", os.getenv("DEFAULT_PAGE", "unknown_page")),
        ),
        "job_id": _get_session_state_value("job_id", os.getenv("JOB_ID", "local")),
    }


class ContextualLogger:
    def __init__(self, name: Optional[str] = None, base_context: Optional[Dict[str, Any]] = None):
        self._logger = get_logger(name)
        self._base_context = base_context or {}

    def _merge_context(self, context: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update({k: v for k, v in self._base_context.items() if v is not None})
        for key, value in _dynamic_context().items():
            if value is not None:
                merged[key] = value
        if context:
            merged.update({k: v for k, v in context.items() if v is not None})
        if extra:
            merged.update(extra)
        return merged

    def debug(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.debug(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def info(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.info(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def warning(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.warning(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def error(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.error(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def exception(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.exception(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def critical(self, msg, *args, context: Optional[Dict[str, Any]] = None, **kwargs):
        extra = kwargs.pop("extra", None)
        self._logger.critical(msg, *args, extra=self._merge_context(context, extra), **kwargs)

    def get_child(self, suffix: str) -> "ContextualLogger":
        return ContextualLogger(f"{self._logger.name}.{suffix}", base_context=self._base_context)


class StepTimer:
    def __init__(self, logger: ContextualLogger, step_name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.step_name = step_name
        self.context = context or {}
        self._start: Optional[float] = None
        self._previous_step: Any = None

    def __enter__(self):
        self._start = time.perf_counter()
        self._previous_step = _get_session_state_value("current_step", None)
        _set_session_state_value("current_step", self.step_name)
        self.logger.info("Step started", context=self._ctx())
        return self

    def _ctx(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = {"step": self.step_name}
        ctx.update(self.context)
        if extra:
            ctx.update(extra)
        return ctx

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed = round(time.perf_counter() - (self._start or time.perf_counter()), 2)
        ctx = self._ctx({"elapsed_seconds": elapsed})
        _set_session_state_value("current_step", self._previous_step)
        if exc_type:
            self.logger.exception("Step failed", context=ctx)
            return False
        self.logger.info("Step finished", context=ctx)
        return False


def contextual_logger(name: Optional[str] = None, **base_context) -> ContextualLogger:
    return ContextualLogger(name, base_context=base_context)


def step_context(logger: Union[ContextualLogger, logging.Logger], step_name: str, context: Optional[Dict[str, Any]] = None) -> StepTimer:
    contextual = logger if isinstance(logger, ContextualLogger) else contextual_logger(logger.name if isinstance(logger, logging.Logger) else logger)
    return StepTimer(contextual, step_name, context=context)


def log_step(logger: Union[ContextualLogger, logging.Logger], step_name: str, context: Optional[Dict[str, Any]] = None):
    contextual = logger if isinstance(logger, ContextualLogger) else contextual_logger(logger.name if isinstance(logger, logging.Logger) else logger)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with StepTimer(contextual, step_name, context=context):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _patch_print():
    """
    Route builtin print statements through logging so legacy print calls
    automatically respect the configured handlers. Set DISABLE_PRINT_REDIRECT=1
    to opt out.
    """
    global _PRINT_PATCHED
    if _PRINT_PATCHED or os.getenv("DISABLE_PRINT_REDIRECT"):
        return

    def logging_print(*args, **kwargs):
        file = kwargs.get("file", sys.stdout)
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        flush = kwargs.get("flush", False)

        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message = f"{message}{end}"

        if file is sys.stderr:
            logging.getLogger("app.print.stderr").error(message)
            target_logger = logging.getLogger("app.print.stderr")
        elif file in (None, sys.stdout):
            logging.getLogger("app.print").info(message)
            target_logger = logging.getLogger("app.print")
        else:
            return _ORIGINAL_PRINT(*args, **kwargs)

        if flush:
            for handler in target_logger.handlers or logging.getLogger().handlers:
                try:
                    handler.flush()
                except Exception:
                    continue

    builtins.print = logging_print
    _PRINT_PATCHED = True
