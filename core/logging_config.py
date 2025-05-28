from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import sys


@dataclass
class ResearchLog:
    """Structured log entry for research activities"""

    timestamp: datetime
    level: str
    action: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "action": self.action,
            "message": self.message,
            "context": self.context,
        }


class ResearchLogger:
    """Custom logger for research activities with structured logging"""

    def __init__(self, name: str = "research_agent"):
        self.logger = logging.getLogger(name)
        self.logs: List[ResearchLog] = []
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with appropriate handlers and formatters"""
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers to prevent duplicates
        self.logger.handlers.clear()

        # Prevent propagation to parent loggers to avoid duplicate messages
        self.logger.propagate = False

        # Console handler for general output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_format)

        # File handler for detailed logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(file_format)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def _log_structured(
        self,
        level: str,
        action: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Create a structured log entry"""
        log_entry = ResearchLog(
            timestamp=datetime.now(),
            level=level,
            action=action,
            message=message,
            context=context or {},
        )
        self.logs.append(log_entry)

        # Also log to standard logger
        log_message = f"[{action}] {message}"
        if context:
            log_message += f" | Context: {context}"

        getattr(self.logger, level.lower())(log_message)

    def info(self, action: str, message: str, **context):
        """Log info level message"""
        self._log_structured("INFO", action, message, context)

    def warning(self, action: str, message: str, **context):
        """Log warning level message"""
        self._log_structured("WARNING", action, message, context)

    def error(self, action: str, message: str, **context):
        """Log error level message"""
        self._log_structured("ERROR", action, message, context)

    def debug(self, action: str, message: str, **context):
        """Log debug level message"""
        self._log_structured("DEBUG", action, message, context)

    def section(self, title: str):
        """Log a section separator"""
        separator = "=" * 60
        self.logger.info(f"\n{separator}\n{title}\n{separator}")

    def subsection(self, title: str):
        """Log a subsection separator"""
        separator = "-" * 60
        self.logger.info(f"\n{separator}\n{title}\n{separator}")

    def action_start(self, action_type: str, details: Dict[str, Any]):
        """Log the start of an action"""
        self.info(f"{action_type}_START", f"Starting {action_type} action", **details)

    def action_complete(self, action_type: str, result: Dict[str, Any]):
        """Log the completion of an action"""
        self.info(
            f"{action_type}_COMPLETE", f"Completed {action_type} action", **result
        )

    def action_failed(self, action_type: str, error: str, details: Dict[str, Any]):
        """Log a failed action"""
        self.error(
            f"{action_type}_FAILED", f"Failed {action_type} action: {error}", **details
        )

    def progress(
        self,
        step: int,
        total_steps: int,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log research progress"""
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        self.info(
            "PROGRESS",
            f"Step {step}/{total_steps} ({progress_pct:.1f}%): {message}",
            **(metrics or {}),
        )

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all structured logs as dictionaries"""
        return [log.to_dict() for log in self.logs]

    def save_logs(self, filepath: Path):
        """Save structured logs to a JSON file"""
        import json

        with open(filepath, "w") as f:
            json.dump(self.get_logs(), f, indent=2)


class ResearchError(Exception):
    """Base exception for research-related errors"""

    def __init__(
        self, message: str, action: str = "", context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.action = action
        self.context = context or {}


class SearchError(ResearchError):
    """Error during search operations"""

    pass


class ExtractionError(ResearchError):
    """Error during content extraction"""

    pass


class SchemaError(ResearchError):
    """Error related to schema operations"""

    pass


class LLMError(ResearchError):
    """Error during LLM interactions"""

    pass


# Global logger instance
_logger: Optional[ResearchLogger] = None


def get_logger() -> ResearchLogger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = ResearchLogger()
    return _logger


def set_logger(logger: ResearchLogger):
    """Set the global logger instance"""
    global _logger
    _logger = logger
