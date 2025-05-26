from abc import ABC, abstractmethod
from typing import Any, Dict

from core.data_structures import ResearchContext, ResearchAction
from core.config import get_config
from core.logging_config import get_logger


class ActionExecutor(ABC):
    """Base class for action executors"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()

    @abstractmethod
    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute the action and return results"""
        pass

    def pre_execute(self, action: ResearchAction, context: ResearchContext):
        """Hook for pre-execution tasks"""
        self.logger.action_start(
            action.action_type.value,
            {
                "reason": action.reason,
                "parameters": action.parameters,
                "step": context.current_step,
            },
        )

    def post_execute(
        self, action: ResearchAction, context: ResearchContext, result: Dict[str, Any]
    ):
        """Hook for post-execution tasks"""
        # Update context
        context.actions_taken.append(action)
        context.current_step += 1

        # Log completion
        self.logger.action_complete(
            action.action_type.value,
            {
                "step": context.current_step,
                "success": result.get("success", False),
                "items_processed": result.get("items_processed", 0),
            },
        )

    def handle_error(
        self, action: ResearchAction, context: ResearchContext, error: Exception
    ):
        """Handle execution errors"""
        self.logger.action_failed(
            action.action_type.value,
            str(error),
            {"step": context.current_step, "error_type": type(error).__name__},
        )

        # Still record the failed action
        context.actions_taken.append(action)
        context.current_step += 1
