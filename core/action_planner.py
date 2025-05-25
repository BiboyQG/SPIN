from typing import List, Optional, Dict, Any
from datetime import datetime

from core.data_structures import (
    ResearchContext,
    ResearchAction,
    ActionType,
    KnowledgeType,
    EvaluationResult,
)
from core.config import get_config
from core.logging_config import get_logger


class ActionPlanner:
    """Decides next research actions based on current state"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()

    def decide_next_action(self, context: ResearchContext) -> Optional[ResearchAction]:
        """Determine the best next action based on current research state"""
        # Check if research should continue
        if not context.should_continue_research():
            self.logger.info(
                "ACTION_PLANNING",
                "Research complete or budget exhausted",
                progress=context.get_progress_percentage(),
                tokens_used=context.total_tokens_used,
            )
            return None

        # Get candidate actions
        candidates = self._generate_candidate_actions(context)

        if not candidates:
            self.logger.warning("ACTION_PLANNING", "No candidate actions generated")
            return None

        # Score and rank actions
        scored_actions = []
        for action in candidates:
            score = self._score_action(action, context)
            scored_actions.append((score, action))

        # Sort by score (descending)
        scored_actions.sort(key=lambda x: x[0], reverse=True)

        # Select best action
        best_action = scored_actions[0][1]

        self.logger.info(
            "ACTION_SELECTED",
            f"Selected action: {best_action.action_type.value}",
            reason=best_action.reason,
            score=scored_actions[0][0],
        )

        return best_action

    def _generate_candidate_actions(
        self, context: ResearchContext
    ) -> List[ResearchAction]:
        """Generate possible actions based on current state"""
        candidates = []

        # Consider SEARCH action
        if self._should_search(context):
            search_action = self._create_search_action(context)
            if search_action:
                candidates.append(search_action)

        # Consider VISIT action
        if self._should_visit(context):
            visit_action = self._create_visit_action(context)
            if visit_action:
                candidates.append(visit_action)

        # Consider REFLECT action
        if self._should_reflect(context):
            reflect_action = self._create_reflect_action(context)
            if reflect_action:
                candidates.append(reflect_action)

        # Consider EXTRACT action
        if self._should_extract(context):
            extract_action = self._create_extract_action(context)
            if extract_action:
                candidates.append(extract_action)

        # Consider EVALUATE action
        if self._should_evaluate(context):
            evaluate_action = self._create_evaluate_action(context)
            if evaluate_action:
                candidates.append(evaluate_action)

        return candidates

    def _should_search(self, context: ResearchContext) -> bool:
        """Determine if a search action is appropriate"""
        # Don't search if we've done too many already
        if len(context.search_queries) >= self.config.max_search_queries:
            return False

        # Search if we have many empty fields and few URLs
        if len(context.empty_fields) > 3 and len(context.discovered_urls) < 10:
            return True

        # Search if we haven't searched recently
        recent_searches = sum(
            1
            for action in context.actions_taken[-5:]
            if action.action_type == ActionType.SEARCH
        )
        if recent_searches == 0 and context.current_step > 2:
            return True

        return False

    def _should_visit(self, context: ResearchContext) -> bool:
        """Determine if a visit action is appropriate"""
        # Must have unvisited URLs
        unvisited = set(context.discovered_urls.keys()) - context.visited_urls
        if not unvisited:
            return False

        # Visit if we have promising URLs
        if len(unvisited) > 0:
            return True

        return False

    def _should_reflect(self, context: ResearchContext) -> bool:
        """Determine if a reflect action is appropriate"""
        # Don't reflect too often
        recent_reflects = sum(
            1
            for action in context.actions_taken[-10:]
            if action.action_type == ActionType.REFLECT
        )
        if recent_reflects >= 2:
            return False

        # Reflect if progress is slow
        if context.current_step > 10 and context.get_progress_percentage() < 30:
            return True

        # Reflect if we have unanswered questions
        if len(context.open_questions) > 0:
            return True

        return False

    def _should_extract(self, context: ResearchContext) -> bool:
        """Determine if an extract action is appropriate"""
        # Extract if we have accumulated enough knowledge
        if len(context.knowledge_items) >= 5:
            # Check if extraction would be beneficial
            knowledge_coverage = self._calculate_knowledge_coverage(context)
            if knowledge_coverage > 0.3:  # At least 30% potential coverage
                return True

        return False

    def _should_evaluate(self, context: ResearchContext) -> bool:
        """Determine if an evaluate action is appropriate"""
        # Evaluate periodically
        if context.current_step % 10 == 0 and context.current_step > 0:
            return True

        # Evaluate if we think we're close to complete
        if context.get_progress_percentage() > 80:
            return True

        return False

    def _create_search_action(
        self, context: ResearchContext
    ) -> Optional[ResearchAction]:
        """Create a search action"""
        # Determine what to search for
        if context.empty_fields:
            # Search for specific empty fields
            target_field = list(context.empty_fields)[0]
            reason = f"Searching for information about {target_field}"
            parameters = {"target_field": target_field, "query_type": "field_specific"}
        else:
            # General search
            reason = "Performing general search for more information"
            parameters = {"query_type": "general"}

        return ResearchAction(
            action_type=ActionType.SEARCH,
            reason=reason,
            parameters=parameters,
            priority=0.8,
            estimated_cost=100,  # Tokens
        )

    def _create_visit_action(
        self, context: ResearchContext
    ) -> Optional[ResearchAction]:
        """Create a visit action"""
        # Get unvisited URLs
        unvisited = set(context.discovered_urls.keys()) - context.visited_urls
        if not unvisited:
            return None

        # Select URLs to visit (this is simplified - actual implementation would rank them)
        urls_to_visit = list(unvisited)[: self.config.max_urls_per_step]

        return ResearchAction(
            action_type=ActionType.VISIT,
            reason=f"Visiting {len(urls_to_visit)} promising URLs",
            parameters={"urls": urls_to_visit},
            priority=0.9,
            estimated_cost=500 * len(urls_to_visit),  # Tokens per URL
        )

    def _create_reflect_action(
        self, context: ResearchContext
    ) -> Optional[ResearchAction]:
        """Create a reflect action"""
        return ResearchAction(
            action_type=ActionType.REFLECT,
            reason="Reflecting on progress and identifying knowledge gaps",
            parameters={"focus_areas": list(context.empty_fields)[:5]},
            priority=0.6,
            estimated_cost=300,
        )

    def _create_extract_action(
        self, context: ResearchContext
    ) -> Optional[ResearchAction]:
        """Create an extract action"""
        return ResearchAction(
            action_type=ActionType.EXTRACT,
            reason="Extracting structured information from accumulated knowledge",
            parameters={"knowledge_count": len(context.knowledge_items)},
            priority=0.7,
            estimated_cost=1000,
        )

    def _create_evaluate_action(
        self, context: ResearchContext
    ) -> Optional[ResearchAction]:
        """Create an evaluate action"""
        return ResearchAction(
            action_type=ActionType.EVALUATE,
            reason="Evaluating current extraction quality and completeness",
            parameters={"evaluation_type": "comprehensive"},
            priority=0.5,
            estimated_cost=500,
        )

    def _score_action(self, action: ResearchAction, context: ResearchContext) -> float:
        """Score an action based on expected value"""
        score = action.priority

        # Adjust based on action type and context
        if action.action_type == ActionType.SEARCH:
            # Boost if we have many empty fields
            empty_ratio = (
                len(context.empty_fields) / len(context.schema) if context.schema else 0
            )
            score += empty_ratio * 0.2

        elif action.action_type == ActionType.VISIT:
            # Boost if we have high-quality unvisited URLs
            score += 0.1  # Simplified - would check URL quality

        elif action.action_type == ActionType.REFLECT:
            # Boost if progress is slow
            if context.get_progress_percentage() < 30 and context.current_step > 5:
                score += 0.2

        elif action.action_type == ActionType.EXTRACT:
            # Boost if we have good knowledge coverage
            coverage = self._calculate_knowledge_coverage(context)
            score += coverage * 0.3

        elif action.action_type == ActionType.EVALUATE:
            # Boost if we're near completion
            if context.get_progress_percentage() > 70:
                score += 0.2

        # Penalize based on cost
        cost_factor = action.estimated_cost / 1000  # Normalize to 0-1 range
        score -= cost_factor * 0.1

        # Penalize if budget is running low
        budget_used = context.total_tokens_used / context.max_tokens
        if budget_used > 0.8:
            score -= (budget_used - 0.8) * 0.5

        return max(0, min(1, score))  # Clamp to [0, 1]

    def _calculate_knowledge_coverage(self, context: ResearchContext) -> float:
        """Calculate how much schema coverage we have from knowledge"""
        if not context.schema:
            return 0.0

        covered_fields = set()
        for item in context.knowledge_items:
            covered_fields.update(item.schema_fields)

        coverage = len(covered_fields) / len(context.schema)
        return coverage

    def get_action_history_summary(self, context: ResearchContext) -> Dict[str, Any]:
        """Get summary of action history"""
        action_counts = {}
        for action in context.actions_taken:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        return {
            "total_actions": len(context.actions_taken),
            "action_counts": action_counts,
            "current_step": context.current_step,
            "progress": context.get_progress_percentage(),
        }
