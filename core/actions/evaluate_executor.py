from typing import Dict, Any, List
from datetime import datetime

from core.actions.base import ActionExecutor
from core.data_structures import ResearchContext, ResearchAction, EvaluationResult
from core.knowledge_accumulator import KnowledgeAccumulator


class EvaluateExecutor(ActionExecutor):
    """Executes EVALUATE actions to assess research quality"""

    def __init__(self, knowledge_accumulator: KnowledgeAccumulator):
        super().__init__()
        self.knowledge_accumulator = knowledge_accumulator

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute an evaluate action"""
        self.pre_execute(action, context)

        try:
            # Perform evaluation
            evaluation = self._evaluate_research(context)

            # Update context metrics
            context.completeness_score = evaluation.completeness
            context.confidence_score = evaluation.overall_score

            # Determine if research should continue
            if evaluation.completeness >= self.config.min_completeness_for_finish:
                context.is_complete = True
                context.completion_reason = "Completeness threshold reached"

            result = {
                "success": True,
                "evaluation": evaluation.__dict__,
                "should_continue": evaluation.should_continue,
                "items_processed": 1,
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _evaluate_research(self, context: ResearchContext) -> EvaluationResult:
        """Evaluate the current state of research"""
        # Calculate completeness
        completeness = context.get_progress_percentage() / 100.0

        # Calculate accuracy based on knowledge confidence
        total_confidence = 0
        for item in context.knowledge_items:
            total_confidence += item.confidence
        accuracy = (
            total_confidence / len(context.knowledge_items)
            if context.knowledge_items
            else 0
        )

        # Calculate consistency (simplified)
        consistency = self._calculate_consistency(context)

        # Calculate freshness (simplified - assume all recent)
        freshness = 0.9

        # Overall score
        overall_score = (
            completeness * 0.4 + accuracy * 0.3 + consistency * 0.2 + freshness * 0.1
        )

        # Identify missing critical fields
        missing_critical = self._identify_critical_missing_fields(context)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            context, completeness, accuracy, missing_critical
        )

        # Determine if should continue
        should_continue = (
            overall_score < self.config.min_completeness_for_finish
            and context.should_continue_research()
        )

        return EvaluationResult(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            freshness=freshness,
            overall_score=overall_score,
            missing_critical_fields=missing_critical,
            recommendations=recommendations,
            should_continue=should_continue,
            reasoning=self._generate_reasoning(
                completeness, accuracy, missing_critical
            ),
        )

    def _calculate_consistency(self, context: ResearchContext) -> float:
        """Calculate consistency of information"""
        if not context.filled_fields:
            return 0.0

        # Check how many fields have consistent information
        consistent_fields = 0
        for field in context.filled_fields:
            field_items = self.knowledge_accumulator.get_knowledge_for_field(field)
            if len(field_items) <= 1:
                consistent_fields += 1
            else:
                # Check if answers are similar
                groups = self.knowledge_accumulator._group_similar_answers(field_items)
                if len(groups) == 1:
                    consistent_fields += 1

        return consistent_fields / len(context.filled_fields)

    def _identify_critical_missing_fields(self, context: ResearchContext) -> List[str]:
        """Identify critical fields that are still missing"""
        # Define critical fields based on entity type
        critical_fields_map = {
            "professor": ["name", "title", "department", "email"],
            "student": ["name", "major", "email", "year"],
            "company": ["name", "industry", "website", "headquarters"],
        }

        critical_fields = critical_fields_map.get(context.entity_type, [])
        missing_critical = [
            field for field in critical_fields if field in context.empty_fields
        ]

        return missing_critical

    def _generate_recommendations(
        self,
        context: ResearchContext,
        completeness: float,
        accuracy: float,
        missing_critical: List[str],
    ) -> List[str]:
        """Generate recommendations for improving research"""
        recommendations = []

        if completeness < 0.5:
            recommendations.append("Expand search queries to find more information")

        if accuracy < 0.7:
            recommendations.append(
                "Visit more authoritative sources to improve accuracy"
            )

        if missing_critical:
            recommendations.append(
                f"Focus on finding critical missing fields: {', '.join(missing_critical)}"
            )

        if len(context.visited_urls) < 5:
            recommendations.append(
                "Visit more URLs to gather comprehensive information"
            )

        return recommendations

    def _generate_reasoning(
        self, completeness: float, accuracy: float, missing_critical: List[str]
    ) -> str:
        """Generate reasoning for the evaluation"""
        parts = []

        if completeness >= 0.9:
            parts.append("Research has achieved high completeness")
        elif completeness >= 0.7:
            parts.append("Research has good coverage but some gaps remain")
        else:
            parts.append("Research needs more work to fill information gaps")

        if accuracy >= 0.8:
            parts.append("Information gathered appears highly reliable")
        elif accuracy >= 0.6:
            parts.append("Information has moderate reliability")
        else:
            parts.append("Information reliability needs improvement")

        if missing_critical:
            parts.append(
                f"Critical fields still missing: {', '.join(missing_critical)}"
            )

        return ". ".join(parts)
