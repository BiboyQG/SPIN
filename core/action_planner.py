from typing import Optional, Dict, Any
from openai import OpenAI

from core.data_structures import (
    ResearchContext,
    ResearchAction,
    ActionType,
)
from core.config import get_config
from core.logging_config import get_logger
from core.response_model import ResponseOfActionPlan, Action


class ActionPlanner:
    """Decides next research actions using LLM-based planning"""

    def __init__(self, llm_client: OpenAI):
        self.config = get_config()
        self.logger = get_logger()
        self.llm_client = llm_client

    def decide_next_action(self, context: ResearchContext) -> Optional[ResearchAction]:
        """Determine the best next action using LLM reasoning"""
        # Check if research should continue
        if not context.should_continue_research():
            self.logger.info(
                "ACTION_PLANNING",
                "Research complete or budget exhausted",
                progress=context.get_progress_percentage(),
                tokens_used=context.total_tokens_used,
            )
            return None

        # Get LLM to plan the next action
        action_plan = self._get_llm_action_plan(context)

        if not action_plan:
            self.logger.warning("ACTION_PLANNING", "LLM failed to generate action plan")
            return None

        # Convert LLM response to ResearchAction
        research_action = self._convert_to_research_action(action_plan, context)

        self.logger.info(
            "ACTION_SELECTED",
            f"Selected action: {research_action.action_type.value}",
            reason=research_action.reason,
        )

        return research_action

    def _get_llm_action_plan(
        self, context: ResearchContext
    ) -> Optional[ResponseOfActionPlan]:
        """Use LLM to determine the next action"""
        # Prepare context summary
        context_summary = self._prepare_context_summary(context)

        # Analyze recent actions
        recent_actions_analysis = self._analyze_recent_actions(context)

        prompt = f"""You are an intelligent research agent planning the next action.

Research Goal: {context.original_query}
Entity Type: {context.entity_type}

Current Progress:
- Fields filled: {len(context.filled_fields)}/{len(context.schema.keys())}
- URLs unvisited in the next step: {
            len(
                list(
                    set(context.discovered_urls.keys())
                    - context.visited_urls
                    - context.failed_urls
                )[: self.config.max_urls_per_step]
            )
        }
- URLs visited: {len(context.visited_urls)}
- Knowledge items collected: {len(context.knowledge_items)}
- Current step: {context.current_step}

Empty fields remaining: {list(context.empty_fields)[:10]}

Recent Actions Analysis:
{recent_actions_analysis}

Context Summary:
{context_summary}

Available actions:
1. SEARCH - Search for new information when you need more URLs or specific information
2. VISIT - Visit unvisited URLs to extract information
3. REFLECT - Analyze progress and identify knowledge gaps
4. EVALUATE - Assess the quality and completeness of current extraction

Based on the current state and recent action results, choose the SINGLE BEST action to take next.
Consider:
- If recent visit actions failed repeatedly for certain URLs, it may be a scraping issue - move on
- If we have many unvisited URLs with good potential, prioritize visiting them
- If progress is slow or we're stuck, reflect to identify new strategies
- If we have substantial information, evaluate to check completeness
- Search when we need more sources or specific information for empty fields

Choose one action and provide a clear reason and return your response in a JSON object, following schema:
{ResponseOfActionPlan.model_json_schema()}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research planning assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfActionPlan.model_json_schema()},
            )

            if self.config.llm_config.enable_reasoning:
                reasoning_content = response.choices[0].message.reasoning_content
                print("=" * 80)
                print("Reasoning content:\n\n")
                print(reasoning_content)
                print("=" * 80)

            try:
                result = ResponseOfActionPlan.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                print("=" * 80)
                print("Error:\n\n")
                print(e)
                print("=" * 80)
                result = ResponseOfActionPlan.model_validate_json(
                    response.choices[0].message.reasoning_content
                )

            return result

        except Exception as e:
            self.logger.error(
                "LLM_PLANNING_ERROR", f"Failed to get action plan: {str(e)}"
            )
            return None

    def _prepare_context_summary(self, context: ResearchContext) -> str:
        """Prepare a concise summary of the current context"""
        summary_parts = []

        # Knowledge summary
        if context.knowledge_items:
            fields_covered = set()
            for item in context.knowledge_items:
                fields_covered.update(item.schema_fields)
            summary_parts.append(
                f"Knowledge collected for fields: {list(fields_covered)}"
            )

        # URL status
        unvisited = (
            set(context.discovered_urls.keys())
            - context.visited_urls
            - context.failed_urls
        )
        if unvisited:
            summary_parts.append(f"{len(unvisited)} unvisited URLs available")
            # Show sample of unvisited URLs
            sample_urls = list(unvisited)[:3]
            for url in sample_urls:
                url_info = context.discovered_urls[url]
                summary_parts.append(f"  - {url_info.title or 'No title'}: {url}")

        # Failed URLs
        if context.failed_urls:
            summary_parts.append(f"{len(context.failed_urls)} URLs failed to scrape")

        # Open questions
        if context.open_questions:
            summary_parts.append(f"Open questions: {context.open_questions[:3]}")

        return "\n".join(summary_parts)

    def _analyze_recent_actions(self, context: ResearchContext) -> str:
        """Analyze recent actions to identify patterns and issues"""
        if not context.actions_taken:
            return "No previous actions taken."

        # Look at last 5 actions
        recent_actions = context.actions_taken[-5:]
        analysis_parts = []

        # Count action types
        action_counts = {}
        for action in recent_actions:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        analysis_parts.append(f"Recent action distribution: {action_counts}")

        # Check for repeated failures
        if context.failed_urls:
            # Count recent visit failures
            recent_visit_actions = [
                a for a in recent_actions if a.action_type == ActionType.VISIT
            ]
            if recent_visit_actions:
                failed_in_recent = 0
                for action in recent_visit_actions:
                    if "urls" in action.parameters:
                        for url in action.parameters["urls"]:
                            if url in context.failed_urls:
                                failed_in_recent += 1

                if failed_in_recent > 2:
                    analysis_parts.append(
                        f"WARNING: {failed_in_recent} recent visit attempts failed - possible scraping issues"
                    )

        # Check if stuck in a pattern
        if len(recent_actions) >= 3:
            recent_types = [a.action_type.value for a in recent_actions[-3:]]
            if len(set(recent_types)) == 1:
                analysis_parts.append(
                    f"Pattern detected: Repeating {recent_types[0]} actions"
                )

        # Progress check
        if context.current_step > 10:
            progress_rate = len(context.filled_fields) / context.current_step
            if progress_rate < 0.5:
                analysis_parts.append(
                    "Slow progress detected - consider changing strategy"
                )

        return "\n".join(analysis_parts)

    def _convert_to_research_action(
        self, action_plan: ResponseOfActionPlan, context: ResearchContext
    ) -> ResearchAction:
        """Convert LLM action plan to ResearchAction"""
        action_type_map = {
            Action.search: ActionType.SEARCH,
            Action.visit: ActionType.VISIT,
            Action.reflect: ActionType.REFLECT,
            Action.evaluate: ActionType.EVALUATE,
        }

        action_type = action_type_map[action_plan.action]

        # Prepare parameters based on action type
        parameters = {}

        if action_type == ActionType.SEARCH:
            # Determine search focus
            if context.empty_fields:
                target_fields = list(context.empty_fields)[:3]
                parameters = {
                    "target_fields": target_fields,
                    "query_type": "field_specific",
                }
            else:
                parameters = {"query_type": "general"}

        elif action_type == ActionType.VISIT:
            # Select unvisited URLs
            unvisited = (
                set(context.discovered_urls.keys())
                - context.visited_urls
                - context.failed_urls
            )
            urls_to_visit = list(unvisited)[: self.config.max_urls_per_step]
            parameters = {"urls": urls_to_visit}

        elif action_type == ActionType.REFLECT:
            # Focus on empty fields
            parameters = {"focus_areas": list(context.empty_fields)[:5]}

        elif action_type == ActionType.EVALUATE:
            parameters = {"evaluation_type": "comprehensive"}

        return ResearchAction(
            action_type=action_type,
            reason=action_plan.reason,
            parameters=parameters,
            priority=1.0,  # LLM has already prioritized
            estimated_cost=self._estimate_action_cost(action_type),
        )

    def _estimate_action_cost(self, action_type: ActionType) -> float:
        """Estimate token cost for an action"""
        cost_map = {
            ActionType.SEARCH: 0,
            ActionType.VISIT: 0,
            ActionType.REFLECT: 0,
            ActionType.EVALUATE: 0,
            ActionType.EXTRACT: 0,
        }
        return cost_map.get(action_type, 0)

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
