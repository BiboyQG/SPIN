from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json
import time
from openai import OpenAI

from core.data_structures import ResearchContext, ActionType
from core.config import get_config, ResearchConfig
from core.logging_config import get_logger
from core.url_manager import URLManager
from core.knowledge_accumulator import KnowledgeAccumulator
from core.search_engine import SearchEngine
from core.action_planner import ActionPlanner
from core.actions.search_executor import SearchExecutor
from core.actions.visit_executor import VisitExecutor
from core.actions.reflect_executor import ReflectExecutor
from core.actions.extract_executor import ExtractExecutor
from core.actions.evaluate_executor import EvaluateExecutor
from schemas.schema_manager import schema_manager
from scraper import WebScraper


class ResearchAgent:
    """Main orchestrator for entity research"""

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger()

        # Initialize components
        self.url_manager = URLManager()
        self.knowledge_accumulator = KnowledgeAccumulator()
        self.search_engine = SearchEngine()
        self.action_planner = ActionPlanner()
        self.web_scraper = WebScraper()

        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=self.config.llm_config.api_key,
            base_url=self.config.llm_config.base_url,
        )

        # Initialize action executors
        self.executors = {
            ActionType.SEARCH: SearchExecutor(
                self.search_engine, self.url_manager, self.knowledge_accumulator
            ),
            ActionType.VISIT: VisitExecutor(
                self.url_manager, self.knowledge_accumulator, self.web_scraper
            ),
            ActionType.REFLECT: ReflectExecutor(
                self.knowledge_accumulator, self.llm_client
            ),
            ActionType.EXTRACT: ExtractExecutor(
                self.knowledge_accumulator, self.llm_client
            ),
            ActionType.EVALUATE: EvaluateExecutor(self.knowledge_accumulator),
        }

    def research_entity(
        self, query: str, entity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main research orchestration function

        Args:
            query: Entity description (e.g., "John Smith professor MIT")
            entity_type: Optional entity type. If not provided, will be detected

        Returns:
            Dictionary containing extracted data and research metadata
        """
        self.logger.section(f"Starting Research: {query}")
        start_time = datetime.now()

        try:
            # Detect entity type and schema if not provided
            if not entity_type:
                entity_type, initial_url = self._detect_entity_type(query)
            else:
                initial_url = None

            # Get schema
            schema_class = schema_manager.get_schema(entity_type)
            if not schema_class:
                raise ValueError(f"Unknown entity type: {entity_type}")

            # Initialize research context
            context = self._initialize_context(query, entity_type, schema_class)

            # If we have an initial URL from detection, add it
            if initial_url:
                self._add_initial_url(context, initial_url)

            # Main research loop
            while context.should_continue_research():
                # Decide next action
                action = self.action_planner.decide_next_action(context)
                if not action:
                    self.logger.info("RESEARCH_COMPLETE", "No more actions to take")
                    break

                # Execute action
                executor = self.executors.get(action.action_type)
                if executor:
                    result = executor.execute(action, context)
                    self._update_token_usage(context, result)
                else:
                    self.logger.error(
                        "UNKNOWN_ACTION",
                        f"No executor for action type: {action.action_type}",
                    )

                # Add delay between actions
                time.sleep(self.config.step_delay)

                # Progress update
                self.logger.progress(
                    context.current_step,
                    context.max_steps,
                    f"Completed {action.action_type.value} action",
                    {
                        "progress": context.get_progress_percentage(),
                        "tokens_used": context.total_tokens_used,
                        "urls_visited": len(context.visited_urls),
                    },
                )

            # Final extraction if needed
            if (
                not context.current_extraction
                or context.get_progress_percentage() < 100
            ):
                self._perform_final_extraction(context)

            # Prepare results
            duration = (datetime.now() - start_time).total_seconds()
            results = self._prepare_results(context, duration)

            self.logger.section("Research Complete")
            self.logger.info(
                "RESEARCH_SUMMARY",
                f"Completed research for {query}",
                duration=duration,
                fields_filled=len(context.filled_fields),
                total_fields=len(context.schema),
                urls_visited=len(context.visited_urls),
                knowledge_items=len(context.knowledge_items),
            )

            return results

        except Exception as e:
            self.logger.error(
                "RESEARCH_FAILED", f"Research failed for query: {query}", error=str(e)
            )
            raise

    def _detect_entity_type(self, query: str) -> Tuple[str, Optional[str]]:
        """Detect entity type from query using search and initial page analysis"""
        self.logger.subsection("Detecting Entity Type")

        # First, search for the entity
        search_results = self.search_engine.search(query, num_results=1)

        if not search_results:
            raise ValueError("No search results found for entity detection")

        # Get the top URL
        top_url = search_results[0].url

        # Scrape the page
        scrape_result = self.web_scraper.scrape_url(top_url)
        content = scrape_result.get("markdown")

        if not content:
            raise ValueError("Failed to scrape content for entity detection")

        # Use schema detection logic from api.py
        available_schemas = schema_manager.get_schema_names()
        ResponseOfSchema = schema_manager.get_response_of_schema()

        response = self.llm_client.chat.completions.create(
            model=self.config.llm_config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an expert at analyzing webpage content and determining the type of entity being described. You will analyze the content and determine if it matches one of the following schemas: {', '.join(available_schemas)}. Return your analysis as a JSON object with the matched schema name and reason. If no schema matches, return 'No match'.",
                },
                {
                    "role": "user",
                    "content": f"Analyze this webpage content and determine which schema it matches:\n{content[:5000]}",  # Limit content size
                },
            ],
            temperature=0.0,
            extra_body={"guided_json": ResponseOfSchema.model_json_schema()},
        )

        result = ResponseOfSchema.model_validate_json(
            response.choices[0].message.content
        )

        if result.schema == "No match":
            # TODO: Handle schema generation like in api.py
            raise ValueError("Entity type could not be determined")

        self.logger.info(
            "ENTITY_TYPE_DETECTED",
            f"Detected entity type: {result.schema}",
            reason=result.reason,
            url=top_url,
        )

        return result.schema, top_url

    def _initialize_context(
        self, query: str, entity_type: str, schema_class: type
    ) -> ResearchContext:
        """Initialize research context"""
        # Get schema fields
        schema_dict = {}
        for field_name, field_info in schema_class.model_fields.items():
            schema_dict[field_name] = {
                "type": str(field_info.annotation),
                "required": field_info.is_required(),
                "description": field_info.description or "",
            }

        context = ResearchContext(
            original_query=query,
            entity_type=entity_type,
            schema=schema_dict,
            max_steps=self.config.max_steps,
            max_tokens=self.config.max_tokens_budget,
            max_urls_per_step=self.config.max_urls_per_step,
        )

        # Initialize empty fields
        context.empty_fields = set(schema_dict.keys())

        return context

    def _add_initial_url(self, context: ResearchContext, url: str):
        """Add initial URL to context"""
        from core.data_structures import URLInfo

        url_info = URLInfo(
            url=url,
            title="Initial detection page",
            relevance_score=1.0,
            schema_fields_coverage=list(context.schema.keys()),
            metadata={"source": "entity_detection"},
        )

        context.discovered_urls[url] = url_info

    def _update_token_usage(self, context: ResearchContext, result: Dict[str, Any]):
        """Update token usage in context"""
        # Simplified - in production would track actual token usage
        estimated_tokens = result.get("items_processed", 0) * 100
        context.total_tokens_used += estimated_tokens

    def _perform_final_extraction(self, context: ResearchContext):
        """Perform final extraction if not done yet"""
        self.logger.subsection("Performing Final Extraction")

        from core.data_structures import ResearchAction

        extract_action = ResearchAction(
            action_type=ActionType.EXTRACT,
            reason="Final extraction to consolidate all findings",
            parameters={"final": True},
        )

        executor = self.executors[ActionType.EXTRACT]
        result = executor.execute(extract_action, context)

        if not result["success"]:
            self.logger.warning(
                "FINAL_EXTRACTION_FAILED",
                "Final extraction failed, using partial results",
            )

    def _prepare_results(
        self, context: ResearchContext, duration: float
    ) -> Dict[str, Any]:
        """Prepare final results"""
        return {
            "success": True,
            "entity_type": context.entity_type,
            "query": context.original_query,
            "extracted_data": context.current_extraction,
            "metadata": {
                "duration_seconds": duration,
                "fields_filled": len(context.filled_fields),
                "total_fields": len(context.schema),
                "completeness": context.get_progress_percentage(),
                "confidence_score": context.confidence_score,
                "urls_visited": len(context.visited_urls),
                "urls_discovered": len(context.discovered_urls),
                "knowledge_items": len(context.knowledge_items),
                "search_queries": len(context.search_queries),
                "total_steps": context.current_step,
                "tokens_used": context.total_tokens_used,
                "completion_reason": context.completion_reason or "Max steps reached",
            },
            "research_trail": {
                "actions_taken": [
                    {
                        "type": action.action_type.value,
                        "reason": action.reason,
                        "step": idx + 1,
                    }
                    for idx, action in enumerate(context.actions_taken)
                ],
                "search_queries": context.search_queries,
                "visited_urls": list(context.visited_urls),
                "knowledge_summary": self.knowledge_accumulator.generate_summary(),
            },
        }
