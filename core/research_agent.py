from typing import Dict, Any, Optional, Tuple, get_origin, get_args, Union
from pydantic import BaseModel
from datetime import datetime
from openai import OpenAI
import inspect
import time

from core.data_structures import ResearchContext, ResearchAction, ActionType
from core.knowledge_accumulator import KnowledgeAccumulator
from core.actions.evaluate_executor import EvaluateExecutor
from core.actions.reflect_executor import ReflectExecutor
from core.actions.extract_executor import ExtractExecutor
from core.actions.search_executor import SearchExecutor
from core.actions.visit_executor import VisitExecutor
from core.config import get_config, ResearchConfig
from schemas.schema_manager import schema_manager
from core.action_planner import ActionPlanner
from core.search_engine import SearchEngine
from core.logging_config import get_logger
from core.url_manager import URLManager
from scraper import WebScraper


class ResearchAgent:
    """Main orchestrator for entity research"""

    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger()

        # Initialize LLM client
        self.llm_client = OpenAI(
            api_key=self.config.llm_config.api_key,
            base_url=self.config.llm_config.base_url,
        )

        # Initialize components
        self.url_manager = URLManager(self.llm_client)
        self.knowledge_accumulator = KnowledgeAccumulator(self.llm_client)
        self.action_planner = ActionPlanner(self.llm_client)
        self.search_engine = SearchEngine()
        self.web_scraper = WebScraper()

        # Initialize action executors
        self.executors = {
            ActionType.SEARCH: SearchExecutor(
                self.search_engine,
                self.url_manager,
                self.knowledge_accumulator,
                self.llm_client,
                self.config,
            ),
            ActionType.VISIT: VisitExecutor(
                self.url_manager,
                self.knowledge_accumulator,
                self.web_scraper,
                self.llm_client,
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
                entity_type, initial_urls = self._detect_entity_type(query)
            else:
                initial_urls = None

            # Get schema
            schema_class = schema_manager.get_schema(entity_type)
            if not schema_class:
                raise ValueError(f"Unknown entity type: {entity_type}")

            # Initialize research context
            context = self._initialize_context(query, entity_type, schema_class)

            # If we have initial URLs from detection, add them
            if initial_urls:
                for url in initial_urls:
                    self._add_initial_url(context, url)

            # Main research loop
            while context.should_continue_research():
                # Check if all fields of the current extraction are complete before planning next action
                if context.is_complete:
                    # All fields have values, trigger extract action
                    self.logger.info(
                        "SCHEMA_COMPLETE",
                        "All schema fields have values, triggering extraction",
                    )
                    break
                else:
                    # Decide next action normally
                    action = self.action_planner.decide_next_action(context)
                    if not action:
                        self.logger.info("RESEARCH_COMPLETE", "No more actions to take")
                        break

                # Execute action
                executor = self.executors.get(action.action_type)
                if executor:
                    result = executor.execute(action, context)
                    self._update_token_usage(context, result)

                    # Update current extraction after knowledge-gathering actions
                    if action.action_type in [
                        ActionType.VISIT,
                        ActionType.REFLECT,
                    ]:
                        self._update_current_extraction(context)
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
                        "tokens_used": context.total_tokens_used,
                        "urls_visited": len(context.visited_urls),
                    },
                )

            # Prepare results
            duration = (datetime.now() - start_time).total_seconds()
            results = self._prepare_results(context, duration)

            self.logger.section("Research Complete")
            self.logger.info(
                "RESEARCH_SUMMARY",
                f"Completed research for {query}",
                duration=duration,
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

        # # First, search for the entity
        # search_results = self.search_engine.search(query, num_results=6)

        # if not search_results:
        #     raise ValueError("No search results found for entity detection")

        # initial_urls = [search_result.url for search_result in search_results]

        # TODO: Remove mock initial urls
        initial_urls = [
            "https://www.linkedin.com/in/banghao-chi-550737276",
            "https://biboyqg.github.io/",
        ]

        visited_urls = []

        content = ""

        for idx in range(0, len(initial_urls), 2):
            # Get first URL content
            if idx < len(initial_urls):
                scrape_result1 = self.web_scraper.scrape_url(initial_urls[idx])
                content1 = scrape_result1.get("markdown")[:10000]
                if content1:
                    visited_urls.append(initial_urls[idx])
                    content += content1 + "\n\n"

            # Get second URL content
            if idx + 1 < len(initial_urls):
                scrape_result2 = self.web_scraper.scrape_url(initial_urls[idx + 1])
                content2 = scrape_result2.get("markdown")[:10000]
                if content2:
                    visited_urls.append(initial_urls[idx + 1])
                    content += content2

            if content:
                break

        # Use schema detection logic from api.py
        available_schemas = schema_manager.get_schema_names()
        ResponseOfSchema = schema_manager.get_response_of_schema()

        prompt = f"You are an expert at analyzing webpage content and determining the type of entity being described. You will analyze the content and determine if the main entity in the webpages matches one of the following schemas: {', '.join(available_schemas)}. Return your analysis as a JSON object with the matched schema name and reason. The JSON schema is: {ResponseOfSchema.model_json_schema()}. If no schema matches, return 'No match' in the schema field."

        if self.config.llm_config.enable_reasoning:
            prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

        response = self.llm_client.chat.completions.create(
            model=self.config.llm_config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": f"Analyze the webpage content and determine which schema it matches:\n{content}",
                },
            ],
            temperature=self.config.llm_config.temperature,
            extra_body={"guided_json": ResponseOfSchema.model_json_schema()},
        )

        try:
            result = ResponseOfSchema.model_validate_json(
                response.choices[0].message.content
            )
        except Exception as e:
            try:
                result = ResponseOfSchema.model_validate_json(
                    response.choices[0].message.reasoning_content
                )
            except Exception as e:
                self.logger.error(
                    "LLM_SCHEMA_DETECTION_ERROR",
                    f"Failed to detect entity type: {str(e)}",
                )
                raise ValueError("Entity type could not be determined")

        if result.schema == "No match":
            # TODO: Handle schema generation like in api.py
            raise ValueError("Entity type could not be determined")

        self.logger.info(
            "ENTITY_TYPE_DETECTED",
            f"Detected entity type: {result.schema}",
            reason=result.reason,
            url="\n".join(visited_urls),
        )

        return result.schema, initial_urls

    def _initialize_context(
        self, query: str, entity_type: str, schema_class: BaseModel
    ) -> ResearchContext:
        """Initialize research context"""
        context = ResearchContext(
            original_query=query,
            entity_type=entity_type,
            max_steps=self.config.max_steps,
            max_tokens=self.config.max_tokens_budget,
            max_urls_per_step=self.config.max_urls_per_step,
            empty_fields=self._get_all_field_names(
                self._extract_schema_fields(schema_class)
            ),
        )

        return context

    def _add_initial_url(self, context: ResearchContext, url: str):
        """Add initial URL to context"""
        from core.data_structures import URLInfo

        url_info = URLInfo(
            url=url,
            title="Initial detection page",
            metadata={"source": "entity_detection"},
        )

        context.discovered_urls[url] = url_info

    def _update_token_usage(self, context: ResearchContext, result: Dict[str, Any]):
        """Update token usage in context"""
        # Simplified - in production would track actual token usage
        estimated_tokens = result.get("items_processed", 0) * 100
        context.total_tokens_used += estimated_tokens

    def _update_current_extraction(self, context: ResearchContext):
        """Update current extraction with latest knowledge"""
        try:
            # Create a lightweight extraction action
            extract_action = ResearchAction(
                action_type=ActionType.EXTRACT,
                reason="Incremental extraction to update current state after knowledge-gathering actions",
                parameters={"incremental": True},
            )

            # Execute extraction to update current_extraction
            executor = self.executors[ActionType.EXTRACT]
            result = executor.execute(extract_action, context, skip_step=True)

            if result.get("success"):
                self.logger.debug(
                    "EXTRACTION_UPDATED",
                    "Current extraction updated with latest knowledge",
                )
            else:
                self.logger.warning(
                    "EXTRACTION_UPDATE_FAILED", "Failed to update current extraction"
                )

        except Exception as e:
            self.logger.warning(
                "EXTRACTION_UPDATE_ERROR",
                f"Error updating current extraction: {str(e)}",
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
                "knowledge_summary": self.knowledge_accumulator.generate_summary(),
            },
        }

    def _extract_schema_fields(self, schema_class: BaseModel, prefix: str = "") -> dict:
        """Extract schema fields recursively, including nested models"""
        schema_dict = {}

        for field_name, field_info in schema_class.model_fields.items():
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            field_type = field_info.annotation

            # Handle the field info
            field_data = {
                "type": str(field_type),
                "required": field_info.is_required(),
                "description": field_info.description or "",
            }

            # Check if this is a nested Pydantic model
            if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                # It's a nested Pydantic model
                field_data["nested_fields"] = self._extract_schema_fields(
                    field_type, full_field_name
                )
            else:
                # Handle generic types like List, Optional, Union
                origin = get_origin(field_type)
                args = get_args(field_type)

                if origin is list and args:
                    # Handle List[SomeModel]
                    list_type = args[0]
                    if inspect.isclass(list_type) and issubclass(list_type, BaseModel):
                        field_data["list_item_type"] = str(list_type)
                        field_data["nested_fields"] = self._extract_schema_fields(
                            list_type, f"{full_field_name}[item]"
                        )
                elif origin is Union and args:
                    # Handle Optional (Union[X, None]) and other unions
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if len(non_none_types) == 1:
                        # This is Optional[X]
                        optional_type = non_none_types[0]
                        if inspect.isclass(optional_type) and issubclass(
                            optional_type, BaseModel
                        ):
                            field_data["optional_type"] = str(optional_type)
                            field_data["nested_fields"] = self._extract_schema_fields(
                                optional_type, full_field_name
                            )

            schema_dict[full_field_name] = field_data

        return schema_dict

    def _get_all_field_names(self, schema_dict: dict) -> set:
        """Extract all field names including nested fields from schema dictionary"""
        all_fields = set()

        for field_name, field_data in schema_dict.items():
            # Add the current field
            all_fields.add(field_name)

            # If this field has nested fields, recursively add them
            if "nested_fields" in field_data:
                nested_fields = self._get_all_field_names(field_data["nested_fields"])
                all_fields.update(nested_fields)

        return all_fields
