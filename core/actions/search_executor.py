from typing import Dict, Any, List
import time

from core.knowledge_accumulator import KnowledgeAccumulator
from core.response_model import ResponseOfSearchQueries
from core.actions.base import ActionExecutor
from core.search_engine import SearchEngine
from core.url_manager import URLManager
from core.data_structures import (
    ResearchContext,
    ResearchAction,
)
from openai import OpenAI


class SearchExecutor(ActionExecutor):
    """Executes SEARCH actions"""

    def __init__(
        self,
        search_engine: SearchEngine,
        url_manager: URLManager,
        knowledge_accumulator: KnowledgeAccumulator,
        llm_client: OpenAI,
        config: Any,
    ):
        super().__init__()
        self.search_engine = search_engine
        self.url_manager = url_manager
        self.knowledge_accumulator = knowledge_accumulator
        self.llm_client = llm_client
        self.config = config

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute a search action"""
        self.pre_execute(action, context)

        try:
            # Generate search queries
            queries = self._generate_queries(action, context)

            # Execute searches
            all_results = []
            for query in queries:
                results = self.search_engine.search(query)
                time.sleep(1)  # Rate limit 1 second between searches
                all_results.extend(results)
                context.search_queries.append(query)

            # Process search results
            discovered_urls = self.url_manager.discover_urls_from_search(all_results)

            result = {
                "success": True,
                "queries_executed": queries,
                "results_found": len(all_results),
                "new_urls_discovered": len(discovered_urls),
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _generate_queries(
        self, action: ResearchAction, context: ResearchContext
    ) -> List[str]:
        """Generate search queries based on action parameters using an LLM"""
        try:
            # Construct prompt for LLM
            prompt_parts = [
                "You are an expert research assistant tasked with generating effective search queries.",
                f"Research Objective: {context.original_query}",
                f"Entity Type: {context.entity_type}",
                f"Current information we have: {context.current_extraction}",
                f"Empty fields we need to fill: {list(context.empty_fields)}",
            ]

            if action.parameters.get("query_type") == "field_specific":
                target_fields = action.parameters.get("target_fields")
                if target_fields:
                    prompt_parts.append(
                        f"Focus on generating queries to find information for the fields: '{', '.join(target_fields)}'. You should generate one query for each field above."
                    )
                else:
                    prompt_parts.append(
                        "This is a field-specific search, but no target fields were specified. Generate general queries related to the research objective and empty fields."
                    )
            else:
                prompt_parts.append(
                    "Generate general search queries to achieve the research objective and fill the empty fields."
                )

            prompt_parts.append(
                " The queries should be specific to the field and the research objective. For example, if the research objective is 'Minjia Zhang UIUC', then your generated queries should be something like 'Minjia Zhang UIUC {field_name} or more detailed things that you want to know about the field (in case of nested fields)'"
            )
            prompt_parts.append(
                "Return your response as a JSON object with a 'queries' field containing a list of strings (the search queries), and an optional 'reasoning' field explaining your choices."
            )

            prompt = "\n\n".join(prompt_parts)

            if self.config.llm_config.enable_reasoning:
                prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research assistant that generates search queries.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfSearchQueries.model_json_schema()},
            )

            try:
                result = ResponseOfSearchQueries.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                if response.choices[0].message.reasoning_content:
                    try:
                        result = ResponseOfSearchQueries.model_validate_json(
                            response.choices[0].message.reasoning_content
                        )
                    except Exception as e_reasoning:
                        self.logger.warning(
                            "SEARCH_QUERY_PARSE_ERROR_REASONING",
                            f"Failed to parse LLM reasoning_content for search query generation: {e_reasoning}. Content: {response.choices[0].message.reasoning_content}",
                        )
                        return [context.original_query]
                else:
                    return [context.original_query]

            queries = result.queries

            if not queries:
                self.logger.warning(
                    "LLM_EMPTY_QUERIES",
                    "LLM returned no search queries. Using original query as fallback.",
                )
                queries = [context.original_query]

            # Limit number of queries
            return queries[: self.config.max_search_queries]

        except Exception as e:
            self.logger.error(
                "LLM_QUERY_GENERATION_ERROR",
                f"Error during LLM query generation: {e}",
                exc_info=True,
            )
            self.logger.warning(
                "LLM_QUERY_GENERATION_FALLBACK",
                "Falling back to original query due to LLM error.",
            )
            return [context.original_query][: self.config.max_search_queries]
