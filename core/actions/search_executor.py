from typing import Dict, Any, List
from datetime import datetime

from core.actions.base import ActionExecutor
from core.data_structures import (
    ResearchContext,
    ResearchAction,
    KnowledgeItem,
    KnowledgeType,
)
from core.search_engine import SearchEngine
from core.url_manager import URLManager
from core.knowledge_accumulator import KnowledgeAccumulator


class SearchExecutor(ActionExecutor):
    """Executes SEARCH actions"""

    def __init__(
        self,
        search_engine: SearchEngine,
        url_manager: URLManager,
        knowledge_accumulator: KnowledgeAccumulator,
    ):
        super().__init__()
        self.search_engine = search_engine
        self.url_manager = url_manager
        self.knowledge_accumulator = knowledge_accumulator

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
                all_results.extend(results)
                context.search_queries.append(query)

            # Process search results
            discovered_urls = self.url_manager.discover_urls_from_search(all_results)

            # Create knowledge items from search snippets
            for result in all_results:
                knowledge_item = KnowledgeItem(
                    question=f"What does the search result say about {context.original_query}?",
                    answer=result.snippet,
                    source_urls=[result.url],
                    confidence=result.relevance_score
                    * 0.5,  # Lower confidence for snippets
                    timestamp=datetime.now(),
                    item_type=KnowledgeType.SEARCH_RESULT,
                    schema_fields=self._identify_related_fields(
                        result.snippet, context
                    ),
                )
                self.knowledge_accumulator.add_knowledge(knowledge_item)

            result = {
                "success": True,
                "queries_executed": queries,
                "results_found": len(all_results),
                "new_urls_discovered": len(discovered_urls),
                "items_processed": len(all_results),
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _generate_queries(
        self, action: ResearchAction, context: ResearchContext
    ) -> List[str]:
        """Generate search queries based on action parameters"""
        queries = []

        if action.parameters.get("query_type") == "field_specific":
            # Generate field-specific queries
            target_field = action.parameters.get("target_field")
            if target_field:
                base_query = context.original_query
                field_query = self.search_engine.rewrite_query_for_field(
                    base_query, target_field
                )
                queries.append(field_query)

                # Add variations
                queries.extend(
                    self.search_engine.generate_queries(
                        field_query, {"empty_fields": [target_field]}
                    )
                )
        else:
            # Generate general queries
            queries = self.search_engine.generate_queries(
                context.original_query,
                {
                    "empty_fields": list(context.empty_fields)[:3],
                    "entity_type": context.entity_type,
                },
            )

        # Limit number of queries
        return queries[: self.config.max_search_queries]

    def _identify_related_fields(
        self, text: str, context: ResearchContext
    ) -> List[str]:
        """Identify which schema fields a text snippet might relate to"""
        related_fields = []

        # Simple keyword matching (in production, would use NLP)
        text_lower = text.lower()

        for field in context.schema.keys():
            # Convert field name to readable format
            field_words = field.replace("_", " ").lower().split()

            # Check if field words appear in text
            if any(word in text_lower for word in field_words):
                related_fields.append(field)

        return related_fields
