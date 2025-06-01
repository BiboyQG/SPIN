from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

from core.data_structures import KnowledgeItem, ResearchContext
from core.response_model import ResponseOfConsolidation
from core.logging_config import get_logger
from core.config import get_config


class KnowledgeAccumulator:
    """Manages research findings and knowledge building using LLM-based consolidation"""

    def __init__(self, llm_client: OpenAI):
        self.config = get_config()
        self.logger = get_logger()
        self.llm_client = llm_client

        # Simple field -> value mapping with sources
        self.field_values: Dict[str, Dict[str, Any]] = {}
        # field_values structure: {
        #     "field_name": {
        #         "value": "actual value",
        #         "sources": ["url1", "url2"],
        #         "last_updated": datetime
        #     }
        # }

        # Raw knowledge items for reference
        self.knowledge_items: List[KnowledgeItem] = []

        # Track all discovered information by field
        self.field_discoveries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # field_discoveries structure: {
        #     "field_name": [
        #         {"value": "v1", "source": "url1", "context": "..."},
        #         {"value": "v2", "source": "url2", "context": "..."}
        #     ]
        # }

    def add_knowledge(
        self, item: KnowledgeItem, context: ResearchContext = None
    ) -> None:
        """Add a knowledge item and update field values"""
        # Store raw knowledge item
        self.knowledge_items.append(item)

        # Extract field-specific information
        self._extract_field_values(item, context)

        self.logger.info(
            "KNOWLEDGE_ADDED",
            f"Added knowledge item of type {item.item_type.value}",
            fields=item.schema_fields,
            sources=item.source_urls[:2],  # Log first 2 sources
        )

    def _extract_field_values(
        self, item: KnowledgeItem, context: ResearchContext = None
    ) -> None:
        """Extract field values from a knowledge item"""
        # For each field this knowledge item relates to
        for field in item.schema_fields:
            # Store the discovery
            self.field_discoveries[field].append(
                {
                    "value": item.answer,
                    "source": item.source_urls[0] if item.source_urls else "unknown",
                    "context": item.question,
                    "timestamp": item.timestamp,
                }
            )

            # Update consolidated field value if needed
            self._update_field_value(field, context)

    def _update_field_value(
        self, field_name: str, context: ResearchContext = None
    ) -> None:
        """Update the consolidated value for a field using LLM"""
        discoveries = self.field_discoveries.get(field_name, [])
        if not discoveries:
            return

        # If only one discovery, use it directly
        if len(discoveries) == 1:
            self.field_values[field_name] = {
                "value": discoveries[0]["value"],
                "sources": [discoveries[0]["source"]],
                "last_updated": datetime.now(),
            }
            return

        # Use LLM to consolidate multiple discoveries
        consolidated = self._llm_consolidate_field(field_name, discoveries)
        if consolidated:
            self.field_values[field_name] = consolidated

    def _llm_consolidate_field(
        self, field_name: str, discoveries: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to consolidate multiple discoveries for a field"""
        # Prepare discoveries for prompt
        discovery_text = ""
        for i, disc in enumerate(discoveries, 1):
            discovery_text += f"\n{i}. Value: {disc['value']}\n   Source: {disc['source']}\n   Context: {disc['context']}\n"

        prompt = f"""You are consolidating information for the field '{field_name}'.

Multiple sources have provided the following information:
{discovery_text}

Based on these sources, determine:
1. The most accurate/complete information for this field
2. Which sources support this value

Respond in JSON format:
{{
    "value": "the consolidated information",
    "sources": ["url1", "url2"],
    "reasoning": "brief explanation of why this value was chosen"
}}

Consider:
- If values agree, combine sources
- If values conflict, choose the most credible/detailed one
- If values complement each other, merge them appropriately"""

        if self.config.llm_config.enable_reasoning:
            prompt += "\n\nPlease reason and think about the given context and instructions before answering the question in JSON format."

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at consolidating information from multiple sources.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": ResponseOfConsolidation.model_json_schema()},
            )

            try:
                result = ResponseOfConsolidation.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                try:
                    result = ResponseOfConsolidation.model_validate_json(
                        response.choices[0].message.reasoning_content
                    )
                except Exception as e:
                    self.logger.error(
                        "LLM_CONSOLIDATION_ERROR",
                        f"Failed to consolidate field {field_name}: {str(e)}",
                    )
                    raise ValueError("Consolidation could not be determined")

            return {
                "value": result.value,
                "sources": result.sources,
                "last_updated": datetime.now(),
            }

        except Exception as e:
            self.logger.error(
                "LLM_CONSOLIDATION_ERROR",
                f"Failed to consolidate field {field_name}: {str(e)}",
            )
            # Fallback to most recent discovery
            return {
                "value": discoveries[-1]["value"],
                "sources": [discoveries[-1]["source"]],
                "last_updated": datetime.now(),
            }

    def check_schema_completeness(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if all fields (including nested) in the data have values.
        A list field is considered complete if it's not empty AND at least one of its items is complete.
        An item is complete if all its own fields have values (and so on, recursively).
        """
        overall_missing_fields: List[str] = []

        def _recursive_check_and_collect(
            node_data: Any, current_path_prefix: str, target_missing_list: List[str]
        ):
            if isinstance(node_data, dict):
                # If the dict itself is None, its path would have been caught by the parent.
                # An empty dict as a field value means all its (potential) required fields are effectively missing
                # or it's an optional empty object. Pydantic handles required fields by them being None if not provided.
                # So, we just iterate through its keys.
                for key, value in node_data.items():
                    field_path = (
                        f"{current_path_prefix}.{key}" if current_path_prefix else key
                    )
                    _recursive_check_and_collect(value, field_path, target_missing_list)

            elif isinstance(node_data, list):
                # current_path_prefix is the path to the list field itself, e.g., "education"
                if not node_data:  # Empty list
                    target_missing_list.append(f"{current_path_prefix} (list is empty)")
                else:
                    # List is not empty. Check if at least one item is complete.
                    one_item_is_fully_complete = False
                    # This list will store missing fields from all items, used if NO item is complete.
                    all_items_missing_details_if_list_fails: List[str] = []

                    for index, item_in_list in enumerate(node_data):
                        item_path = f"{current_path_prefix}[{index}]"

                        # Check this item_in_list for its own completeness.
                        # Findings for this specific item go into a temporary list.
                        item_specific_missing_fields: List[str] = []
                        # Recursive call for the item. Its missing fields are collected in item_specific_missing_fields.
                        _recursive_check_and_collect(
                            item_in_list, item_path, item_specific_missing_fields
                        )

                        if not item_specific_missing_fields:  # This item is complete
                            one_item_is_fully_complete = True
                            all_items_missing_details_if_list_fails.clear()  # Clear details, list is complete.
                            break  # Found a complete item, so the list field (current_path_prefix) is complete.
                        else:
                            # This item is not complete. Store its missing fields in case the whole list fails.
                            all_items_missing_details_if_list_fails.extend(
                                item_specific_missing_fields
                            )

                    if not one_item_is_fully_complete:
                        # No item in the list was complete. The list field itself is incomplete.
                        # Add a general message for the list field.
                        target_missing_list.append(
                            f"{current_path_prefix} (no complete item in list)"
                        )
                        # And add all the detailed missing fields from all its items that we collected.
                        target_missing_list.extend(
                            all_items_missing_details_if_list_fails
                        )
                    # If one_item_is_fully_complete is true, we add nothing for this list field to target_missing_list.

            elif node_data is None:
                if current_path_prefix:  # Path must exist and be non-empty
                    target_missing_list.append(f"{current_path_prefix} (is None)")
            # Primitive types (str, int, float, bool, etc.) are implicitly complete if they are not None.

        _recursive_check_and_collect(
            data, "", overall_missing_fields
        )  # Root path is empty for the initial call
        return not overall_missing_fields, overall_missing_fields

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of accumulated knowledge"""
        total_fields = len(self.field_values)
        total_discoveries = sum(
            len(discoveries) for discoveries in self.field_discoveries.values()
        )

        # Get source distribution
        all_sources = set()
        for field_data in self.field_values.values():
            all_sources.update(field_data["sources"])

        summary = {
            "total_fields_filled": total_fields,
            "total_discoveries": total_discoveries,
            "unique_sources": len(all_sources),
            "fields_with_multiple_sources": sum(
                1
                for discoveries in self.field_discoveries.values()
                if len(discoveries) > 1
            ),
            "knowledge_items_count": len(self.knowledge_items),
            "latest_update": max(
                (data["last_updated"] for data in self.field_values.values()),
                default=None,
            ),
        }

        return summary

    def get_fields_needing_verification(self) -> List[str]:
        """Identify fields that might benefit from additional verification"""
        fields_to_verify = []

        for field, discoveries in self.field_discoveries.items():
            if len(discoveries) > 1:
                # Check if values differ significantly
                values = [d["value"] for d in discoveries]
                if len(set(values)) > 1:  # Multiple different values
                    fields_to_verify.append(field)

        return fields_to_verify

    def generate_knowledge_summary(
        self,
        entity_query: str = "Entity",
        entity_type: str = "Unknown",
        include_sources: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """Generate a comprehensive markdown report of accumulated knowledge"""

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Start building the markdown report
        markdown_lines = []

        # Header
        markdown_lines.extend(
            [
                f"# Knowledge Summary Report",
                f"",
                f"**Entity/Original Query:** {entity_query}",
                f"**Entity Type:** {entity_type}",
                f"**Generated At:** {timestamp}",
                f"",
                "---",
                f"",
            ]
        )

        # Overview section
        summary_stats = self.generate_summary()
        markdown_lines.extend(
            [
                "## Overview",
                f"",
                f"- **Total discoveries:** {summary_stats['total_discoveries']}",
                f"- **Unique sources:** {summary_stats['unique_sources']}",
                f"- **Knowledge items:** {summary_stats['knowledge_items_count']}",
                f"- **Fields with multiple sources:** {summary_stats['fields_with_multiple_sources']}",
                f"",
            ]
        )

        # Field-by-field breakdown
        if self.field_values:
            markdown_lines.extend(["## Field Information", f""])

            # Sort fields alphabetically for consistent output
            sorted_fields = sorted(self.field_values.keys())

            for field in sorted_fields:
                field_data = self.field_values[field]
                discoveries = self.field_discoveries.get(field, [])

                # Format field name (convert snake_case to Title Case)
                display_name = field.replace("_", " ").title()

                markdown_lines.extend(
                    [
                        f"### {display_name}",
                        f"",
                        f"**Value:** {field_data['value']}",
                        f"",
                    ]
                )

                if include_sources and field_data.get("sources"):
                    markdown_lines.extend([f"**Sources:**"])
                    for i, source in enumerate(field_data["sources"], 1):
                        markdown_lines.append(f"{i}. {source}")
                    markdown_lines.append("")

                # Show discovery details if multiple discoveries exist
                if len(discoveries) > 1:
                    markdown_lines.extend(
                        [f"**Multiple discoveries found ({len(discoveries)}):**", f""]
                    )
                    for i, discovery in enumerate(discoveries, 1):
                        markdown_lines.extend(
                            [
                                f"- **Discovery {i}:** {discovery['value']}",
                                f"  - *Source:* {discovery['source']}",
                                f"  - *Context:* {discovery['context']}",
                            ]
                        )
                    markdown_lines.append("")

                if include_metadata:
                    last_updated = field_data.get("last_updated")
                    if last_updated:
                        formatted_time = last_updated.strftime("%Y-%m-%d %H:%M:%S")
                        markdown_lines.extend(
                            [f"*Last updated: {formatted_time}*", f""]
                        )

                markdown_lines.append("---")
                markdown_lines.append("")

        # Knowledge items breakdown by type
        if self.knowledge_items:
            markdown_lines.extend(["## Knowledge Items Analysis", f""])

            # Group by knowledge type
            items_by_type = {}
            for item in self.knowledge_items:
                item_type = item.item_type.value
                if item_type not in items_by_type:
                    items_by_type[item_type] = []
                items_by_type[item_type].append(item)

            for item_type, items in items_by_type.items():
                type_display = item_type.replace("_", " ").title()
                markdown_lines.extend([f"### {type_display} ({len(items)} items)", f""])

                # Show top items
                # TODO: sort items
                sorted_items = items
                for i, item in enumerate(sorted_items[:5], 1):  # Show top 5
                    markdown_lines.extend(
                        [
                            f"{i}. **Q:** {item.question}",
                            f"   **A:** {item.answer}",
                            f"   **Fields:** {', '.join(item.schema_fields)}",
                            f"",
                        ]
                    )

                if len(items) > 5:
                    markdown_lines.extend(
                        [f"*... and {len(items) - 5} more items*", f""]
                    )

                markdown_lines.append("")

        # Source analysis
        if include_sources:
            all_sources = set()
            source_usage = {}

            for field_data in self.field_values.values():
                for source in field_data.get("sources", []):
                    all_sources.add(source)
                    source_usage[source] = source_usage.get(source, 0) + 1

            if all_sources:
                markdown_lines.extend(
                    [
                        "## Source Analysis",
                        f"",
                        f"**Total unique sources:** {len(all_sources)}",
                        f"",
                    ]
                )

                # Show most frequently used sources
                sorted_sources = sorted(
                    source_usage.items(), key=lambda x: x[1], reverse=True
                )
                markdown_lines.extend(["### Most Referenced Sources", f""])

                for source, count in sorted_sources[:10]:  # Top 10 sources
                    markdown_lines.append(
                        f"- {source} ({count} field{'s' if count > 1 else ''})"
                    )

                markdown_lines.append("")

        # Fields needing verification
        verification_fields = self.get_fields_needing_verification()
        if verification_fields:
            markdown_lines.extend(
                [
                    "## Fields Needing Verification, But can be used during Extraction",
                    f"",
                    "The following fields have information from multiple sources:",
                    f"",
                ]
            )

            for field in verification_fields:
                discoveries = self.field_discoveries[field]
                unique_values = list(set(d["value"] for d in discoveries))

                display_name = field.replace("_", " ").title()
                markdown_lines.extend([f"### {display_name}", f"**Values found:**"])

                for i, value in enumerate(unique_values, 1):
                    sources = [d["source"] for d in discoveries if d["value"] == value]
                    markdown_lines.append(
                        f'{i}. "{value}" (from {len(sources)} source{"s" if len(sources) > 1 else ""})'
                    )

                markdown_lines.append("")

        # Research recommendations
        markdown_lines.extend(["## Research Recommendations", f""])

        # Identify fields with single sources that might need more validation
        single_source_fields = [
            field
            for field, data in self.field_values.items()
            if len(data.get("sources", [])) == 1
        ]

        if single_source_fields:
            markdown_lines.extend(
                [
                    "### Additional Validation Suggested",
                    f"The following fields are based on single sources and might benefit from additional validation:",
                    f"",
                ]
            )
            for field in single_source_fields[:5]:  # Show first 5
                display_name = field.replace("_", " ").title()
                markdown_lines.append(f"- {display_name}")

            if len(single_source_fields) > 5:
                markdown_lines.append(f"- ... and {len(single_source_fields) - 5} more")

            markdown_lines.append("")

        # Footer
        markdown_lines.extend(
            [
                "---",
                f"",
                f"*Report generated by Knowledge Accumulator on {timestamp}*",
            ]
        )

        with open(
            f"knowledges/knowledge_summary_{entity_query.replace(' ', '_')}.md", "w"
        ) as f:
            f.write("\n".join(markdown_lines))

        return "\n".join(markdown_lines)
