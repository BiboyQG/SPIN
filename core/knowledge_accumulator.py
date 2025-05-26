from typing import List, Dict, Set, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime
from openai import OpenAI
from collections import defaultdict

from core.data_structures import KnowledgeItem, KnowledgeType
from core.config import get_config
from core.logging_config import get_logger
from core.response_model import ResponseOfConsolidation

if TYPE_CHECKING:
    from core.data_structures import ResearchContext


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
        self, item: KnowledgeItem, context: Optional["ResearchContext"] = None
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
        self, item: KnowledgeItem, context: Optional["ResearchContext"] = None
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
        self, field_name: str, context: Optional["ResearchContext"] = None
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

        # When there are multiple discoveries, add field to context.filled_fields if context is provided
        if context is not None and len(discoveries) > 1:
            context.filled_fields.add(field_name)

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
                print("=" * 80)
                print("Error:\n\n")
                print(e)
                print("=" * 80)
                result = ResponseOfConsolidation.model_validate_json(
                    response.choices[0].message.reasoning_content
                )

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

    def get_field_value(self, field_name: str) -> Optional[str]:
        """Get the consolidated value for a field"""
        field_data = self.field_values.get(field_name)
        return field_data["value"] if field_data else None

    def get_field_sources(self, field_name: str) -> List[str]:
        """Get all sources for a field value"""
        field_data = self.field_values.get(field_name)
        return field_data["sources"] if field_data else []

    def get_all_field_values(self) -> Dict[str, Any]:
        """Get all field values with their sources"""
        return {
            field: {"value": data["value"], "sources": data["sources"]}
            for field, data in self.field_values.items()
        }

    def check_schema_completeness(
        self, schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Check if all fields (including nested) in the schema have values"""
        missing_fields = []

        def check_fields(obj: Dict[str, Any], prefix: str = ""):
            for key, value in obj.items():
                field_path = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict) and not value.get("type"):
                    # Nested object
                    check_fields(value, field_path)
                else:
                    # Regular field
                    if field_path not in self.field_values:
                        missing_fields.append(field_path)

        check_fields(schema)

        is_complete = len(missing_fields) == 0
        return is_complete, missing_fields

    def get_knowledge_for_field(self, field_name: str) -> List[Dict[str, Any]]:
        """Get all discoveries for a specific field"""
        return self.field_discoveries.get(field_name, [])

    def identify_knowledge_gaps(self, schema_fields: List[str]) -> List[str]:
        """Identify schema fields that don't have values yet"""
        gaps = []
        for field in schema_fields:
            if field not in self.field_values:
                gaps.append(field)
        return gaps

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

    def merge_knowledge_bases(self, other: "KnowledgeAccumulator") -> None:
        """Merge another knowledge accumulator into this one"""
        # Merge knowledge items
        for item in other.knowledge_items:
            self.add_knowledge(item, None)

        # Merge discoveries (will trigger re-consolidation)
        for field, discoveries in other.field_discoveries.items():
            self.field_discoveries[field].extend(discoveries)
            self._update_field_value(field, None)

    def export_extraction(self) -> Dict[str, Any]:
        """Export the current extraction with sources"""
        extraction = {}
        for field, data in self.field_values.items():
            # Handle nested fields
            parts = field.split(".")
            current = extraction
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value with metadata
            current[parts[-1]] = {"value": data["value"], "sources": data["sources"]}

        return extraction

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
                f"**Entity:** {entity_query}",
                f"**Type:** {entity_type}",
                f"**Generated:** {timestamp}",
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
                f"- **Fields with data:** {summary_stats['total_fields_filled']}",
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

                # Show top items by confidence
                sorted_items = sorted(items, key=lambda x: x.confidence, reverse=True)
                for i, item in enumerate(sorted_items[:5], 1):  # Show top 5
                    confidence_pct = int(item.confidence * 100)
                    markdown_lines.extend(
                        [
                            f"{i}. **Q:** {item.question}",
                            f"   **A:** {item.answer[:2000]}{'...' if len(item.answer) > 2000 else ''}",
                            f"   **Confidence:** {confidence_pct}% | **Fields:** {', '.join(item.schema_fields)}",
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
                    "## Fields Needing Verification",
                    f"",
                    "The following fields have conflicting information from multiple sources:",
                    f"",
                ]
            )

            for field in verification_fields:
                discoveries = self.field_discoveries[field]
                unique_values = list(set(d["value"] for d in discoveries))

                display_name = field.replace("_", " ").title()
                markdown_lines.extend(
                    [f"### {display_name}", f"**Conflicting values found:**"]
                )

                for i, value in enumerate(unique_values, 1):
                    sources = [d["source"] for d in discoveries if d["value"] == value]
                    markdown_lines.append(
                        f'{i}. "{value}" (from {len(sources)} source{"s" if len(sources) > 1 else ""})'
                    )

                markdown_lines.append("")

        # Research recommendations
        markdown_lines.extend(["## Research Recommendations", f""])

        if verification_fields:
            markdown_lines.extend(
                [
                    "### Priority Actions",
                    f"1. **Verify conflicting information** for {len(verification_fields)} field{'s' if len(verification_fields) > 1 else ''}: {', '.join(verification_fields)}",
                    f"2. **Cross-reference sources** to determine most reliable information",
                    f"",
                ]
            )

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
                f"*Report generated by SPIN Knowledge Accumulator on {timestamp}*",
            ]
        )

        return "\n".join(markdown_lines)
