from typing import Dict, Any, Type
from datetime import datetime
import json
from openai import OpenAI
from pydantic import BaseModel

from core.actions.base import ActionExecutor
from core.data_structures import ResearchContext, ResearchAction
from core.knowledge_accumulator import KnowledgeAccumulator
from core.logging_config import LLMError
from schemas.schema_manager import schema_manager


class ExtractExecutor(ActionExecutor):
    """Executes EXTRACT actions to structure information into schema format"""

    def __init__(self, knowledge_accumulator: KnowledgeAccumulator, llm_client: OpenAI):
        super().__init__()
        self.knowledge_accumulator = knowledge_accumulator
        self.llm_client = llm_client

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute an extract action"""
        self.pre_execute(action, context)

        try:
            # Get the schema class
            schema_class = schema_manager.get_schema(context.entity_type)
            if not schema_class:
                raise ValueError(f"Unknown schema type: {context.entity_type}")

            # Prepare knowledge for extraction
            knowledge_text = self._prepare_knowledge_for_extraction(context)

            print("="*80)
            print("Knowledge text:\n\n")
            print(knowledge_text)
            print("="*80)

            # Extract structured data using LLM
            extracted_data = self._extract_with_llm(
                knowledge_text,
                context.original_query,
                context.entity_type,
                schema_class,
            )

            # Update context with extraction
            context.current_extraction = extracted_data
            context.update_field_status()

            # Calculate extraction quality
            extraction_quality = self._assess_extraction_quality(
                extracted_data, context
            )

            result = {
                "success": True,
                "extracted_data": extracted_data,
                "fields_filled": len(context.filled_fields),
                "total_fields": len(context.schema),
                "extraction_quality": extraction_quality,
                "items_processed": 1,
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _prepare_knowledge_for_extraction(self, context: ResearchContext) -> str:
        return self.knowledge_accumulator.generate_knowledge_summary()

    def _extract_with_llm(
        self,
        knowledge_text: str,
        entity_query: str,
        entity_type: str,
        schema_class: Type[BaseModel],
    ) -> Dict[str, Any]:
        """Extract structured data using LLM with guided JSON"""

        prompt = f"""You are extracting structured information about a {entity_type} entity.

Entity Query: {entity_query}

Based on the following research findings, extract all available information and structure it according to the provided JSON schema. The JSON schema is: {schema_class.model_json_schema()}.
If a field cannot be determined from the available information, leave it as null or empty (depending on the field type).

Research Findings:
{knowledge_text}

Important guidelines:
1. Only extract information that is explicitly stated or can be reasonably inferred
2. Maintain accuracy - do not make up information
3. For lists, include all relevant items found
4. For contact information, ensure proper formatting
5. Preserve URLs and email addresses exactly as found

Return the extracted information as a JSON object matching the schema."""

        if self.config.llm_config.enable_reasoning:
            prompt += "\n\nPlease reason and think about the given context and instructions before generating the JSON object."

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert at extracting {entity_type} information in JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens,
                extra_body={"guided_json": schema_class.model_json_schema()},
            )

            if self.config.llm_config.enable_reasoning:
                reasoning_content = response.choices[0].message.reasoning_content
                print("="*80)
                print("Reasoning content:\n\n")
                print(reasoning_content)
                print("="*80)

            # Parse and validate response
            extracted_json = response.choices[0].message.content
            extracted_data = json.loads(extracted_json)

            # Validate against schema
            validated_data = schema_class.model_validate(extracted_data)

            return validated_data.model_dump()

        except Exception as e:
            raise LLMError(f"Failed to extract structured data: {str(e)}")

    def _assess_extraction_quality(
        self, extracted_data: Dict[str, Any], context: ResearchContext
    ) -> Dict[str, float]:
        """Assess the quality of the extraction"""
        total_fields = len(context.schema)
        filled_fields = 0
        high_confidence_fields = 0

        for field, value in extracted_data.items():
            if value is not None and value != "" and value != []:
                filled_fields += 1

                # Check confidence for this field
                field_confidence = (
                    self.knowledge_accumulator.calculate_field_confidence(field)
                )
                if field_confidence > 0.8:
                    high_confidence_fields += 1

        completeness = filled_fields / total_fields if total_fields > 0 else 0
        confidence = high_confidence_fields / filled_fields if filled_fields > 0 else 0

        return {
            "completeness": completeness,
            "confidence": confidence,
            "filled_fields": filled_fields,
            "high_confidence_fields": high_confidence_fields,
            "total_fields": total_fields,
        }
