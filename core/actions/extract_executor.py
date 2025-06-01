from typing import Dict, Any, Type
from pydantic import BaseModel
from openai import OpenAI

from core.data_structures import ResearchContext, ResearchAction
from core.knowledge_accumulator import KnowledgeAccumulator
from schemas.schema_manager import schema_manager
from core.actions.base import ActionExecutor
from core.logging_config import LLMError


class ExtractExecutor(ActionExecutor):
    """Executes EXTRACT actions to structure information into schema format"""

    def __init__(self, knowledge_accumulator: KnowledgeAccumulator, llm_client: OpenAI):
        super().__init__()
        self.knowledge_accumulator = knowledge_accumulator
        self.llm_client = llm_client

    def execute(
        self,
        action: ResearchAction,
        context: ResearchContext,
        skip_step: bool = False,
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

            # Extract structured data using LLM
            extracted_data = self._extract_with_llm(
                knowledge_text,
                context.original_query,
                context.entity_type,
                schema_class,
            )

            # Update context with extraction
            context.current_extraction = extracted_data
            is_complete, empty_fields = (
                self.knowledge_accumulator.check_schema_completeness(extracted_data)
            )
            context.empty_fields = empty_fields
            if is_complete:
                context.is_complete = True
                context.completion_reason = (
                    "All schema fields have been filled with values"
                )
            else:
                percentage = len(empty_fields) / len(context.all_fields)
                if percentage <= 0.1:
                    context.is_complete = True
                    context.completion_reason = "Nearly all schema fields have been filled with values, with only a few fields (less than 10%) left to be verified"

            result = {
                "success": True,
                "extracted_data": extracted_data,
                "items_processed": 1,
            }

            self.post_execute(action, context, result, skip_step)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _prepare_knowledge_for_extraction(self, context: ResearchContext) -> str:
        return self.knowledge_accumulator.generate_knowledge_summary(
            entity_query=context.original_query,
            entity_type=context.entity_type,
            include_sources=True,
            include_metadata=True,
        )

    def _extract_with_llm(
        self,
        knowledge_text: str,
        entity_query: str,
        entity_type: str,
        schema_class: Type[BaseModel],
    ) -> Dict[str, Any]:
        """Extract structured data using LLM with guided JSON"""

        prompt = f"""You are updating structured information about a {entity_type} entity.

Entity Query: {entity_query}

Based on the following research findings, update all available information and structure it according to the provided JSON schema. The JSON schema is: {schema_class.model_json_schema()}.
If a field cannot be determined from the available information, leave it as null or empty (depending on the field type).

Research Findings (if any):
{knowledge_text}

Important guidelines:
1. Only update information that is explicitly stated or can be reasonably inferred
2. Maintain accuracy - do not make up information
3. For lists, include all relevant items found
4. For contact information, ensure proper formatting
5. Preserve URLs and email addresses exactly as found

Return the updated information as a JSON object matching the schema."""

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

            try:
                result = schema_class.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                try:
                    result = schema_class.model_validate_json(
                        response.choices[0].message.reasoning_content
                    )
                except Exception as e:
                    self.logger.error(
                        "LLM_EXTRACTION_ERROR",
                        f"Failed to extract structured data: {str(e)}",
                    )
                    raise ValueError("Extraction could not be determined")

            return result.model_dump()

        except Exception as e:
            raise LLMError(f"Failed to extract structured data: {str(e)}")
