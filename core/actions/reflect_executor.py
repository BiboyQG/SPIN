from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI

from core.actions.base import ActionExecutor
from core.data_structures import (
    ResearchContext,
    ResearchAction,
    KnowledgeItem,
    KnowledgeType,
)
from core.knowledge_accumulator import KnowledgeAccumulator
from core.logging_config import LLMError


class ReflectExecutor(ActionExecutor):
    """Executes REFLECT actions to analyze knowledge gaps"""

    def __init__(self, knowledge_accumulator: KnowledgeAccumulator, llm_client: OpenAI):
        super().__init__()
        self.knowledge_accumulator = knowledge_accumulator
        self.llm_client = llm_client

    def execute(
        self, action: ResearchAction, context: ResearchContext
    ) -> Dict[str, Any]:
        """Execute a reflect action"""
        self.pre_execute(action, context)

        try:
            # Analyze current state
            analysis = self._analyze_research_state(context)

            # Generate sub-questions for gaps
            sub_questions = self._generate_sub_questions(context, analysis)

            # Add new questions to context
            new_questions = []
            for question in sub_questions:
                if question not in context.answered_questions:
                    context.open_questions.append(question)
                    new_questions.append(question)

            # Create reflection knowledge item
            reflection_item = KnowledgeItem(
                question="What are the knowledge gaps in our research?",
                answer=analysis["gap_summary"],
                source_urls=[],
                confidence=0.8,
                timestamp=datetime.now(),
                item_type=KnowledgeType.REFLECTION,
                schema_fields=list(context.empty_fields),
                metadata={
                    "sub_questions": sub_questions,
                    "progress": context.get_progress_percentage(),
                },
            )
            self.knowledge_accumulator.add_knowledge(reflection_item)

            result = {
                "success": True,
                "analysis": analysis,
                "new_questions": new_questions,
                "total_questions": len(sub_questions),
                "items_processed": 1,
            }

            self.post_execute(action, context, result)
            return result

        except Exception as e:
            self.handle_error(action, context, e)
            return {"success": False, "error": str(e), "items_processed": 0}

    def _analyze_research_state(self, context: ResearchContext) -> Dict[str, Any]:
        """Analyze current research state using LLM"""
        # Prepare context summary
        knowledge_summary = self._summarize_knowledge(context)

        prompt = f"""Analyze the current state of research for the entity: {context.original_query}

Entity Type: {context.entity_type}

Current Progress:
- Fields filled: {len(context.filled_fields)}/{len(context.schema)}
- Knowledge items collected: {len(context.knowledge_items)}
- URLs discovered: {len(context.discovered_urls)}
- URLs visited: {len(context.visited_urls)}

Empty Fields: {", ".join(list(context.empty_fields)[:10])}

Knowledge Summary:
{knowledge_summary}

Provide a brief analysis of:
1. What information we have found so far
2. What critical information is still missing
3. Why certain fields might be difficult to fill
4. Suggested strategies for finding missing information

Format your response as a JSON object with keys: 'found_info', 'missing_info', 'difficulties', 'strategies', 'gap_summary'"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research analyst helping to identify knowledge gaps.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            import json

            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            raise LLMError(f"Failed to analyze research state: {str(e)}")

    def _generate_sub_questions(
        self, context: ResearchContext, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate sub-questions to address knowledge gaps"""
        focus_fields = list(context.empty_fields)[:5]  # Focus on top 5 empty fields

        prompt = f"""Based on the research for "{context.original_query}", generate specific sub-questions that would help find information for these missing fields:

Missing Fields: {", ".join(focus_fields)}

Analysis of gaps: {analysis.get("missing_info", "Various information is missing")}

Previously asked questions: {", ".join(context.search_queries[-5:])}

Generate 3-5 specific, searchable questions that would help find the missing information.
These should be different from previously asked questions and target specific aspects of the missing fields.

Return as a JSON object with key 'questions' containing a list of questions."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_config.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research assistant generating targeted questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
            )

            import json

            result = json.loads(response.choices[0].message.content)
            return result.get("questions", [])

        except Exception as e:
            self.logger.error(
                "SUB_QUESTION_GENERATION_FAILED",
                "Failed to generate sub-questions",
                error=str(e),
            )
            # Fallback to simple questions
            return [f"{context.original_query} {field}" for field in focus_fields[:3]]

    def _summarize_knowledge(self, context: ResearchContext) -> str:
        """Summarize current knowledge"""
        summary_parts = []

        # Group knowledge by field
        for field in list(context.filled_fields)[:5]:  # Top 5 filled fields
            knowledge = self.knowledge_accumulator.get_best_knowledge_for_field(field)
            if knowledge:
                summary_parts.append(f"- {field}: {knowledge.answer[:100]}...")

        if not summary_parts:
            return "No significant information found yet."

        return "\n".join(summary_parts)
