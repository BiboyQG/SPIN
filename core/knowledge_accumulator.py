from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

from core.data_structures import KnowledgeItem, KnowledgeType
from core.config import get_config
from core.logging_config import get_logger


class KnowledgeAccumulator:
    """Manages research findings and knowledge building"""

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.knowledge_base: List[KnowledgeItem] = []
        self.field_knowledge: Dict[str, List[KnowledgeItem]] = defaultdict(list)
        self.source_credibility: Dict[str, float] = {}

    def add_knowledge(self, item: KnowledgeItem) -> None:
        """Add a knowledge item to the accumulator"""
        # Validate knowledge item
        if not self._validate_knowledge(item):
            return

        # Add to main knowledge base
        self.knowledge_base.append(item)

        # Index by schema fields DIDN"T HANDLE NESTED FIELDS
        for field in item.schema_fields:
            self.field_knowledge[field].append(item)

        # Update source credibility
        for url in item.source_urls:
            self._update_source_credibility(url, item.confidence)

        self.logger.info(
            "KNOWLEDGE_ADDED",
            f"Added knowledge item of type {item.item_type.value}",
            fields=item.schema_fields,
            confidence=item.confidence,
        )

    def _validate_knowledge(self, item: KnowledgeItem) -> bool:
        """Validate a knowledge item before adding"""
        if not item.answer or not item.answer.strip():
            self.logger.warning("KNOWLEDGE_VALIDATION", "Rejected empty knowledge item")
            return False

        if item.confidence < 0.1:  # Very low confidence threshold
            self.logger.warning(
                "KNOWLEDGE_VALIDATION",
                "Rejected knowledge item with very low confidence",
                confidence=item.confidence,
            )
            return False

        return True

    def _update_source_credibility(self, url: str, confidence: float) -> None:
        """Update credibility score for a source"""
        if url not in self.source_credibility:
            self.source_credibility[url] = confidence
        else:
            # Running average
            self.source_credibility[url] = (
                self.source_credibility[url] * 0.7 + confidence * 0.3
            )

    def get_knowledge_for_field(self, field_name: str) -> List[KnowledgeItem]:
        """Get all knowledge items related to a specific field"""
        return self.field_knowledge.get(field_name, [])

    def get_best_knowledge_for_field(self, field_name: str) -> Optional[KnowledgeItem]:
        """Get the most confident knowledge item for a field"""
        field_items = self.get_knowledge_for_field(field_name)
        if not field_items:
            return None

        # Sort by confidence and recency
        def score_item(item: KnowledgeItem) -> float:
            recency_factor = 1.0  # Could decay based on age
            source_factor = (
                max(self.source_credibility.get(url, 0.5) for url in item.source_urls)
                if item.source_urls
                else 0.5
            )

            return item.confidence * 0.6 + source_factor * 0.3 + recency_factor * 0.1

        return max(field_items, key=score_item)

    def consolidate_field_knowledge(self, field_name: str) -> Optional[str]:
        """Consolidate multiple knowledge items for a field into a single answer"""
        field_items = self.get_knowledge_for_field(field_name)
        if not field_items:
            return None

        # If only one item, return it
        if len(field_items) == 1:
            return field_items[0].answer

        # Group by answer similarity
        answer_groups = self._group_similar_answers(field_items)

        # Select the best group (most items with high confidence)
        best_group = max(
            answer_groups, key=lambda g: sum(item.confidence for item in g) * len(g)
        )

        # Return the answer from the most confident item in the best group
        best_item = max(best_group, key=lambda item: item.confidence)

        self.logger.debug(
            "KNOWLEDGE_CONSOLIDATION",
            f"Consolidated {len(field_items)} items for field {field_name}",
            selected_confidence=best_item.confidence,
        )

        return best_item.answer

    def _group_similar_answers(
        self, items: List[KnowledgeItem]
    ) -> List[List[KnowledgeItem]]:
        """Group knowledge items with similar answers"""
        groups = []

        for item in items:
            # Try to find a matching group
            matched = False
            for group in groups:
                if self._are_answers_similar(item.answer, group[0].answer):
                    group.append(item)
                    matched = True
                    break

            # Create new group if no match
            if not matched:
                groups.append([item])

        return groups

    def _are_answers_similar(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are similar enough to be grouped"""
        # Normalize answers
        norm1 = self._normalize_answer(answer1)
        norm2 = self._normalize_answer(answer2)

        # Exact match after normalization
        if norm1 == norm2:
            return True

        # Check token overlap (simple similarity)
        tokens1 = set(norm1.split())
        tokens2 = set(norm2.split())

        if not tokens1 or not tokens2:
            return False

        overlap = len(tokens1 & tokens2)
        total = len(tokens1 | tokens2)

        similarity = overlap / total if total > 0 else 0
        return similarity > 0.7

    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison"""
        # Convert to lowercase
        answer = answer.lower()
        # Remove extra whitespace
        answer = " ".join(answer.split())
        # Remove punctuation from edges
        answer = answer.strip(".,!?;:")
        return answer

    def calculate_field_confidence(self, field_name: str) -> float:
        """Calculate overall confidence for a schema field"""
        field_items = self.get_knowledge_for_field(field_name)
        if not field_items:
            return 0.0

        # Consider multiple factors
        max_confidence = max(item.confidence for item in field_items)
        avg_confidence = sum(item.confidence for item in field_items) / len(field_items)

        # Agreement bonus - higher confidence if multiple items agree
        answer_groups = self._group_similar_answers(field_items)
        largest_group_ratio = max(len(g) for g in answer_groups) / len(field_items)

        # Weighted combination
        confidence = (
            max_confidence * 0.4 + avg_confidence * 0.3 + largest_group_ratio * 0.3
        )

        return min(confidence, 1.0)

    def get_schema_completion_map(self, schema_fields: List[str]) -> Dict[str, float]:
        """Get completion status for all schema fields"""
        completion_map = {}

        for field in schema_fields:
            confidence = self.calculate_field_confidence(field)
            completion_map[field] = confidence

        return completion_map

    def identify_knowledge_gaps(
        self, schema_fields: List[str], threshold: float = 0.7
    ) -> List[str]:
        """Identify schema fields that need more research"""
        gaps = []
        completion_map = self.get_schema_completion_map(schema_fields)

        for field, confidence in completion_map.items():
            if confidence < threshold:
                gaps.append(field)

        # Sort by lowest confidence first
        gaps.sort(key=lambda f: completion_map[f])

        return gaps

    def generate_summary(self) -> Dict[str, any]:
        """Generate a summary of accumulated knowledge"""
        summary = {
            "total_items": len(self.knowledge_base),
            "by_type": defaultdict(int),
            "by_field": {},
            "source_count": len(
                set(url for item in self.knowledge_base for url in item.source_urls)
            ),
            "avg_confidence": sum(item.confidence for item in self.knowledge_base)
            / len(self.knowledge_base)
            if self.knowledge_base
            else 0,
        }

        # Count by type
        for item in self.knowledge_base:
            summary["by_type"][item.item_type.value] += 1

        # Count and confidence by field
        for field, items in self.field_knowledge.items():
            summary["by_field"][field] = {
                "count": len(items),
                "confidence": self.calculate_field_confidence(field),
            }

        return dict(summary)

    def merge_knowledge_bases(self, other: "KnowledgeAccumulator") -> None:
        """Merge another knowledge accumulator into this one"""
        for item in other.knowledge_base:
            self.add_knowledge(item)

    def prune_low_quality_knowledge(self, min_confidence: float = 0.3) -> int:
        """Remove low-quality knowledge items"""
        original_count = len(self.knowledge_base)

        # Filter knowledge base
        self.knowledge_base = [
            item for item in self.knowledge_base if item.confidence >= min_confidence
        ]

        # Rebuild field index
        self.field_knowledge.clear()
        for item in self.knowledge_base:
            for field in item.schema_fields:
                self.field_knowledge[field].append(item)

        pruned_count = original_count - len(self.knowledge_base)

        if pruned_count > 0:
            self.logger.info(
                "KNOWLEDGE_PRUNING",
                f"Pruned {pruned_count} low-quality knowledge items",
                min_confidence=min_confidence,
            )

        return pruned_count
