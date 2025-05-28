from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ActionType(Enum):
    """Types of research actions"""

    SEARCH = "search"
    VISIT = "visit"
    REFLECT = "reflect"
    EXTRACT = "extract"
    EVALUATE = "evaluate"


class KnowledgeType(Enum):
    """Types of knowledge items"""

    SEARCH_RESULT = "search_result"
    EXTRACTION = "extraction"
    INFERENCE = "inference"
    REFLECTION = "reflection"


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge gathered during research"""

    question: str
    answer: str
    source_urls: List[str]
    timestamp: datetime
    item_type: KnowledgeType
    schema_fields: List[str]  # Which schema fields this relates to
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result"""

    url: str
    title: str
    snippet: str
    relevance_score: float
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class URLInfo:
    """Information about a URL"""

    url: str
    title: str
    relevance_score: float
    schema_fields_coverage: List[str]  # Which schema fields this URL might help fill
    content: Optional[str] = None
    last_visited: Optional[datetime] = None
    visit_count: int = 0
    extraction_success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchAction:
    """Represents a research action to be taken"""

    action_type: ActionType
    reason: str
    parameters: Dict[str, Any]
    priority: float = 0.5
    estimated_cost: float = 0.0  # Token/time cost estimation


@dataclass
class ResearchContext:
    """Maintains the state of ongoing research"""

    # Query and entity information
    original_query: str
    entity_type: str
    schema: Dict[str, Any]

    # Knowledge accumulation
    knowledge_items: List[KnowledgeItem] = field(default_factory=list)

    # URL management
    discovered_urls: Dict[str, URLInfo] = field(default_factory=dict)
    visited_urls: Set[str] = field(default_factory=set)
    failed_urls: Set[str] = field(default_factory=set)

    # Current extraction state
    current_extraction: Dict[str, Any] = field(default_factory=dict)
    filled_fields: Set[str] = field(default_factory=set)
    empty_fields: Set[str] = field(default_factory=set)

    # Research progress
    actions_taken: List[ResearchAction] = field(default_factory=list)
    current_step: int = 1
    total_tokens_used: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    # Constraints
    max_steps: int = 20
    max_tokens: int = 1000000
    max_urls_per_step: int = 5

    # Research state
    is_complete: bool = False
    completion_reason: Optional[str] = None

    # Sub-questions for reflection
    open_questions: List[str] = field(default_factory=list)
    answered_questions: Set[str] = field(default_factory=set)

    # Search history
    search_queries: List[str] = field(default_factory=list)

    def update_field_status(self):
        """Update which fields are filled/empty based on current extraction"""
        self.filled_fields = {
            k
            for k, v in self.current_extraction.items()
            if v is not None and v != "" and v != []
        }
        self.empty_fields = set(self.schema.keys()) - self.filled_fields

    def add_knowledge(self, item: KnowledgeItem):
        """Add a knowledge item and update relevant fields"""
        self.knowledge_items.append(item)
        # Update filled fields if this knowledge relates to schema fields
        for field in item.schema_fields:
            if field in self.empty_fields:
                self.filled_fields.add(field)
                self.empty_fields.discard(field)

    def should_continue_research(self) -> bool:
        """Determine if research should continue"""
        if self.is_complete:
            return False
        if self.current_step >= self.max_steps:
            return False
        if self.total_tokens_used >= self.max_tokens:  # Ignore token budget for now
            return False
        return True


@dataclass
class EvaluationResult:
    """Result of evaluating the current research state"""

    accuracy: float  # 0-1 score
    consistency: float  # 0-1 score
    overall_score: float  # 0-1 score
    missing_critical_fields: List[str]
    recommendations: List[str]
    should_continue: bool
    reasoning: str
