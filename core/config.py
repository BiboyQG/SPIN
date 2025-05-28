from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class SearchConfig:
    """Configuration for search functionality"""

    provider: str = "serpapi"  # Options: brave, duck, serpapi
    max_results_per_query: int = 10
    safe_search: bool = True
    api_key: Optional[str] = None

    def __post_init__(self):
        # Load API keys from environment if not provided
        if not self.api_key:
            if self.provider == "brave":
                self.api_key = os.getenv("BRAVE_SEARCH_API_KEY")
            elif self.provider == "serpapi":
                self.api_key = os.getenv("SERPAPI_API_KEY")


@dataclass
class LLMConfig:
    """Configuration for LLM interactions"""

    provider: str = "openai"
    model_name: str = "Qwen/Qwen3-32B-AWQ"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 32768
    enable_reasoning: bool = True

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ResearchConfig:
    """Main configuration for research operations"""

    # Research constraints
    max_steps: int = 20
    max_tokens_budget: int = 1000000
    max_urls_per_step: int = 5
    max_search_queries: int = 10
    max_reflection_depth: int = 3

    # Timeouts and delays
    request_timeout: int = 30  # seconds
    step_delay: float = 1  # seconds between steps

    # URL filtering
    blocked_domains: list = None
    allowed_domains: list = None

    # TODO: Other advanced features
    # enable_caching: bool = True
    # enable_parallel_research: bool = False
    # enable_smart_extraction: bool = True

    # Sub-configurations
    search_config: SearchConfig = None
    llm_config: LLMConfig = None

    def __post_init__(self):
        if self.blocked_domains is None:
            self.blocked_domains = []
        if self.search_config is None:
            self.search_config = SearchConfig()
        if self.llm_config is None:
            self.llm_config = LLMConfig()


class ConfigManager:
    """Manages configuration loading and validation"""

    @staticmethod
    def load_from_env() -> ResearchConfig:
        """Load configuration from environment variables"""
        config = ResearchConfig()

        # Override with environment variables if present
        if os.getenv("RESEARCH_MAX_STEPS"):
            config.max_steps = int(os.getenv("RESEARCH_MAX_STEPS"))
        if os.getenv("RESEARCH_MAX_TOKENS"):
            config.max_tokens_budget = int(os.getenv("RESEARCH_MAX_TOKENS"))

        return config

    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> ResearchConfig:
        """Load configuration from dictionary"""
        search_config = SearchConfig(**config_dict.get("search", {}))
        llm_config = LLMConfig(**config_dict.get("llm", {}))

        research_config = config_dict.get("research", {})
        research_config["search_config"] = search_config
        research_config["llm_config"] = llm_config

        return ResearchConfig(**research_config)

    @staticmethod
    def validate_config(config: ResearchConfig) -> bool:
        """Validate configuration settings"""
        if config.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if config.max_tokens_budget <= 0:
            raise ValueError("max_tokens_budget must be positive")

        # Validate sub-configs
        if config.search_config.provider not in ["brave", "serpapi"]:
            raise ValueError(
                f"Unknown search provider: {config.search_config.provider}"
            )

        return True


# Global configuration instance
_config: Optional[ResearchConfig] = None


def get_config() -> ResearchConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigManager.load_from_env()
    return _config


def set_config(config: ResearchConfig):
    """Set the global configuration instance"""
    global _config
    ConfigManager.validate_config(config)
    _config = config
