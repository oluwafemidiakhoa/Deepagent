"""
DeepAgent Integrations Module

Provides integrations with external services:
- LLM providers (OpenAI, Anthropic, Ollama)
- Vector stores (Chroma, Qdrant)
- Databases (PostgreSQL, Redis)
- Observability (Logging, Metrics, Tracing)
"""

from .llm_providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    get_llm_provider
)

from .vector_stores import (
    VectorStore,
    ChromaVectorStore,
    QdrantVectorStore
)

from .databases import (
    DatabaseConnection,
    PostgreSQLConnection,
    RedisConnection
)

from .observability import (
    setup_logging,
    track_metric,
    trace_execution
)

__all__ = [
    # LLM Providers
    "LLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "get_llm_provider",

    # Vector Stores
    "VectorStore",
    "ChromaVectorStore",
    "QdrantVectorStore",

    # Databases
    "DatabaseConnection",
    "PostgreSQLConnection",
    "RedisConnection",

    # Observability
    "setup_logging",
    "track_metric",
    "trace_execution",
]
