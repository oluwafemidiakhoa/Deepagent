"""
Production LLM Example

Demonstrates DeepAgent with real LLM providers and all production features:
- OpenAI GPT-4 or Anthropic Claude integration
- Sentence-transformers embeddings with FAISS indexing
- Automatic retry with circuit breakers
- Observability (logging, metrics, tracing)
- Persistent memory (vector stores, databases)

Author: Oluwafemi Idiakhoa
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepagent.core.agent import DeepAgent, AgentConfig
from deepagent.integrations.observability import create_observability


def example_1_basic_llm_usage():
    """
    Example 1: Basic usage with real LLM

    Prerequisites:
    - Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    - Install: pip install openai anthropic sentence-transformers faiss-cpu
    """
    print("\n" + "="*80)
    print("Example 1: Basic LLM Usage with Production Features")
    print("="*80 + "\n")

    # Create observability manager
    obs = create_observability(service_name="deepagent-example", log_level="INFO")
    obs.log("info", "example_started", example="basic_llm_usage")

    # Configure agent with real LLM
    config = AgentConfig(
        llm_provider="openai",  # or "anthropic" or "ollama"
        llm_model="gpt-4",      # or "claude-3-5-sonnet-20241022" for Anthropic
        llm_temperature=0.7,
        use_mock_llm=False,     # Use real LLM
        max_reasoning_steps=5
    )

    # Create agent
    with obs.start_span("agent_creation"):
        agent = DeepAgent(config=config)
        obs.log("info", "agent_created", provider=config.llm_provider, model=config.llm_model)

    # Example task: Research CRISPR gene editing
    task = "Find recent research on CRISPR gene editing and summarize the top 3 findings"

    print(f"Task: {task}\n")
    obs.log("info", "task_started", task=task)

    # Execute task with tracing
    with obs.start_span("task_execution", attributes={"task": task}):
        result = agent.execute_task(task)

    # Log results
    obs.log("info", "task_completed",
            success=result.success,
            steps=len(result.history),
            execution_time=result.execution_time)

    # Record metrics
    obs.record_metric("counter", "tasks_completed", 1, {"status": "success" if result.success else "failure"})
    obs.record_metric("histogram", "task_execution_time", result.execution_time)
    obs.record_metric("gauge", "reasoning_steps", len(result.history))

    # Print results
    print(f"\nSuccess: {result.success}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Reasoning Steps: {len(result.history)}")
    print(f"Execution Time: {result.execution_time:.2f}s")

    if result.history:
        print("\nReasoning Trace:")
        for i, step in enumerate(result.history[-3:], 1):  # Last 3 steps
            print(f"  Step {i}: {step[:100]}...")

    # Export metrics
    obs.export_metrics("metrics_example1.json")
    obs.log("info", "example_completed")


def example_2_with_persistent_memory():
    """
    Example 2: Using persistent memory with vector stores

    Prerequisites:
    - Install: pip install chromadb sentence-transformers
    """
    print("\n" + "="*80)
    print("Example 2: Persistent Memory with Vector Stores")
    print("="*80 + "\n")

    obs = create_observability(service_name="deepagent-persistent", log_level="INFO")

    try:
        from deepagent.integrations.vector_stores import ChromaVectorStore
        from deepagent.core.memory import ThreeLayerMemory

        # Create memory with vector store
        vector_store = ChromaVectorStore(
            collection_name="deepagent_memory",
            persist_directory="./memory_db"
        )

        # Simple embedding function (in production, use sentence-transformers)
        def simple_embedding(text: str):
            import numpy as np
            # Hash-based embedding for demo
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(384)

        memory = ThreeLayerMemory()
        memory.episodic.vector_store = vector_store
        memory.episodic.embedding_function = simple_embedding

        obs.log("info", "vector_store_initialized", collection="deepagent_memory")

        # Configure agent
        config = AgentConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            use_mock_llm=False
        )

        agent = DeepAgent(config=config)
        agent.memory = memory  # Use custom memory

        # Execute task
        task = "Analyze the binding affinity of aspirin to COX-2 enzyme"
        obs.log("info", "task_started", task=task)

        result = agent.execute_task(task)

        # Persist episodic memories to vector store
        if memory.episodic.persist_to_vector_store():
            obs.log("info", "memories_persisted", count=len(memory.episodic.memories))
            print(f"\n✓ Persisted {len(memory.episodic.memories)} memories to vector store")
        else:
            obs.log("warning", "memory_persistence_failed")

        print(f"\nTask completed: {result.success}")
        print(f"Memories stored: {len(memory.episodic.memories)}")

    except ImportError as e:
        print(f"\nError: {e}")
        print("Install required packages: pip install chromadb sentence-transformers")
        obs.log("error", "import_error", error=str(e))


def example_3_with_retry_and_circuit_breaker():
    """
    Example 3: Demonstrating retry logic and circuit breakers

    Prerequisites:
    - Install: pip install tenacity
    """
    print("\n" + "="*80)
    print("Example 3: Retry Logic and Circuit Breakers")
    print("="*80 + "\n")

    obs = create_observability(service_name="deepagent-resilience", log_level="INFO")

    from deepagent.tools.executor import ToolExecutor, ExecutionStatus

    # Create executor with retry enabled
    executor = ToolExecutor(
        timeout=30.0,
        max_retries=3,
        retry_on_errors=True,
        use_circuit_breaker=True,
        circuit_failure_threshold=5
    )

    obs.log("info", "executor_created",
            max_retries=3,
            circuit_breaker=True,
            tenacity_available=executor.tenacity_available)

    # Test successful execution
    print("Test 1: Successful tool execution")
    result = executor.execute("search_pubmed", {"query": "CRISPR", "max_results": 5})
    print(f"  Status: {result.status.value}")
    print(f"  Retry count: {result.retry_count}")
    print(f"  Execution time: {result.execution_time:.3f}s")

    obs.record_metric("counter", "tool_executions", 1,
                     {"tool": "search_pubmed", "status": result.status.value})

    # Test with non-existent tool (will fail immediately)
    print("\nTest 2: Non-existent tool (fails without retry)")
    result = executor.execute("non_existent_tool", {})
    print(f"  Status: {result.status.value}")
    print(f"  Error: {result.error[:80]}...")

    # Simulate multiple failures to trigger circuit breaker
    print("\nTest 3: Circuit breaker demonstration")
    print("  Attempting to call non-existent tool multiple times...")

    for i in range(7):
        result = executor.execute("failing_tool_" + str(i), {})
        if "Circuit breaker open" in (result.error or ""):
            print(f"  ✓ Circuit breaker opened after {i} failures")
            obs.log("info", "circuit_breaker_triggered", failures=i)
            break
        obs.record_metric("counter", "tool_failures", 1)

    obs.export_metrics("metrics_example3.json")


def example_4_full_production_stack():
    """
    Example 4: Full production stack with all features

    Prerequisites:
    - All production dependencies installed
    - Environment variables set (OPENAI_API_KEY or ANTHROPIC_API_KEY)
    """
    print("\n" + "="*80)
    print("Example 4: Full Production Stack")
    print("="*80 + "\n")

    # Check environment
    has_openai = os.getenv("OPENAI_API_KEY") is not None
    has_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None

    if not (has_openai or has_anthropic):
        print("⚠️  Warning: No API keys found in environment")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use real LLMs")
        print("   Falling back to mock LLM for demonstration\n")
        use_mock = True
        provider = "openai"
    else:
        use_mock = False
        provider = "openai" if has_openai else "anthropic"

    # Initialize observability
    obs = create_observability(
        service_name="deepagent-production",
        log_level="INFO"
    )

    obs.log("info", "production_example_started",
            provider=provider,
            use_mock=use_mock)

    # Configure agent with all production features
    config = AgentConfig(
        llm_provider=provider,
        llm_model="gpt-4" if provider == "openai" else "claude-3-5-sonnet-20241022",
        llm_temperature=0.7,
        use_mock_llm=use_mock,
        max_reasoning_steps=10
    )

    # Create agent
    with obs.start_span("agent_initialization"):
        agent = DeepAgent(config=config)

    # Display configuration
    print("Configuration:")
    print(f"  LLM Provider: {config.llm_provider}")
    print(f"  Model: {config.llm_model}")
    print(f"  Mock LLM: {config.use_mock_llm}")
    print(f"  Max Steps: {config.max_reasoning_steps}")

    # Get observability stats
    obs_stats = obs.get_stats()
    print(f"\nObservability:")
    print(f"  Structured Logging: {'✓' if obs_stats['logger_available'] else '✗'}")
    print(f"  Metrics Collection: {'✓' if obs_stats['metrics_available'] else '✗'}")
    print(f"  Distributed Tracing: {'✓' if obs_stats['tracing_available'] else '✗'}")

    # Get tool retrieval stats
    retrieval_stats = agent.tool_registry.retriever.get_stats()
    print(f"\nTool Retrieval:")
    print(f"  Total Tools: {retrieval_stats['total_tools']}")
    print(f"  Sentence Transformers: {'✓' if retrieval_stats['using_sentence_transformers'] else '✗'}")
    print(f"  FAISS Indexing: {'✓' if retrieval_stats['using_faiss'] else '✗'}")
    print(f"  Embedding Dimension: {retrieval_stats['embedding_dim']}")

    # Execute complex task
    task = """
    Design an experiment to test CRISPR gene editing on cancer cells:
    1. Find relevant research on CRISPR and cancer
    2. Identify suitable target genes
    3. Plan experimental protocol
    """

    print(f"\nTask: {task}\n")
    obs.log("info", "complex_task_started", task=task[:100])

    # Execute with full tracing
    with obs.start_span("complex_task_execution", attributes={"task_type": "experiment_design"}):
        result = agent.execute_task(task)

    # Log comprehensive results
    obs.log("info", "task_completed",
            success=result.success,
            steps=len(result.history),
            time=result.execution_time,
            answer_length=len(result.final_answer))

    # Record detailed metrics
    obs.record_metric("counter", "tasks_executed", 1,
                     {"complexity": "high", "success": str(result.success)})
    obs.record_metric("histogram", "execution_time_seconds", result.execution_time)
    obs.record_metric("gauge", "reasoning_steps_used", len(result.history))

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Success: {'✓' if result.success else '✗'}")
    print(f"Reasoning Steps: {len(result.history)}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"\nFinal Answer:\n{result.final_answer}\n")

    if result.history:
        print("Reasoning Trace (last 3 steps):")
        for i, step in enumerate(result.history[-3:], 1):
            print(f"  {i}. {step[:150]}...")

    # Export metrics
    metrics_file = "metrics_production.json"
    obs.export_metrics(metrics_file)
    print(f"\n✓ Metrics exported to {metrics_file}")

    # Display memory summary
    memory_context = agent.memory.get_full_context()
    print(f"\nMemory Summary:")
    for line in memory_context.split('\n')[:10]:
        print(f"  {line}")

    obs.log("info", "production_example_completed")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEEPAGENT PRODUCTION LLM EXAMPLES")
    print("="*80)

    # Run examples
    try:
        example_1_basic_llm_usage()
    except Exception as e:
        print(f"\nExample 1 error: {e}")
        print("This is expected if API keys are not configured\n")

    try:
        example_2_with_persistent_memory()
    except Exception as e:
        print(f"\nExample 2 error: {e}\n")

    try:
        example_3_with_retry_and_circuit_breaker()
    except Exception as e:
        print(f"\nExample 3 error: {e}\n")

    try:
        example_4_full_production_stack()
    except Exception as e:
        print(f"\nExample 4 error: {e}\n")

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80 + "\n")
