"""
Basic Usage Examples for DeepAgent

Demonstrates simple task execution with autonomous reasoning.
"""

import sys
sys.path.insert(0, '..')

from deepagent import DeepAgent, AgentConfig


def example_1_simple_task():
    """Example 1: Simple information retrieval"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Information Retrieval")
    print("=" * 80)

    # Create agent with default config
    agent = DeepAgent()

    # Execute task
    result = agent.run(
        task="Search for information about DNA replication"
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Success: {result.success}")
    print(f"Steps taken: {result.total_steps}")
    print(f"Tools used: {', '.join(result.tools_used)}")


def example_2_bioinformatics():
    """Example 2: Bioinformatics analysis"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Bioinformatics Analysis")
    print("=" * 80)

    # Create agent with custom config
    config = AgentConfig(
        max_steps=30,
        verbose=True
    )
    agent = DeepAgent(config)

    # Complex bioinformatics task
    result = agent.run(
        task="Find the protein structure for BRCA1 and analyze its binding sites"
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Execution time: {result.execution_time:.2f}s")


def example_3_drug_discovery():
    """Example 3: Drug discovery workflow"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Drug Discovery Workflow")
    print("=" * 80)

    agent = DeepAgent()

    result = agent.run(
        task="Search DrugBank for aspirin and calculate its molecular properties"
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Tools used: {result.tools_used}")


def example_4_memory_persistence():
    """Example 4: Memory across multiple tasks"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Memory Persistence")
    print("=" * 80)

    agent = DeepAgent()

    # First task
    result1 = agent.run(task="Get the structure of protein BRCA1")

    # Second task - agent remembers previous context
    result2 = agent.run(task="Now analyze the binding sites of that protein")

    # Check memory
    print("\nMemory Summary:")
    print(agent.get_memory_summary())

    # Check tool statistics
    print("\nTool Statistics:")
    stats = agent.get_tool_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_5_custom_tools():
    """Example 5: Adding custom tools"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Tools")
    print("=" * 80)

    from deepagent import ToolDefinition

    agent = DeepAgent()

    # Define custom tool
    custom_tool = ToolDefinition(
        name="calculate_gc_content",
        description="Calculate GC content percentage in DNA sequence",
        category="bioinformatics",
        parameters=[
            {"name": "sequence", "type": "str", "description": "DNA sequence"}
        ],
        returns="float",
        examples=["calculate_gc_content('ATCGATCG')"]
    )

    # Implement custom tool
    def gc_content_impl(sequence: str) -> float:
        """Calculate GC content"""
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        return (gc_count / len(sequence)) * 100 if sequence else 0.0

    # Register with agent
    agent.add_custom_tool(custom_tool, gc_content_impl)

    # Use the custom tool
    result = agent.run(
        task="Calculate the GC content of sequence ATCGATCGGCTA"
    )

    print(f"\nAnswer: {result.answer}")


def example_6_reasoning_trace():
    """Example 6: Inspecting reasoning trace"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Reasoning Trace Inspection")
    print("=" * 80)

    config = AgentConfig(verbose=False)  # Disable verbose for cleaner trace
    agent = DeepAgent(config)

    result = agent.run(
        task="Search PubMed for CRISPR research"
    )

    # Print detailed reasoning trace
    print("\nDetailed Reasoning Trace:")
    for i, trace in enumerate(result.reasoning_trace):
        print(f"\nStep {i + 1}: {trace.step_type.value.upper()}")
        print(f"  Content: {trace.content[:100]}...")
        if trace.tool_name:
            print(f"  Tool: {trace.tool_name}")
        if trace.tool_result:
            print(f"  Result: {str(trace.tool_result)[:100]}...")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("DEEPAGENT BASIC USAGE EXAMPLES")
    print("=" * 80)

    try:
        example_1_simple_task()
        example_2_bioinformatics()
        example_3_drug_discovery()
        example_4_memory_persistence()
        example_5_custom_tools()
        example_6_reasoning_trace()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
