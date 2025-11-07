#!/usr/bin/env python3
"""
DeepAgent Quick Start Script

Run this script to see DeepAgent in action immediately.
"""

from deepagent import DeepAgent, AgentConfig


def main():
    print("\n" + "=" * 80)
    print("DEEPAGENT QUICK START")
    print("=" * 80)
    print("\nWelcome to DeepAgent - an advanced agentic AI system!")
    print("This demo will show you how DeepAgent autonomously:")
    print("  1. Reasons about complex tasks")
    print("  2. Discovers and uses tools dynamically")
    print("  3. Maintains memory across reasoning steps")
    print("=" * 80 + "\n")

    # Create agent
    print("Initializing DeepAgent...")
    config = AgentConfig(
        max_steps=30,
        verbose=True
    )
    agent = DeepAgent(config)

    print(f"\nAgent initialized with {len(agent.tool_registry.get_all_tools())} available tools")
    print("\nStarting autonomous reasoning...\n")

    # Run a demo task
    result = agent.run(
        task="Find the protein structure for BRCA1 and analyze its binding sites for drug discovery"
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nFinal Answer:")
    print(f"{result.answer}")
    print(f"\nExecution Details:")
    print(f"  Success: {result.success}")
    print(f"  Total Steps: {result.total_steps}")
    print(f"  Execution Time: {result.execution_time:.2f}s")
    print(f"  Tools Used: {', '.join(result.tools_used)}")

    # Show reasoning trace summary
    print("\n" + "=" * 80)
    print("REASONING TRACE")
    print("=" * 80)
    for i, trace in enumerate(result.reasoning_trace[:5]):  # Show first 5 steps
        print(f"\nStep {i+1}: {trace.step_type.value.upper()}")
        print(f"  {trace.content[:120]}...")

    if len(result.reasoning_trace) > 5:
        print(f"\n... and {len(result.reasoning_trace) - 5} more steps")

    # Show memory state
    print("\n" + "=" * 80)
    print("AGENT MEMORY STATE")
    print("=" * 80)
    print(agent.get_memory_summary())

    # Show tool statistics
    print("\n" + "=" * 80)
    print("TOOL USAGE STATISTICS")
    print("=" * 80)
    stats = agent.get_tool_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if value < 1 else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("QUICK START COMPLETE")
    print("=" * 80)
    print("\nTo learn more, check out:")
    print("  - examples/basic_usage.py - Basic usage examples")
    print("  - examples/genesis_ai_integration.py - Synthetic biology workflows")
    print("  - README.md - Full documentation")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
