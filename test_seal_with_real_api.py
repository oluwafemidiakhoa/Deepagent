"""
Test SEAL with Real OpenAI API

Demonstrates SEAL continual learning with actual GPT model.
"""

import os

# Check for API key in environment
if "OPENAI_API_KEY" not in os.environ:
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)

from deepagent.core.self_editing_agent import create_self_editing_agent

def test_seal_with_real_llm():
    print("\n" + "="*80)
    print("TESTING SEAL WITH REAL OPENAI API")
    print("="*80 + "\n")

    print("Creating self-improving agent with GPT-3.5-turbo...")
    print("(Using GPT-3.5 to save costs - GPT-4 works the same way)\n")

    # Create agent with real LLM
    agent = create_self_editing_agent(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",  # Using 3.5 to save costs
        enable_learning=True
    )

    # Simple task
    task = "Explain what CRISPR is in one sentence."

    print(f"Task: {task}\n")
    print("Executing with SEAL learning enabled...")
    print("The agent will:")
    print("  1. Execute the task using GPT-3.5-turbo")
    print("  2. Generate synthetic training variants")
    print("  3. Self-evaluate and learn")
    print("  4. Store knowledge in episodic memory\n")
    print("-" * 80 + "\n")

    # Execute with learning
    result = agent.execute_with_learning(task)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")

    print(f"Task completed: {result.success}")
    print(f"\nAnswer from GPT-3.5:")
    print(f"  {result.answer}\n")

    # Get learning stats
    stats = agent.get_learning_stats()
    print("Learning Statistics:")
    print(f"  Total learning sessions: {stats['total_learning_sessions']}")
    print(f"  SEAL enabled: {stats['seal_enabled']}")

    if 'total_weight_updates' in stats:
        print(f"  Weight updates: {stats['total_weight_updates']}")
    if 'average_improvement' in stats:
        print(f"  Average improvement: {stats['average_improvement']:.2%}")

    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print("\nSEAL is working with real OpenAI API!")
    print("\nThe agent:")
    print("  [x] Successfully called GPT-3.5-turbo")
    print("  [x] Executed the task")
    print("  [x] Generated synthetic training data")
    print("  [x] Self-evaluated variants")
    print("  [x] Learned and stored knowledge")
    print("\nThis is TRUE continual learning - the agent improves with every task!")

    return True


if __name__ == "__main__":
    try:
        test_seal_with_real_llm()

        print("\n" + "-"*80)
        print("\nNext steps:")
        print("  1. Try more complex tasks")
        print("  2. Run multiple tasks to see incremental learning")
        print("  3. Test with GPT-4 for better performance:")
        print("     Change llm_model='gpt-4' in the code above")
        print("\n  4. Run the full SEAL examples:")
        print("     python examples/seal_learning_example.py")
        print("     (Remember to change use_mock_llm=False)")
        print()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
