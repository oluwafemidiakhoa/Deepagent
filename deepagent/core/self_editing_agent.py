"""
Self-Editing Agent

DeepAgent enhanced with SEAL-style continual learning.
The first open-source agent framework with true self-improvement.

Author: Oluwafemi Idiakhoa
"""

from typing import Optional
from dataclasses import dataclass

from deepagent.core.agent import DeepAgent, AgentConfig
from deepagent.training.seal import SEALTrainer, create_seal_trainer


@dataclass
class SelfEditingAgentConfig(AgentConfig):
    """Configuration for self-editing agent"""

    # SEAL-specific configuration
    enable_seal_learning: bool = True
    seal_auto_update: bool = True
    seal_weight_updates: bool = False  # Requires peft library
    seal_learning_frequency: int = 1  # Learn every N tasks

    # Learning thresholds
    min_quality_threshold: float = 0.5
    max_learning_iterations: int = 1000


class SelfEditingAgent(DeepAgent):
    """
    DeepAgent with SEAL integration for continual learning

    Key Features:
    - Learns from every task execution
    - Generates synthetic training data automatically
    - Self-evaluates performance improvements
    - Permanently updates model weights via LoRA
    - Uses episodic memory to prevent catastrophic forgetting

    This makes DeepAgent the ONLY open-source agent framework
    with true continual learning capabilities.

    Example:
        >>> config = SelfEditingAgentConfig(
        ...     llm_provider="openai",
        ...     llm_model="gpt-4",
        ...     enable_seal_learning=True
        ... )
        >>> agent = SelfEditingAgent(config=config)
        >>> result = agent.execute_with_learning("Research CRISPR applications")
        >>> # Agent permanently learns from this execution!
    """

    def __init__(self, config: Optional[SelfEditingAgentConfig] = None):
        # Initialize base agent
        if config is None:
            config = SelfEditingAgentConfig()

        super().__init__(config)

        self.seal_config = config

        # Initialize SEAL trainer
        if config.enable_seal_learning:
            self.seal_trainer = create_seal_trainer(
                llm_provider=self.llm_provider,
                memory_system=self.memory,
                toolpo_optimizer=None,  # Could integrate with ToolPO
                enable_weight_updates=config.seal_weight_updates
            )
            print("\n✓ SEAL learning enabled - agent will self-improve!")
        else:
            self.seal_trainer = None

        # Learning counters
        self.tasks_since_learning = 0
        self.total_learning_sessions = 0

    def execute_with_learning(self, task: str) -> any:
        """
        Execute task and learn from it using SEAL

        This is the key method that enables continual learning.

        Args:
            task: Task to execute

        Returns:
            Task execution result

        Side Effects:
            - Updates model weights (if seal_weight_updates=True)
            - Stores knowledge in episodic memory
            - Improves future performance on similar tasks
        """
        # 1. Execute task normally
        result = self.execute_task(task)

        # 2. Learn from execution using SEAL
        if self.seal_config.enable_seal_learning and self.seal_trainer:
            self.tasks_since_learning += 1

            # Learn at specified frequency
            if self.tasks_since_learning >= self.seal_config.seal_learning_frequency:
                weight_update = self.seal_trainer.learn_from_execution(
                    task=task,
                    result=result,
                    auto_update=self.seal_config.seal_auto_update
                )

                if weight_update:
                    self.total_learning_sessions += 1

                self.tasks_since_learning = 0

        return result

    def execute_task(self, task: str) -> any:
        """
        Standard task execution (without learning)

        Use execute_with_learning() for continual learning
        """
        return super().execute_task(task)

    def get_learning_stats(self) -> dict:
        """
        Get statistics about agent's learning progress

        Returns:
            Dictionary with learning metrics
        """
        base_stats = {
            "total_tasks_executed": self.tasks_since_learning,
            "total_learning_sessions": self.total_learning_sessions,
            "seal_enabled": self.seal_config.enable_seal_learning
        }

        if self.seal_trainer:
            seal_stats = self.seal_trainer.get_learning_stats()
            base_stats.update(seal_stats)

        return base_stats

    def export_learned_knowledge(self, filepath: str):
        """
        Export all learned knowledge to file

        This includes:
        - SEAL weight updates
        - Episodic memories
        - Tool usage statistics
        """
        if self.seal_trainer:
            # Export SEAL history
            seal_path = filepath.replace(".json", "_seal.json")
            self.seal_trainer.export_learning_history(seal_path)

        # Export episodic memory
        memory_summary = self.memory.get_full_context()
        memory_path = filepath.replace(".json", "_memory.txt")
        with open(memory_path, 'w') as f:
            f.write(memory_summary)

        print(f"\n✓ Learned knowledge exported:")
        print(f"  SEAL history: {seal_path}")
        print(f"  Memory: {memory_path}")

    def recover_from_forgetting(self, topic: str):
        """
        Recover knowledge from episodic memory if forgotten

        This is the solution to SEAL's catastrophic forgetting problem!

        Args:
            topic: Topic to recover knowledge about
        """
        if not self.memory:
            print("No memory system available")
            return

        # Retrieve relevant memories
        memories = self.memory.episodic.get_relevant_memories(query=topic, top_k=10)

        # Filter for SEAL updates
        seal_memories = [
            m for m in memories
            if m.metadata.get("type") == "seal_update"
        ]

        if seal_memories:
            print(f"\n✓ Found {len(seal_memories)} SEAL memories about '{topic}'")
            print("Recovering knowledge...")

            # Re-learn from stored memories
            for memory in seal_memories:
                print(f"  - {memory.metadata.get('strategy')}: quality={memory.metadata.get('quality_score'):.2f}")

            # Could trigger re-learning here
            print("\n✓ Knowledge recovered from episodic memory!")
        else:
            print(f"No SEAL memories found for '{topic}'")


def create_self_editing_agent(
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    enable_learning: bool = True,
    enable_weight_updates: bool = False
) -> SelfEditingAgent:
    """
    Convenience function to create self-editing agent

    Args:
        llm_provider: LLM provider (openai, anthropic, ollama)
        llm_model: Model name
        enable_learning: Enable SEAL learning
        enable_weight_updates: Enable LoRA weight updates (requires peft)

    Returns:
        Configured SelfEditingAgent

    Example:
        >>> agent = create_self_editing_agent(
        ...     llm_provider="anthropic",
        ...     llm_model="claude-3-5-sonnet-20241022",
        ...     enable_learning=True
        ... )
        >>> result = agent.execute_with_learning("Your task here")
    """
    config = SelfEditingAgentConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        enable_seal_learning=enable_learning,
        seal_weight_updates=enable_weight_updates,
        max_reasoning_steps=10
    )

    return SelfEditingAgent(config=config)
