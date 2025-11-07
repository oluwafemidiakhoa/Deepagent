"""
ToolPO - Tool Policy Optimization

Reinforcement learning framework for optimizing tool usage patterns.
Implements PPO-based optimization with token-level advantage attribution.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json


@dataclass
class ToolAction:
    """Represents a tool action in the policy"""
    tool_name: str
    parameters: Dict[str, Any]
    context: str
    log_prob: float = 0.0
    value_estimate: float = 0.0


@dataclass
class TrajectoryStep:
    """Single step in an episode trajectory"""
    state: str
    action: ToolAction
    reward: float
    next_state: str
    done: bool
    advantage: float = 0.0
    return_value: float = 0.0


@dataclass
class Episode:
    """Complete episode of agent interaction"""
    task: str
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False

    def add_step(self, step: TrajectoryStep) -> None:
        self.trajectory.append(step)
        self.total_reward += step.reward


class RewardModel:
    """
    Reward model for tool usage

    Assigns rewards based on:
    - Tool execution success
    - Relevance to task
    - Efficiency (fewer steps preferred)
    """

    def __init__(self):
        self.success_reward = 10.0
        self.failure_penalty = -5.0
        self.step_penalty = -0.1
        self.relevance_weight = 2.0

    def compute_reward(
        self,
        action: ToolAction,
        result: Any,
        task: str,
        success: bool
    ) -> float:
        """
        Compute reward for a tool action

        Args:
            action: The tool action taken
            result: Result of tool execution
            task: Original task description
            success: Whether execution was successful

        Returns:
            Reward value
        """
        reward = 0.0

        # Base reward for success/failure
        if success:
            reward += self.success_reward
        else:
            reward += self.failure_penalty

        # Penalty for taking a step (encourage efficiency)
        reward += self.step_penalty

        # Reward for relevance (simplified semantic matching)
        relevance = self._compute_relevance(action.tool_name, task)
        reward += relevance * self.relevance_weight

        return reward

    def _compute_relevance(self, tool_name: str, task: str) -> float:
        """Compute relevance of tool to task (simplified)"""
        # Simple word overlap metric
        tool_words = set(tool_name.lower().split("_"))
        task_words = set(task.lower().split())

        overlap = len(tool_words & task_words)
        return min(overlap / max(len(tool_words), 1), 1.0)

    def compute_terminal_reward(self, success: bool, num_steps: int) -> float:
        """Compute final reward at episode end"""
        if success:
            # Bonus for successful completion
            base_reward = 50.0
            # Bonus for efficiency
            efficiency_bonus = max(0, 20.0 - num_steps * 0.5)
            return base_reward + efficiency_bonus
        else:
            # Penalty for failure
            return -20.0


class AdvantageEstimator:
    """
    Compute advantages using Generalized Advantage Estimation (GAE)
    """

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_advantages(
        self,
        trajectory: List[TrajectoryStep],
        value_function: callable
    ) -> List[TrajectoryStep]:
        """
        Compute advantages for trajectory using GAE

        Args:
            trajectory: List of trajectory steps
            value_function: Function to estimate state values

        Returns:
            Updated trajectory with advantages
        """
        # Compute TD residuals
        td_residuals = []
        for i, step in enumerate(trajectory):
            if step.done:
                next_value = 0.0
            else:
                next_value = value_function(step.next_state)

            current_value = value_function(step.state)
            td_residual = step.reward + self.gamma * next_value - current_value
            td_residuals.append(td_residual)

        # Compute GAE advantages
        advantages = []
        gae = 0.0

        for i in reversed(range(len(trajectory))):
            td_residual = td_residuals[i]
            gae = td_residual + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)

        # Update trajectory with advantages
        for i, step in enumerate(trajectory):
            step.advantage = advantages[i]
            step.return_value = advantages[i] + value_function(step.state)

        return trajectory


class ToolPolicyOptimizer:
    """
    Main ToolPO optimizer using PPO algorithm

    Optimizes policy for tool selection and usage
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.reward_model = RewardModel()
        self.advantage_estimator = AdvantageEstimator()

        # Training statistics
        self.training_stats = defaultdict(list)

    def compute_policy_loss(
        self,
        old_log_probs: np.ndarray,
        new_log_probs: np.ndarray,
        advantages: np.ndarray
    ) -> float:
        """
        Compute clipped PPO policy loss

        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Computed advantages

        Returns:
            Policy loss value
        """
        # Compute probability ratio
        ratio = np.exp(new_log_probs - old_log_probs)

        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = np.clip(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        return policy_loss

    def compute_value_loss(
        self,
        predicted_values: np.ndarray,
        target_returns: np.ndarray
    ) -> float:
        """Compute value function loss"""
        return np.mean((predicted_values - target_returns) ** 2)

    def compute_entropy_bonus(self, log_probs: np.ndarray) -> float:
        """Compute entropy bonus for exploration"""
        # Entropy = -sum(p * log(p))
        probs = np.exp(log_probs)
        entropy = -np.sum(probs * log_probs)
        return entropy

    def update_policy(
        self,
        episodes: List[Episode],
        policy_network: callable,
        value_network: callable,
        num_epochs: int = 4
    ) -> Dict[str, float]:
        """
        Update policy using collected episodes

        Args:
            episodes: List of collected episodes
            policy_network: Policy network (tool selector)
            value_network: Value network (state evaluator)
            num_epochs: Number of optimization epochs

        Returns:
            Dictionary of training metrics
        """
        # Compute advantages for all episodes
        all_trajectories = []
        for episode in episodes:
            trajectory_with_advantages = self.advantage_estimator.compute_advantages(
                episode.trajectory,
                value_network
            )
            all_trajectories.extend(trajectory_with_advantages)

        # Extract data for training
        states = [step.state for step in all_trajectories]
        actions = [step.action for step in all_trajectories]
        old_log_probs = np.array([step.action.log_prob for step in all_trajectories])
        advantages = np.array([step.advantage for step in all_trajectories])
        returns = np.array([step.return_value for step in all_trajectories])

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Training loop
        metrics = defaultdict(list)

        for epoch in range(num_epochs):
            # Get new policy predictions
            new_log_probs = np.array([
                policy_network(state, action) for state, action in zip(states, actions)
            ])

            # Get value predictions
            predicted_values = np.array([value_network(state) for state in states])

            # Compute losses
            policy_loss = self.compute_policy_loss(old_log_probs, new_log_probs, advantages)
            value_loss = self.compute_value_loss(predicted_values, returns)
            entropy = self.compute_entropy_bonus(new_log_probs)

            # Total loss
            total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )

            # Record metrics
            metrics["policy_loss"].append(float(policy_loss))
            metrics["value_loss"].append(float(value_loss))
            metrics["entropy"].append(float(entropy))
            metrics["total_loss"].append(float(total_loss))

        # Compute average metrics
        avg_metrics = {
            key: np.mean(values) for key, values in metrics.items()
        }

        # Add episode statistics
        avg_metrics["avg_episode_reward"] = np.mean([ep.total_reward for ep in episodes])
        avg_metrics["success_rate"] = np.mean([ep.success for ep in episodes])
        avg_metrics["avg_episode_length"] = np.mean([len(ep.trajectory) for ep in episodes])

        # Store in training stats
        for key, value in avg_metrics.items():
            self.training_stats[key].append(value)

        return avg_metrics

    def simulate_rollout(
        self,
        agent: Any,
        task: str,
        max_steps: int = 20
    ) -> Episode:
        """
        Simulate agent rollout for a task

        This creates training data by running the agent
        and recording trajectories with rewards.

        Args:
            agent: DeepAgent instance
            task: Task to perform
            max_steps: Maximum steps per episode

        Returns:
            Episode with trajectory and rewards
        """
        episode = Episode(task=task)

        # Run agent and collect trajectory
        result = agent.run(task, max_steps=max_steps)

        # Convert reasoning trace to trajectory
        for i, trace in enumerate(result.reasoning_trace):
            if trace.tool_name:
                # This was a tool execution step
                action = ToolAction(
                    tool_name=trace.tool_name,
                    parameters={},  # Would extract from trace
                    context=trace.content,
                    log_prob=0.0  # Would come from policy network
                )

                # Compute reward
                reward = self.reward_model.compute_reward(
                    action=action,
                    result=trace.tool_result,
                    task=task,
                    success=trace.tool_result is not None
                )

                step = TrajectoryStep(
                    state=trace.content,
                    action=action,
                    reward=reward,
                    next_state="" if i == len(result.reasoning_trace) - 1 else result.reasoning_trace[i+1].content,
                    done=(i == len(result.reasoning_trace) - 1)
                )

                episode.add_step(step)

        # Add terminal reward
        terminal_reward = self.reward_model.compute_terminal_reward(
            success=result.success,
            num_steps=result.total_steps
        )

        if episode.trajectory:
            episode.trajectory[-1].reward += terminal_reward

        episode.success = result.success
        episode.total_reward = sum(step.reward for step in episode.trajectory)

        return episode

    def train(
        self,
        agent: Any,
        tasks: List[str],
        num_iterations: int = 100,
        episodes_per_iteration: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train agent using ToolPO

        Args:
            agent: DeepAgent to train
            tasks: List of training tasks
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration

        Returns:
            Training statistics
        """
        print(f"Starting ToolPO training for {num_iterations} iterations...")

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")

            # Collect episodes
            episodes = []
            for _ in range(episodes_per_iteration):
                task = np.random.choice(tasks)
                episode = self.simulate_rollout(agent, task)
                episodes.append(episode)

            # Mock policy and value networks (in production, use actual neural networks)
            def mock_policy_network(state: str, action: ToolAction) -> float:
                return -1.0  # Log probability

            def mock_value_network(state: str) -> float:
                return 0.0  # State value estimate

            # Update policy
            metrics = self.update_policy(
                episodes=episodes,
                policy_network=mock_policy_network,
                value_network=mock_value_network
            )

            # Print metrics
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Avg Reward: {metrics['avg_episode_reward']:.2f}")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

        return dict(self.training_stats)

    def save_stats(self, filepath: str) -> None:
        """Save training statistics"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.training_stats), f, indent=2)
