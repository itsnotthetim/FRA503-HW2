from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class SARSA(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the SARSA algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.TEMPORAL_DIFFERENCE,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs,
        action_idx,
        reward_value,
        # done,
        next_obs, 
        next_action_idx
        
    ):
        """
        Update Q-values using SARSA .

        This method applies the SARSA update rule to improve policy decisions by updating the Q-table.
        """
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        self.q_values[state][action_idx] += self.lr * (reward_value + (self.discount_factor * self.q_values[next_state][next_action_idx]) - self.q_values[state][action_idx])