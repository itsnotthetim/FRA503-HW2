from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
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
        Initialize the Monte Carlo algorithm.

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
            control_type=ControlType.MONTE_CARLO,
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
        done,
        
    ):
        """
        Update Q-values using Monte Carlo.

        This method applies the Monte Carlo update rule to improve policy decisions by updating the Q-table.
        """
        state = self.discretize_state(obs)

        # Add state, action, and reward to history when episode is not done
        if not done:
            self.obs_hist.append(state)
            self.action_hist.append(action_idx)
            self.reward_hist.append(reward_value)
            return
        
        # Initialize return value
        G = 0  
        # Initialize list to store G values
        G_LIST = [0] * len(self.reward_hist)

        for i in reversed(range(len(self.reward_hist))):
            # Calculate the return value
            G = (self.discount_factor * G) + self.reward_hist[i]
            # Store the return value in the list
            G_LIST[i] = G

        # Check if the state and action has been visited before
        visited_states = set()
        for t in range(len(self.obs_hist)):
            state = self.obs_hist[t]
            action = self.action_hist[t]
            # Check if the state-action pair has been visited
            if (state, action) not in visited_states:
                visited_states.add((state, action))
                # Update Q-value for each state-action pair
                self.n_values[state][action] += 1
                self.q_values[state][action] += (G_LIST[t] - self.q_values[state][action]) / self.n_values[state][action]
        
        # Reset history
        self.obs_hist = []
        self.action_hist = []
        self.reward_hist = []