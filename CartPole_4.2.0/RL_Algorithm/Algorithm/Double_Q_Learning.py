from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
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
        next_obs
        #========= put your code here =========#

        
    ):
        """
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule to improve policy decisions by updating the Q-table.
        """

        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        if np.random.rand() < 0.5:  # Update Q_A using Q_B for evaluation
            # If Update(A)
            # best_action = np.argmax(self.qa_values[next_state])
            # target = reward_value + (self.discount_factor * self.qb_values[next_state, best_action] * (1 - done))

            # self.qa_values[state, action_idx] += self.lr * (target - self.qa_values[state, action_idx])
            self.qa_values[state][action_idx] += self.lr * (reward_value + self.discount_factor * self.qb_values[next_state][np.argmax(self.qa_values[next_state])] - self.qa_values[state][action_idx])
        else:  # Update Q_B using Q_A for evaluation
            # If Update(B)
            # best_action = np.argmax(self.qb_values[next_state])
            # target = reward_value + (self.discount_factor * self.qa_values[next_state, best_action] * (1 - done))

            # self.qb_values[state, action_idx] += self.lr * (target - self.qb_values[state, action_idx])
            self.qb_values[state][action_idx] = self.lr * (reward_value + self.discount_factor * self.qa_values[next_state][np.argmax(self.qb_values[next_state])] - self.qb_values[state][action_idx])
        #======================================#