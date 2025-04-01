# FRA503-HW2: Stabilizing CartPole

## Overview
 In this homework, we will work on the Stabilizing Cart-Pole Task, where the goal is to train the agent with learning algorithms (i.e. Q-Learning, Monte-Carlo, Temporal Difference Learning, and Double Q-Learning) to control a cart moving along a frictionless track to keep a pole balanced. The pole starts near an upright position (close to 90° vertical), and the agent must apply the right forces to the cart to prevent it from tipping over. The challenge is to stabilize the system while minimizing unnecessary movement. The episode ends if the pole leans too far or the cart moves too far from the center.

## Objective

You must evaluate the agent's performance in terms of learning efficiency (i.e., how well the agent learns to receive higher rewards) and deployment performance (i.e., how well the agent performs in the Cart-Pole problem). Analyze and visualize the results to determine:

- Which algorithm performs best?
- Why does it perform better than the others?
- How do the resolutions of the action space and observation space affect the learning process? Why?

## 1. Which algorithm performs best?
### 1.1 Test all algorithm
#### Assumption
1. Every algorithm have same parameter
   
python
    # hyperparameters
    num_of_action = 19
    action_range = [-15, 15]  # [min, max]
    discretize_state_weight = [5, 5, 2, 2]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.1
    n_episodes = 10000
    start_epsilon = 1.0
    epsilon_decay = 0.9995  # reduce the exploration over time
    final_epsilon = 0.01
    discount = 0.99
   
### How well the agent learns to receive higher rewards


## 2. Why does it perform better than the others?

### How well the agent performs in the Cart-Pole problem

## 3. How do the resolutions of the action space and observation space affect the learning process? Why?
# FRA503-HW2: Stabilizing CartPole

## Overview
 In this homework, we will work on the Stabilizing Cart-Pole Task, where the goal is to train the agent with learning algorithms (i.e. Q-Learning, Monte-Carlo, Temporal Difference Learning, and Double Q-Learning) to control a cart moving along a frictionless track to keep a pole balanced. The pole starts near an upright position (close to 90° vertical), and the agent must apply the right forces to the cart to prevent it from tipping over. The challenge is to stabilize the system while minimizing unnecessary movement. The episode ends if the pole leans too far or the cart moves too far from the center.

## Objective

You must evaluate the agent's performance in terms of learning efficiency (i.e., how well the agent learns to receive higher rewards) and deployment performance (i.e., how well the agent performs in the Cart-Pole problem). Analyze and visualize the results to determine:

- Which algorithm performs best?
- Why does it perform better than the others?
- How do the resolutions of the action space and observation space affect the learning process? Why?

## 1. Which algorithm performs best?
### 1.1 Test all algorithm
#### Assumption
1. RL-base has the parameters to use at each algorithm but only Monte-Carlo (MC) is not using learning rate in the equation
   
    ```python
        # hyperparameters
        num_of_action = 19
        action_range = [-15, 15]  # [min, max]
        discretize_state_weight = [5, 5, 2, 2]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate = 0.1
        n_episodes = 10000
        start_epsilon = 1.0
        epsilon_decay = 0.9995  # reduce the exploration over time
        final_epsilon = 0.01
        discount = 0.99
    ```
---

##### 1. `num_of_action (int)`

- **Definition**: The number of discrete actions your agent can select at each timestep.  
- **Interpretation**: In a cartpole-like environment, a higher `num_of_action` might capture more fine-grained force commands (e.g., 19 possible forces instead of just left vs. right).

---

##### 2. `action_range (list)`

- **Definition**: A numeric range (e.g., $ [-15, 15] $) mapping discrete action indices to real-valued actions in a continuous action space.   
- **Interpretation**:
    - A **wider** range allows higher forces but can lead to more variance in outcomes.  
    - A **narrower** range might stabilize training but limit maximum performance.

---

#### 3. `discretize_state_weight (list)`

- **Definition**: A list of scaling factors used to **discretize** each dimension of the (continuous) observation vector. For example, in a 4-dimensional state $[x, \dot{x}, \theta, \dot{\theta}] $ (cart position, cart velocity, pole angle, pole angular velocity), you might assign separate discrete “bins” for each.  
- **Interpretation**:  
    - **Higher** weights or more bins produce a **finer** resolution but **larger** Q-tables (slower learning).  
    - **Lower** weights create a **coarser** representation, possibly sacrificing optimal control for faster convergence.

---

##### 4. `learning_rate (float)`

- **Definition**: A factor typically denoted $\alpha$ that controls how quickly your Q-values are updated.  

- **Interpretation**: 
    - A high learning rate means Q-values change rapidly, while a low learning rate may stabilize training but slow convergence.

---

##### 5. `n_episodes (int)`

- **Definition**: The number of episodes the agent will experience (simulate) during training.  
- **Interpretation**: Longer training (more episodes) usually improves policy estimation, but there may be diminishing returns if the environment is fully explored or if hyperparameters are not well tuned.

---

##### 6. `start_epsilon (float)`

- **Definition**: The initial $\epsilon$ value in an **$\epsilon$-greedy** exploration strategy. $\epsilon$ indicates the probability of choosing a random action instead of the greedy (best) action according to the current Q-values.  
- **Interpretation**: Typically set near 1.0 (100% random at the very beginning). The agent then decays $\epsilon$ to gradually shift from exploration to exploitation.

---

##### 7. `epsilon_decay (float)`

- **Definition**: The factor by which $\epsilon$ is multiplied per episode (or per step, depending on your code) to reduce random exploration over time.  
- **Interpretation**:
    - **Decay too fast**: The agent may not explore enough, converging to a suboptimal policy.  
    - **Decay too slow**: The agent wastes opportunities to exploit an improving policy.

---

##### 8. `final_epsilon (float)`

- **Definition**: The minimum $\epsilon$ value. This ensures the agent **always** maintains some level of random exploration, never going fully deterministic.  

---

##### 9. `discount (float)` (commonly $\gamma$\)

- **Definition**: A factor in $[0, 1]$ that discounts future rewards relative to immediate rewards.    
- **Interpretation**:
    - **$\gamma \approx 1.0$**: Long-term rewards strongly influence Q-values. Useful for tasks requiring extended planning.  
    - **$\gamma \ll 1.0$**: The agent cares more about rewards it gets soon, so it focuses less on the future.

---


   
   
### How well the agent learns to receive higher rewards


## 2. Why does it perform better than the others?

### How well the agent performs in the Cart-Pole problem

## 3. How do the resolutions of the action space and observation space affect the learning process? Why?