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