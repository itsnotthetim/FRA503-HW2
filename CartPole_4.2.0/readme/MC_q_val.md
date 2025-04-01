## Result Interpretation: GLiE Monte Carlo
![MC2D](image/MC_2D2.png)
![MC3D](image/MC_3D2.png)

From these figures, each grid cell (in the 2D heatmap) or point on the surface (in the 3D plot) corresponds to a discretized 
(cart position , pole angle) state in the CartPole environment. The color scale (ranging from deep purple to bright yellow) and the vertical axis in the 3D plot represent the learned maximum action‐value 
𝑄( 𝑠, 𝑎 ) for that state, as estimated by Monte Carlo (MC) algorithm with GLiE exploration. Higher (yellow) values of 𝑄 means states from which the agent expects to achieve greater long‐term returns (more time balancing the pole without termination).


#### Analysis of Q-Value 


- High Q in Near‐Balanced States:

    States in which the cart is near the center (around  $x \approx 0$ ) and the pole angle is near upright ($\theta \approx 0 $) show consistently large Q-values (0.8 to 1.0 in heatmap). This reflects **the agent’s learned expectation** that maintaining and adjustment the pole close to vertical and the cart near the origin leads to higher cumulative reward before termination.

- Low Q in Extreme States

    Once the pole deviates significantly from the vertical ( $\theta \approx \pm  2$ radians ) or the cart moves too far to the sides ( $x \approx \pm 10$ to  $\pm 15 $ ), the Q-values become small (dark purple). These regions correspond to near-terminal or unrecoverable conditions (can't be stabilized) in CartPole, where the environment will end quickly, leading to a low return 

- Smoothness vs. Sharp Peaks

    The 3D surface plot shows that most states have medium Q-values, but some states stand out with high "peaks" in value. These peaks represent very good states for the agent. This pattern  methods update Q-values using the total rewards from full episodes. As a result, states that often lead to successful balancing tend to have higher Q-values over time.


#### Summary

These plots clearly show that the GLiE Monte Carlo agent has learned to favor states where the cart stays near the center and the pole remains upright. In these balanced positions, the agent expects to receive higher cumulative rewards, which is why these areas appear as bright yellow or tall peaks on the Q-value plots. In contrast, when the agent is in states where the cart or pole is far from this equilibrium, it anticipates the episode will end soon and gives those states a low value represented by dark purple regions.

---

