import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1) Load the Q-value JSON file
# ------------------------------------------------
file_path = "CartPole_4.2.0/q_value/Stabilize/MC/MC_test_1-1/MC_8000_19_15_5_5.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract maximum Q-values for each (pose_cart, pose_pole) pair
q_values = data["q_values"]
q_dict = {}

for key, q_list in q_values.items():
    # Convert string key (e.g. "(-1, 0, 1, 2)") into a tuple
    pose_cart, pose_pole, vel_cart, vel_pole = eval(key)
    max_q = max(q_list)  # Maximum Q-value among actions
    # We only store (pose_cart, pose_pole) -> max Q
    q_dict[(pose_cart, pose_pole)] = max_q

# ------------------------------------------------
# 2) Convert to numpy arrays for 2D heatmap
# ------------------------------------------------
# We'll collect all unique cart/pole positions
x_vals = np.array([k[0] for k in q_dict.keys()])  # pose_cart
y_vals = np.array([k[1] for k in q_dict.keys()])  # pose_pole

x_unique = np.unique(x_vals)
y_unique = np.unique(y_vals)

# Create a 2D grid
# shape: (len(y_unique), len(x_unique))
X, Y = np.meshgrid(x_unique, y_unique)

# Initialize a 2D array (Z) for the heatmap
Z = np.full_like(X, np.nan, dtype=float)

# Fill Z with the corresponding max Q-value, if it exists
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        # meshgrid means: X[i, j] is the cart pos, Y[i, j] is the pole pos
        pc = X[i, j]
        pp = Y[i, j]
        # Look up the Q-value in our dictionary
        Z[i, j] = q_dict.get((pc, pp), np.nan)

# ------------------------------------------------
# 3) Plot the 2D Heatmap
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Use imshow to render the 2D matrix Z as a heatmap
heatmap = ax.imshow(
    Z,
    origin='lower',   # ensures row 0 is at bottom
    cmap='viridis',
    aspect='auto'     # so we can see the entire range
)

# Configure axis labels
ax.set_title("2D Heatmap of Max Q-Values")
ax.set_xlabel("Cart Position")
ax.set_ylabel("Pole Position")

# Use the sorted unique values as tick labels
ax.set_xticks(np.arange(len(x_unique)))
ax.set_xticklabels(x_unique)
ax.set_yticks(np.arange(len(y_unique)))
ax.set_yticklabels(y_unique)

# Add a colorbar
cbar = plt.colorbar(heatmap, ax=ax)
cbar.set_label("Max Q-Value")

plt.tight_layout()
plt.show()
