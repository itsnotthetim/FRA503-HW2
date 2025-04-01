import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the JSON file

file_path = "CartPole_4.2.0/q_value/Stabilize/MC/MC_test_1-1/MC_8500_19_15_5_5.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract maximum Q-values for each (pose_cart, pose_pole) pair
q_values = data["q_values"]
q_dict = {}

for key, q_list in q_values.items():
    pose_cart, pose_pole, vel_cart, vel_pole = eval(key)  # Convert string key to tuple
    max_q = max(q_list)  # Get the maximum Q-value for this state
    q_dict[(pose_cart, pose_pole)] = max_q

# Convert to numpy arrays for plotting
x_vals = np.array([key[0] for key in q_dict.keys()])  # Cart position
y_vals = np.array([key[1] for key in q_dict.keys()])  # Pole position
z_vals = np.array(list(q_dict.values()))  # Max Q-values

# Create a grid for surface plotting
X, Y = np.meshgrid(np.unique(x_vals), np.unique(y_vals))
Z = np.zeros_like(X, dtype=float)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = q_dict.get((X[i, j], Y[i, j]), np.nan)  # Fill Z with Q-values

# Plot 3D surface with improved visualization
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add color bar
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Max Q-Value")

# Labels and title
ax.set_xlabel("Cart Position")
ax.set_ylabel("Pole Position")
ax.set_zlabel("Max Q-Value")
ax.set_title("3D Surface Plot of Q-Values")

# Show plot
plt.show()