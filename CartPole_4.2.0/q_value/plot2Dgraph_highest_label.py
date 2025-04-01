import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Q-values from JSON file
# file_path = "C:\D\Sheet_Lecture\DRL\HW1\FRA503-HW2-gun\CartPole_4.2.0\q_value\Stabilize\Double_Q_Learning\Double_Q_test_3\Double_Q_Learning_9900_19_15_5_5.json"  # Change this to your file path
file_path = "q_value/Stabilize/MC/MC_test_1/MC_9900_19_15_5_5.json"
# file_path = "C:\D\Sheet_Lecture\DRL\HW1\FRA503-HW2-gun\CartPole_4.2.0\q_value_main\Stabilize\MC\MC_1\MC_5900_19_20_5_5.json"
# file_path = "C:\D\Sheet_Lecture\DRL\HW1\FRA503-HW2-gun\CartPole_4.2.0\q_value_main\Stabilize\MC\MC_test_3\MC_9700_19_15_5_5.json"
# file_path = "C:\D\Sheet_Lecture\DRL\HW1\FRA503-HW2-gun\CartPole_4.2.0\q_value_main\Stabilize\Q_Learning\Q_test_7__\Q_Learning_9900_19_15_5_5.json"
with open(file_path, "r") as file:
    data = json.load(file)

q_values = data["q_values"]

# Extract unique cart and pole positions
cart_positions = set()
pole_positions = set()
q_table = {}

for state, q_list in q_values.items():
    state_tuple = eval(state)  # Convert string key to tuple
    pose_cart, pose_pole = state_tuple[:2]  # Extract Cart & Pole positions

    cart_positions.add(pose_cart)
    pole_positions.add(pose_pole)

    # Store the max Q-value for this (pose_cart, pose_pole)
    q_table[(pose_cart, pose_pole)] = max(q_list)

# Sort positions for consistent axis labels
cart_positions = sorted(cart_positions)
pole_positions = sorted(pole_positions)

# Create 2D array for heatmap
heatmap_data = np.zeros((len(pole_positions), len(cart_positions)))

for i, pole in enumerate(pole_positions):
    for j, cart in enumerate(cart_positions):
        heatmap_data[i, j] = q_table.get((cart, pole), 0)  # Default to 0 if missing

# Plot heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", xticklabels=cart_positions, yticklabels=pole_positions)

# Labels and title
plt.xlabel("Cart Position")
plt.ylabel("Pole Position")
plt.title("Monte Carlo 2D Heatmap of Max Q-Values")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()
