import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Q-value JSON file (e.g., for episode 9900)
json_filename = "q_value/Stabilize/Q_Learning/Q_test_7  /Q_Learning_9900_19_15_5_5.json"
with open(json_filename, 'r') as f:
    data = json.load(f)

# Extract the Q-values dictionary (assumed to be under the key "q_values")
q_values_dict = data["q_values"]

# Convert the dictionary into a DataFrame.
# Rows represent states (keys in the dictionary) and columns represent the index within the Q-value lists.
df_q = pd.DataFrame.from_dict(q_values_dict, orient='index')

# Sort the DataFrame by state labels for consistent ordering
df_q.sort_index(inplace=True)

# Plot the heatmap using matplotlib
plt.figure(figsize=(12, 8))
heatmap = plt.imshow(df_q, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(heatmap, label='Q-value')
plt.xticks(ticks=np.arange(len(df_q.columns)), labels=df_q.columns, rotation=45)
plt.yticks(ticks=np.arange(len(df_q.index)), labels=df_q.index)
plt.xlabel("Q-value Index")
plt.ylabel("State")
plt.title("Heatmap of Q-values for Episode 9900")
plt.tight_layout()
plt.show()
