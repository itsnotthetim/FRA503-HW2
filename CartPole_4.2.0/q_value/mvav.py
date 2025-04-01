import zipfile
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to the ZIP file containing the JSON files
zip_path = "CartPole_4.2.0/q_value/Q_Learning.zip"

# Initialize lists to store episode numbers and aggregated Q-values
episodes = []
avg_q_values = []

# Open and process each JSON file in the ZIP archive
with zipfile.ZipFile(zip_path, 'r') as z:
    # Filter for JSON files in the archive
    json_files = [f for f in z.namelist() if f.endswith('.json')]
    
    # Process each file
    for json_file in json_files:
        # Extract episode number from filename (assuming the first number is the episode)
        match = re.search(r'(\d+)', json_file)
        if match:
            episode = int(match.group(1))
        else:
            continue  # Skip file if no episode number found
        
        with z.open(json_file) as f:
            data = json.load(f)
            
            # Check if the JSON structure is as expected (i.e., has a "q_values" key)
            if "q_values" in data:
                q_dict = data["q_values"]
                # Flatten all Q-value lists into a single list
                all_q = []
                for key, q_list in q_dict.items():
                    # Ensure q_list is a list of numbers
                    all_q.extend(q_list)
                    
                # Compute an aggregate statistic (here, the mean)
                if all_q:
                    avg_q = np.mean(all_q)
                else:
                    avg_q = np.nan
                
                episodes.append(episode)
                avg_q_values.append(avg_q)

# Create a DataFrame for plotting and sort by episode
df = pd.DataFrame({
    "episode": episodes,
    "avg_q_value": avg_q_values
})
df.sort_values("episode", inplace=True)

# Plot the average Q-value per episode
plt.figure(figsize=(10, 6))
plt.plot(df["episode"], df["avg_q_value"], marker='o', linestyle='-', color='b')
plt.xlabel('Episode')
plt.ylabel('Average Q-value')
plt.title('Average Q-value per Episode')
plt.grid(True)
plt.tight_layout()
plt.show()
