
import pandas as pd
import numpy as np

# Define the number of samples in the dataset
num_samples = 300

# Generate synthetic data for shot_distance, shot_angle, and is_goal
np.random.seed(42)
shot_distance = np.random.uniform(5, 35, num_samples)  # Example: 5 to 35 meters
shot_angle = np.random.uniform(0, 360, num_samples)  # Example: 0 to 360 degrees
is_goal = np.random.choice([0, 1], num_samples, p=[0.5, 0.5])  # Example: 30% goals, 70% misses

# Create a DataFrame to store the dataset
data = pd.DataFrame({'shot_distance': shot_distance, 'shot_angle': shot_angle, 'is_goal': is_goal})

# Save the synthetic dataset to a CSV file
data.to_csv('synthetic_xg_dataset.csv', index=False)

# Print the first few rows of the dataset
print(data.head())
