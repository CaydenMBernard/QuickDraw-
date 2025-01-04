import os
import numpy as np

# Network parameters
num_layers = 5
input_size = 1024
hidden_1_size = 512
hidden_2_size = 256
hidden_3_size = 128
output_size = 10

# Initialize weights using He initialization
weights = [np.random.randn(hidden_1_size, input_size) * np.sqrt(2/input_size)]
weights.append(np.random.randn(hidden_2_size, hidden_1_size) * np.sqrt(2/hidden_1_size))
weights.append(np.random.randn(hidden_3_size, hidden_2_size) * np.sqrt(2/hidden_2_size))
weights.append(np.random.randn(output_size, hidden_3_size) * np.sqrt(2/hidden_3_size))

# Initialize biases
biases = [np.zeros(hidden_1_size)]
biases.append(np.zeros(hidden_2_size))
biases.append(np.zeros(hidden_3_size))
biases.append(np.zeros(output_size))  # Output layer biases

# Create the "Weights and Biases" folder
folder_path = os.path.join(os.path.dirname(__file__), "Weights and Biases")
os.makedirs(folder_path, exist_ok=True)

# Save weights and biases to the "Weights and Biases" folder
for i, w in enumerate(weights):
    np.save(os.path.join(folder_path, f'weight_layer_{i}.npy'), w)

for i, b in enumerate(biases):
    np.save(os.path.join(folder_path, f'bias_layer_{i}.npy'), b)

print(f"Weights and biases have been initialized and saved in {folder_path}.")
