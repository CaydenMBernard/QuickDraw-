import numpy as np
import matplotlib.pyplot as plt
import random

loaded_data = np.load("Doodle_Data_Train.npy", allow_pickle=True).tolist()
random_index = random.randint(0, len(loaded_data)-1)
image_label = loaded_data[random_index]["label"]
image_array = loaded_data[random_index]["image"]
loaded_data = random.shuffle(loaded_data)
print(len(loaded_data))
print(image_label)
print(image_array)

plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
plt.title(image_label, fontsize=20)
plt.axis('off')
plt.show()