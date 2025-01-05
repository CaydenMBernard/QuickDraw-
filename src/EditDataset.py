import ndjson
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import random

def append_batch(file_path, batch_size):
    # Load the data from the file
    with open(file_path) as f:
        data = ndjson.load(f)
    
    # Randomly sample the data
    train_plus_test_batch_size = batch_size + (batch_size // 4)
    length = len(data)
    random_indices = random.sample(range(0, length), train_plus_test_batch_size)


    # Take irregular image data and convert it to a 256x256 image
    for i in range(len(random_indices)):
        drawing = data[random_indices[i]]['drawing']
        label = data[random_indices[i]]['word']
    
        image = Image.new('L', (256, 256), 0)
        draw = ImageDraw.Draw(image)
        for stroke in drawing:
            for j in range(len(stroke[0]) - 1):
                x0, y0 = int(stroke[0][j] * 256 / 255), int(stroke[1][j] * 256 / 255)
                x1, y1 = int(stroke[0][j + 1] * 256 / 255), int(stroke[1][j + 1] * 256 / 255)
                draw.line([x0, y0, x1, y1], fill=255, width=2)

        # Resize the image to 32x32 using bicubic interpolation and increase the contrast
        resized_image = image.resize((32, 32), Image.BICUBIC)
        enhancer = ImageEnhance.Contrast(resized_image)
        enhanced_image = enhancer.enhance(5.0)
        image_array = np.array(enhanced_image)

        # Append the image to either train or test dataset
        if i < batch_size:
            train_dataset.append({"label": label, "image": image_array})
        else:
            test_dataset.append({"label": label, "image": image_array})

        # Print the progress
        print(f"progress: {len(train_dataset)+len(test_dataset)} / {train_plus_test_batch_size * len(datasets)}", end="\r")

#def display_image(image):
    # Resize the image to 28x28 using bicubic interpolation
  #  resized_image = image.resize((32, 32), Image.BICUBIC)
    
    # Enhance the contrast of the image
 #   enhancer = ImageEnhance.Contrast(resized_image)
   # enhanced_image = enhancer.enhance(5.0)  # Increase the contrast

  #  image_array = np.array(enhanced_image)
   # print(image_array)
    
    # Display the image using matplotlib
  #  plt.imshow(enhanced_image, cmap='gray')
  #  plt.axis('off')
  #  plt.show()

if __name__ == "__main__":
    # Load the data from the file
    datasets = ["full_simplified_angel", "full_simplified_basketball", "full_simplified_car", "full_simplified_cat",
                 "full_simplified_crab" , "full_simplified_dolphin", "full_simplified_helicopter", "full_simplified_mushroom",
                 "full_simplified_octopus", "full_simplified_skull"]
    
    # Create the train and test datasets
    train_dataset = []
    test_dataset = []

    # Append the data to the datasets
    for dataset in datasets:
        file_path = f'DoodleData/{dataset}.ndjson'
        append_batch(file_path, 10000)

    # Save the datasets
    np.save("Doodle_Data_Train.npy", train_dataset)
    np.save("Doodle_Data_Test.npy", test_dataset)
    