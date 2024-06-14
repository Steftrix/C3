import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Define paths
dataset_path = "Date"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "validation")
test_path = os.path.join(dataset_path, "test")

# Create directories if they don't exist
for path in [train_path, val_path, test_path]:
    os.makedirs(path, exist_ok=True)

# List and sort breeds
breed_list = sorted(os.listdir(os.path.join(dataset_path, 'Annotation')))

# Create label index for easy lookup
label2index = {name: index for index, name in enumerate(breed_list)}
index2label = {index: name for index, name in enumerate(breed_list)}

# Prepare lists of images and annotations
images, annotations = [], []
for breed in breed_list:
    image_files = sorted(os.listdir(os.path.join(dataset_path, 'Images', breed)))
    images.extend([os.path.join(dataset_path, 'Images', breed, f) for f in image_files])
    annotations.extend([breed] * len(image_files))

# Convert lists to numpy arrays and shuffle data
Xs, Ys = np.asarray(images), np.asarray(annotations)
indices = np.arange(len(Xs))
np.random.shuffle(indices)
Xs, Ys = Xs[indices], Ys[indices]

# Split data into train_validate + test data
split1 = int(0.9 * len(Xs))
train_validate_x, test_x = Xs[:split1], Xs[split1:]
train_validate_y, test_y = Ys[:split1], Ys[split1:]

# Split train_validate into train and validation data
split2 = int(0.9 * len(train_validate_x))
train_x, val_x = train_validate_x[:split2], train_validate_x[split2:]
train_y, val_y = train_validate_y[:split2], train_validate_y[split2:]

# Copy images to respective directories
def copy_files(file_list, label_list, target_dir):
    for file, label in zip(file_list, label_list):
        breed = file.split(os.sep)[-2]
        breed_dir = os.path.join(target_dir, breed)
        os.makedirs(breed_dir, exist_ok=True)
        shutil.copy(file, os.path.join(breed_dir, os.path.basename(file)))

copy_files(train_x, train_y, train_path)
copy_files(val_x, val_y, val_path)
copy_files(test_x, test_y, test_path)

# View a few train images
fig = plt.figure(figsize=(15, 10))
for idx in range(9):
    sample_input = Image.open(train_x[idx])
    breed = train_y[idx]
    axs = fig.add_subplot(3, 3, idx + 1)
    axs.set_title(breed)
    plt.imshow(sample_input)
    plt.axis('off')
plt.show()

# Parameters for data generation
image_width, image_height, num_channels = 128, 128, 3
num_classes = len(breed_list)

epochs = 30
train_batch_size, validation_batch_size, test_batch_size = 32, 32, 32

# Convert labels to binary class matrix (One-hot-encoded)
train_processed_y = to_categorical([label2index[label] for label in train_y], num_classes=num_classes).astype('float32')
validate_processed_y = to_categorical([label2index[label] for label in val_y], num_classes=num_classes).astype('float32')
test_processed_y = to_categorical([label2index[label] for label in test_y], num_classes=num_classes).astype('float32')

train_data_count = len(train_x)
steps_per_epoch = train_data_count // train_batch_size
validation_data_count = len(val_x)
validation_steps = validation_data_count // validation_batch_size

# Pre-processing functions
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize(image, [image_height, image_width])
    return image, label

def normalize(image, label):
    image = image / 255.0
    return image, label

def build_data_generators(train_data_process_list=[load_image, normalize], validate_data_process_list=[load_image, normalize], test_data_process_list=[load_image, normalize]):
    # Create TF Dataset
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_processed_y))
    validation_data = tf.data.Dataset.from_tensor_slices((val_x, validate_processed_y))
    test_data = tf.data.Dataset.from_tensor_slices((test_x, test_processed_y))

    # Apply pre-processing and batching
    for process in train_data_process_list:
        train_data = train_data.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(train_data_count).repeat(epochs).batch(train_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    for process in validate_data_process_list:
        validation_data = validation_data.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    validation_data = validation_data.shuffle(validation_data_count).repeat(epochs).batch(validation_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    for process in test_data_process_list:
        test_data = test_data.map(process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data = test_data.batch(test_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_data, validation_data, test_data

# Build data generators
train_data, validation_data, test_data = build_data_generators()

print("train_data", train_data)
print("validation_data", validation_data)
print("test_data", test_data)

# Export data for main.py
def get_data():
    return train_data, validation_data, test_data, steps_per_epoch, validation_steps, label2index, index2label, test_x, test_y

if __name__ == "__main__":
    get_data()
