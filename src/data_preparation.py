import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def get_datapaths(dataset_name):
    dataset_path = os.path.join("data", dataset_name)
    train_path = os.path.join(dataset_path, "Train")
    test_path = os.path.join(dataset_path, "Test")
    train_input_path = os.path.join(train_path, "input")
    train_target_path = os.path.join(train_path, "target")
    test_input_path = os.path.join(test_path, "input")
    test_target_path = os.path.join(test_path, "target")

    train_input = os.listdir(train_input_path)
    train_target = os.listdir(train_target_path)
    test_input = os.listdir(test_input_path)
    test_target = os.listdir(test_target_path)

    print(f"Train input = {len(train_input)}, Train target = {len(train_target)},Test input = {len(test_input)}, Test target = {len(test_target)}")

    return train_input_path,train_target_path,test_input_path,test_target_path

def normalize(input, target):
    return tf.cast(input, tf.float32) / 255.0, tf.cast(target, tf.float32) / 255.0

def preprocess_dataset(input_path, target_path, img_size = (256,256), batch_size = 1 ):
    input = tf.keras.utils.image_dataset_from_directory(
        input_path,
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False)
    target = tf.keras.utils.image_dataset_from_directory(
        target_path,
        labels=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False)
    data = tf.data.Dataset.zip((input, target))

    data = data.map(normalize)
    data = data.prefetch(tf.data.AUTOTUNE)

    return data

def plot_sample_image(data, title):
    plt.figure(figsize=(10, 5))
    
    for input_images, target_images in data.take(1):
        input_image = input_images[0].numpy()
        target_image = target_images[0].numpy()
        
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title(f'{title} - Input (Low-light)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(target_image)
        plt.title(f'{title} - Target (Enhanced)')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()




