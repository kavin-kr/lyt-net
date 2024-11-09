import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from data_preparation import *

#Get dataset name from user
dataset_name = input("Enter the dataset name (LOLv1, LOLv2_real, LOLv2_synthetic): ")
img_size = (256, 256)
batch_size = 32

#Get the data path, preprocess it and then plot a sample image pair from train and test
train_input_path,train_target_path,test_input_path,test_target_path = get_datapaths(dataset_name)
train_data = preprocess_dataset(train_input_path, train_target_path, img_size, batch_size )
test_data = preprocess_dataset(test_input_path, test_target_path, img_size, batch_size )
plot_sample_image(train_data, "Training Data")
plot_sample_image(test_data, "Test Data")




