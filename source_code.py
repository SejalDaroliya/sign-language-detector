import tensorflow as tf
from tensorflow import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Define the dataset path and labels
dataset_path = 'asl_dataset'
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
          'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
          'u', 'v', 'w', 'x', 'y', 'z']  # list of labels in the same order as the folders

# Define the image dimensions
img_height, img_width = 224, 224

# Create an ImageDataGenerator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create training and validation generators
train_generator = train_datagen.flow_from_directory(dataset_path, target_size=(img_height, img_width), batch_size=32, class_mode='categorical', classes=labels)
validation_generator = validation_datagen.flow_from_directory(dataset_path, target_size=(img_height, img_width), batch_size=32, class_mode='categorical', classes=labels)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save('asl_recognition_model.h5')