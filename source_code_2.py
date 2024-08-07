import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Flatten, Dense, Dropout

# Define constants
dataset_path = r'wlasl-complete/videos'
img_size = (100, 100)
num_samples = 2000

# Load data
video_data = []
labels = []
label_dict = {}
print(f"Dataset path: {dataset_path}")
#print(f"Files and folders in dataset path: {os.listdir(dataset_path)}")
'''for folder in os.listdir(dataset_path):
    
    folder_path = os.path.join(dataset_path, folder)
    label_dict[folder] = len(label_dict)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.mp4'):'''
#print(f"Dataset path: {dataset_path}")
video_files = os.listdir(dataset_path)
#print(f"Video files: {video_files}")
video_files = video_files[:num_samples]
for video_file in video_files:
    video_path = os.path.join(dataset_path, video_file)
    print(f"Trying to read video file: {video_path}")
    #... rest of the code...
                
    #print(f"Trying to read video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames in {video_path}: {num_frames}")
    while True:
        ret, frame = cap.read()
        print(f"Ret value: {ret}")
        if not ret:
            break
        print(f"Frame size: {frame.shape}")
        frame = cv2.resize(frame, img_size)
        frames.append(frame)
        if frames:
            print(f"Frames list is not empty: {len(frames)}")
            video_data.append(frames)
            #labels.append(label_dict[folder])
        else:
            print(f"Frames list is empty for file: {video_path}")
    cap.release()
    print(f"Frames extracted: {len(frames)}")  # Add this print statement
    if frames:  # Check if the frames list is not empty
        video_data.append(frames)
        #labels.append(label_dict[folder])
        print(f"Added video data and label to lists")  # Add this print statement

# Check if video_data is not empty
if not video_data:
    print("Error: video_data is empty. Please check the dataset path and file formats.")
    exit()

# Convert lists to numpy arrays
video_data = np.array(video_data)
labels = np.array(labels)

# Split data into training and validation sets
train_video_data, val_video_data, train_labels, val_labels = train_test_split(video_data, labels, test_size=0.2, random_state=42)

# Reshape data for ConvLSTM2D layer
train_video_data = train_video_data.reshape(-1, 1, *img_size, 3)
val_video_data = val_video_data.reshape(-1, 1, *img_size, 3)

# Define the model
model = Sequential()
model.add(ConvLSTM2D(40, (3, 3), activation='relu', input_shape=(1, *img_size, 3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2000, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(train_video_data, train_labels, epochs=10, validation_data=(val_video_data, val_labels))
model.save('sign_detection.h5')