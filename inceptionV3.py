import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import pickle

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Specify the height and width.
IMAGE_HEIGHT, IMAGE_WIDTH = 75, 75  # Updated to match InceptionV3 minimum size

# Specify the number sequence.
SEQUENCE_LENGTH = 20

# Specify the dataset path. 
DATASET_Path = '/home/bharath31/Desktop/autisum detection/video samples'

# Specify the list containing the names classes.
CLASSES_LIST = ["Finger Flicking", "Hand Flapping", "Head Shaking"]

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = preprocess_input(resized_frame)
        frames_list.append(normalized_frame)

    video_reader.release()
    
    # Pad the frames list if it has less than SEQUENCE_LENGTH frames.
    while len(frames_list) < SEQUENCE_LENGTH:
        frames_list.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    return frames_list

def create_dataset():
    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(DATASET_Path, class_name))

        for file_name in files_list:
            video_file_path = os.path.join(DATASET_Path, class_name, file_name)
            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels, video_files_paths

# Create dataset
features, labels, video_files_paths = create_dataset()

# One-hot encoding
one_hot_encoded_labels = to_categorical(labels)

# Split test and train
features_train, features_test, labels_train, labels_test = train_test_split(
    features, one_hot_encoded_labels, test_size=0.25, shuffle=True, random_state=seed_constant
)

def create_inceptionv3_model():
    inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    inceptionv3.trainable = False  # Freeze the InceptionV3 layers

    model = Sequential()
    model.add(TimeDistributed(inceptionv3, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))

    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()
    return model

# Construct the required InceptionV3 model.
inceptionv3_model = create_inceptionv3_model()

# Display the success message. 
print("Model Created Successfully!")

# Create an Instance of Early Stopping Callback.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

# Compile the model and specify loss function.
inceptionv3_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=["accuracy"])

# Start training the model.
inceptionv3_model_training_history = inceptionv3_model.fit(
    x=features_train, y=labels_train, epochs=50, batch_size=4, shuffle=True, validation_split=0.2, callbacks=[early_stopping_callback]
)

# Evaluate the model.
model_evaluation_history = inceptionv3_model.evaluate(features_test, labels_test)

# Save the model to a pickle file.
model_json = inceptionv3_model.to_json()
model_weights = inceptionv3_model.get_weights()

with open('autism_detection_inceptionv3_model.pkl', 'wb') as f:
    pickle.dump((model_json, model_weights), f)
print(f"Model saved to autism_detection_inceptionv3_model.pkl")

# Function to load the model from a pickle file.
def load_model_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        model_json, model_weights = pickle.load(f)
    
    model = model_from_json(model_json)
    model.set_weights(model_weights)
    return model

# Function to load the model and make predictions.
def load_model_and_predict(video_path, pickle_file):
    # Load the trained model.
    model = load_model_from_pickle(pickle_file)

    # Extract frames from the video.
    frames = frames_extraction(video_path)

    if len(frames) != SEQUENCE_LENGTH:
        raise ValueError(f"Video does not have the required {SEQUENCE_LENGTH} frames.")

    frames = np.asarray(frames)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    prediction = model.predict(frames)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CLASSES_LIST[predicted_class_index]

    return predicted_class_label

# Path to the new video.
new_video_path = 'trash:///Copy%20of%20Hand_flapping48.mp4'

# Predict the class of the new video using the saved model.
predicted_class = load_model_and_predict(new_video_path, 'autism_detection_inceptionv3_model.pkl')

# Display the predicted class.
print(f'The predicted class for the video is: {predicted_class}')
