import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

# Define constants
IMAGE_HEIGHT, IMAGE_WIDTH = 75, 75
SEQUENCE_LENGTH = 20
CLASSES_LIST = ["Finger Flicking", "Hand Flapping", "Head Shaking"]

# Function to extract frames from the video
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
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    video_reader.release()
    
    # Pad the frames list if it has less than SEQUENCE_LENGTH frames.
    while len(frames_list) < SEQUENCE_LENGTH:
        frames_list.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

    return frames_list

# Function to load the model from a pickle file
def load_model_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        model_json, model_weights = pickle.load(f)
    
    model = model_from_json(model_json)
    model.set_weights(model_weights)
    return model

# Function to load the model and make predictions
def load_model_and_predict(video_path, pickle_file):
    # Load the trained model
    model = load_model_from_pickle(pickle_file)

    # Extract frames from the video
    frames = frames_extraction(video_path)

    if len(frames) != SEQUENCE_LENGTH:
        raise ValueError(f"Video does not have the required {SEQUENCE_LENGTH} frames.")

    frames = np.asarray(frames)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)

    prediction = model.predict(frames)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = CLASSES_LIST[predicted_class_index]

    return predicted_class_label

# Streamlit app
def main():
    st.title("Autism Detection from Video")
    st.write("Upload a video file to classify the type of autism-related behavior.")

    # Upload video file
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load model and make prediction
        try:
            pickle_file = 'autism_detection_inceptionv3_model.pkl'
            predicted_class = load_model_and_predict(temp_video_path, pickle_file)
            st.success(f'The predicted class for the video is: {predicted_class}')
        except Exception as e:
            st.error(f"Error: {e}")

        # Display the uploaded video
        st.video(uploaded_file)

if __name__ == "__main__":
    main()
