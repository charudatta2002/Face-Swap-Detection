import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def preprocess_data(dataset_path, output_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Project\src\shape_predictor_68_face_landmarks.dat")
    
    original_video_files = os.listdir(os.path.join(dataset_path, "original_videos"))
    manipulated_video_files = os.listdir(os.path.join(dataset_path, "manipulated_videos"))
    
    X = []  # Features (velocity vectors)
    y = []  # Labels (0 for genuine, 1 for manipulated)
    
    for video_file in original_video_files + manipulated_video_files:
        if "original" in video_file:
            label = 0  # Genuine
            video_path = os.path.join(dataset_path, "original_videos", video_file)
        else:
            label = 1  # Manipulated
            video_path = os.path.join(dataset_path, "manipulated_videos", video_file)
            
        cap = cv2.VideoCapture(video_path)
        
        prev_landmarks = None
        features = []
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to grayscale for facial landmark detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use the detector to find faces in the frame
            faces = detector(gray_frame)

            if len(faces) == 1:
                # Get the bounding box coordinates of the detected face
                x, y, width, height = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()

                # Extract the region of interest (ROI) containing the face for facial landmark tracking
                face_roi = gray_frame[y:y+height, x:x+width]

                # Predict facial landmarks for the ROI
                shape = predictor(face_roi, dlib.rectangle(0, 0, width, height))
                shape = [(shape.part(i).x + x, shape.part(i).y + y) for i in range(shape.num_parts)]

                # Calculate velocity vectors
                if prev_landmarks is not None:
                    velocity_vectors = np.array(shape) - np.array(prev_landmarks)
                    features.append(velocity_vectors.flatten())

                prev_landmarks = shape
                
        if features:
            X.append(features)
            y.append(label)
            
    # Save preprocessed data
    np.save(os.path.join(output_path, "X.npy"), X)
    np.save(os.path.join(output_path, "y.npy"), y)
    print("Preprocessed data saved.")
    
def preprocess_data_in_batches(video_files, dataset_path, batch_size, output_path):
    # Initialize variables to accumulate features and labels
    X_accumulator = []
    y_accumulator = []
    
    for video_file in video_files:
        if "original" in video_file:
            label = 0  # Genuine
            video_path = os.path.join(dataset_path, "original_videos", video_file)
        else:
            label = 1  # Manipulated
            video_path = os.path.join(dataset_path, "manipulated_videos", video_file)
        
        cap = cv2.VideoCapture(video_path)
        
        prev_landmarks = None
        features = []
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to grayscale for facial landmark detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use the detector to find faces in the frame
            faces = detector(gray_frame)

            if len(faces) == 1:
                # Get the bounding box coordinates of the detected face
                x, y, width, height = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()

                # Extract the region of interest (ROI) containing the face for facial landmark tracking
                face_roi = gray_frame[y:y+height, x:x+width]

                # Predict facial landmarks for the ROI
                shape = predictor(face_roi, dlib.rectangle(0, 0, width, height))
                shape = [(shape.part(i).x + x, shape.part(i).y + y) for i in range(shape.num_parts)]

                # Calculate velocity vectors
                if prev_landmarks is not None:
                    velocity_vectors = np.array(shape) - np.array(prev_landmarks)
                    features.append(velocity_vectors.flatten())

                prev_landmarks = shape
                
        if features:
            X_accumulator.extend(features)
            y_accumulator.extend([label] * len(features))
            
            # If accumulated batch size reaches the specified batch_size
            if len(X_accumulator) >= batch_size:
                # Save the accumulated batch
                np.save(os.path.join(output_path, f"X_batch_{len(X_accumulator)}.npy"), X_accumulator)
                np.save(os.path.join(output_path, f"y_batch_{len(y_accumulator)}.npy"), y_accumulator)
                print(f"Batch {len(X_accumulator)} saved.")
                
                # Reset the accumulators
                X_accumulator = []
                y_accumulator = []
    
    # Save any remaining data
    if X_accumulator:
        np.save(os.path.join(output_path, f"X_batch_{len(X_accumulator)}.npy"), X_accumulator)
        np.save(os.path.join(output_path, f"y_batch_{len(y_accumulator)}.npy"), y_accumulator)
        print(f"Batch {len(X_accumulator)} saved.")
    
if __name__ == "__main__":
    dataset_path = "Project\data"
    output_path = "Project"
    batch_size = 100  # Set the desired batch size
    
    original_video_files = os.listdir(os.path.join(dataset_path, "original_videos"))
    manipulated_video_files = os.listdir(os.path.join(dataset_path, "manipulated_videos"))
    all_video_files = original_video_files + manipulated_video_files
    
    preprocess_data(dataset_path, output_path)
    preprocess_data_in_batches(all_video_files, dataset_path, batch_size, output_path)