import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def preprocess_data(dataset_path, output_path, video_label):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Project\src\shape_predictor_68_face_landmarks.dat")
    
    X = []  # Features (velocity vectors)
    labels = []  # Labels (0 for genuine, 1 for manipulated)
    
    video_files = os.listdir(os.path.join(dataset_path, video_label + "_videos"))
    
    for video_file in video_files:
        video_path = os.path.join(dataset_path, video_label + "_videos", video_file)
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
            X.extend(features)
            labels.extend([video_label] * len(features))
            
    # Save preprocessed data
    np.save(os.path.join(output_path, f"X_{video_label}.npy"), X)
    np.save(os.path.join(output_path, f"y_{video_label}.npy"), labels)
    print(f"Preprocessed {video_label} data saved.")

# Main section
if __name__ == "__main__":
    dataset_path = "Project\data"
    output_path = "Project"
    
    labels = ["original", "manipulated"]
    
    for label in labels:
        preprocess_data(dataset_path, output_path, label)
