import cv2
from mtcnn.mtcnn import MTCNN
import dlib
import sys 

def calculate_velocity_vectors(curr_landmarks, prev_landmarks):
    velocity_vectors = []
    for (prev_x, prev_y), (curr_x, curr_y) in zip(prev_landmarks, curr_landmarks):
        velocity_vectors.append((curr_x - prev_x, curr_y - prev_y))
    return velocity_vectors

def detect_faces_and_track_landmarks():
    try:
        # Create a VideoCapture object to access the laptop camera
        cap = cv2.VideoCapture(0)

        # Create an MTCNN detector instance for face detection
        detector = MTCNN()

        # Load the dlib facial landmark detector
        predictor = dlib.shape_predictor("Project/src/shape_predictor_68_face_landmarks.dat")

        # Initialize variables to store previous facial landmarks and features
        prev_landmarks = []
        features = []

        while True:
            # Capture a frame from the camera feed
            ret, frame = cap.read()

            if not ret:
                break

            # Convert the frame to grayscale for facial landmark detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Use MTCNN to detect faces in the frame
            faces = detector.detect_faces(frame)

            for face in faces:
                # Get the bounding box coordinates of the detected face
                x, y, width, height = face['box']

                # Draw bounding box around the detected face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

                # Extract the region of interest (ROI) containing the face for facial landmark tracking
                face_roi = gray_frame[y:y+height, x:x+width]

                # Predict facial landmarks for the ROI
                shape = predictor(face_roi, dlib.rectangle(0, 0, width, height))
                shape = [(shape.part(i).x + x, shape.part(i).y + y) for i in range(shape.num_parts)]

                # Perform temporal analysis to track facial landmark movements
                if prev_landmarks:
                    # Calculate velocity vectors for this frame and store them as features
                    velocity_vectors = calculate_velocity_vectors(shape, prev_landmarks)
                    features.append(velocity_vectors)

                    # Draw temporal analysis lines
                    for (prev_x, prev_y), (curr_x, curr_y) in zip(prev_landmarks, shape):
                        cv2.line(frame, (prev_x, prev_y), (curr_x, curr_y), (0, 0, 255), 1)

                # Draw facial landmarks on the frame as circles
                for (x_, y_) in shape:
                    cv2.circle(frame, (x_, y_), 2, (0, 255, 0), -1)

                # Update previous landmarks with current landmarks for the next iteration
                prev_landmarks = shape

            # Display the frame with bounding boxes, facial landmarks, and temporal analysis lines
            cv2.imshow('Face Detection and Landmark Tracking', frame)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the VideoCapture and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    detect_faces_and_track_landmarks()
