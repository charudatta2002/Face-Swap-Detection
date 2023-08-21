import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.applications.xception import Xception

# Load Xception model without top layer
weights_path = 'Project/src/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = Xception(weights=weights_path, include_top=False)

#base_model = Xception(weights='Project\xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

# Add classification layer
x = base_model.output
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Load face detector
detector = MTCNN()

# Access webcam
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    
    if ret:
        # Detect faces
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            
            # Crop and resize to 299x299
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (299, 299))
            
            # Normalize input
            face_img = (face_img - 127.5) / 127.5
            
            # Reshape
            face_img = np.expand_dims(face_img, axis=0)
            
            # Get deepfake prediction
            predictions = model.predict(face_img)
        for prediction in predictions:
            prediction_probability = prediction[0]    
            # Draw label on frame
            for prob in prediction_probability:
             if prob > 0.5: 
                text = 'Deepfake'
                color = (0, 0, 255)
             else:
                text = 'Real'
                color = (0, 255, 0)
                
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) == 27: 
            break
        
cap.release()
cv2.destroyAllWindows()