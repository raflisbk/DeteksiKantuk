import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
import imutils
import time
from playsound import playsound

# Load pre-trained dlib face landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load pre-trained LSTM model
model = load_model("lstm.h5")

# Function to calculate EAR, MAR, MOE
def calculate_features(landmarks):
    # Calculate Eye Aspect Ratio (EAR)
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    ear = (ear_left + ear_right) / 2.0

    # Calculate Mouth Aspect Ratio (MAR)
    mouth = landmarks[48:68]
    mar = mouth_aspect_ratio(mouth)

    # Calculate Mouth Over Eye (MOE)
    moe = mar / ear

    return ear, mar, moe

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B) / (2.0 * C)
    return mar

# Function to predict drowsiness
def predict_drowsiness(features):
    features = np.array(features)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    return prediction[0][0]

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize variables for tracking drowsiness streak
start_time = None
drowsiness_streak = 0
features_history = []
reset_time = time.time()

# Sound alarm function
def sound_alarm():
    # Replace "alarm.mp3" with the path to your alarm sound file
    playsound("alarm.wav")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        ear, mar, moe = calculate_features(landmarks)
        features_history.append([ear, mar, moe])
        
        if len(features_history) == 5:
            prediction = predict_drowsiness(features_history)
            features_history.pop(0)
            
            if prediction > 0.5:
                if start_time is None:
                    start_time = time.time()
                else:
                    if time.time() - start_time >= 4:
                        sound_alarm()
                        cv2.putText(frame, "Awas Kantuk", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        start_time = None
            else:
                start_time = None
                cv2.putText(frame, "Siaga", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "MOE: {:.2f}".format(moe), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Drowsiness Detection', frame)
    
    # Reset features_history after 10 seconds
    if time.time() - reset_time >= 10:
        features_history = []
        reset_time = time.time()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
