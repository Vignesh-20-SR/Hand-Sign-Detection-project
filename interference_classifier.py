import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Label mapping for all alphabets from 'a' to 'z'
labels_dict = {i: chr(97 + i) for i in range(26)}  # ASCII values for 'a' to 'z'

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Variables for tracking the predicted word and timer
predicted_word = ""
start_time = time.time()
TIME_LIMIT = 5  # Seconds to store letters for forming a word

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read from webcam.")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Prepare feature data for the model
            data_aux = []
            x_coords = []
            y_coords = []

            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)

            # Normalize coordinates relative to the bounding box
            min_x, min_y = min(x_coords), min(y_coords)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Ensure feature size matches the model's expectation
            if len(data_aux) == model.n_features_in_:
                # Predict the gesture
                prediction = model.predict([np.asarray(data_aux)])

                # Handle direct string predictions or index-based predictions
                if isinstance(prediction[0], str):
                    predicted_character = prediction[0]  # Direct string prediction
                else:
                    predicted_character = labels_dict[int(prediction[0])]  # Index-based prediction

                # Append the character to the word if within the time limit
                if time.time() - start_time <= TIME_LIMIT:
                    predicted_word += predicted_character
                else:
                    predicted_word = predicted_character
                    start_time = time.time()  # Reset timer

                # Draw bounding box and predicted label
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, 
                    predicted_character, 
                    (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, 
                    (0, 0, 255), 
                    3, 
                    cv2.LINE_AA
                )
            else:
                print(f"Feature size mismatch. Expected {model.n_features_in_}, got {len(data_aux)}.")

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == 13:  # Enter key to display the word formed so far
        print(f"Word formed so far: {predicted_word}")

# Release resources
cap.release()
cv2.destroyAllWindows()
