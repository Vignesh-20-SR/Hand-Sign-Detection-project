import os
import pickle
import mediapipe as mp
import cv2

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where data is stored
#DATA_DIR = '""C:\Users\Admin\Vignesh\ML\HandSignDetection\HandSignDetection\data""'
DATA_DIR = r'C:\Users\Admin\Vignesh\ML\HandSignDetection\HandSignDetection\data'
# Initialize lists for data and labels "C:\Users\Vignesh\Html\HandSignDetection\data"
data = []
labels = []

# Loop through each class directory in the data folder
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if it's not a directory
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing images for class: {dir_}")
    for img_file in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_file)

        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Convert to RGB for Mediapipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to extract hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_coords = []
                y_coords = []

                # Extract x and y coordinates of landmarks
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)

                # Normalize landmarks relative to the top-left corner
                min_x, min_y = min(x_coords), min(y_coords)
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

                # Append the processed data and label
                data.append(data_aux)
                labels.append(dir_)

# Save the processed data into a pickle file
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data successfully saved to {output_file}")
