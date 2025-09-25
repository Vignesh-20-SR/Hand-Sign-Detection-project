import os
import cv2

DATA_DIR = './data_3'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100
# Use the appropriate camera device index (0 is the default, but this might vary)
cap = cv2.VideoCapture(0)  # Default camera index, adjust if necessary

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            continue

        cv2.putText(frame, 'Ready? Press "Q" or "ESC" to exit! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            done = True
        elif key == 27:  # ESC key
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == 27:  # ESC key
            break
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()