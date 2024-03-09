import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                             )

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0, 153, 0), thickness=4, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255, 100, 100), thickness=4, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2)
                             )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand  = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, right_hand])


DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])

num_sequences = 40
sequence_length = 30

for action in actions:
    for sequence in range(num_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass



cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Looping through each action
    for action in actions:

        # Looping through each video
        for sequence in range(num_sequences):

            # Looping through the entire sequence length
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                # Performing detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Drawing landmarks
                draw_styled_landmarks(image, results)

                # Collecting frames by having delay between each sequence
                if frame_num == 0:
                    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (255, 255, 255), -1)  # Draw white rectangle
                    cv2.putText(image, 'Starting collection...', (120, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for "{}". Video number: {}'.format(action, sequence), (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (255, 255, 255), -1)  # Draw white rectangle
                    cv2.putText(image, 'Collecting frames for "{}". Video number: {}'.format(action, sequence), (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)


                # Exporting keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
