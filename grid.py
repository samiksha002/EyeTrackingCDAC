""" import cv2
import mediapipe as mp
import pyautogui
import numpy as npq

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

pyautogui.FAILSAFE = False
# Initialize face mesh
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                           refine_landmarks=True) as face_mesh:
    sensitivity_scale = 2
    prev_right_pupil_x, prev_right_pupil_y = 100, 100

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with face mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Assuming you have only one face in the frame
            face_landmarks = results.multi_face_landmarks[0]

            # Extract coordinates of the eyes
            right_eye_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark[474:478]]
            left_eye_coords = [(landmark.x, landmark.y) for landmark in face_landmarks.landmark[469:473]]

            # Calculate coordinates of the pupils
            right_pupil_x = int((right_eye_coords[0][0] + right_eye_coords[2][0]) / 2 * frame.shape[1])
            right_pupil_y = int((right_eye_coords[1][1] + right_eye_coords[3][1]) / 2 * frame.shape[0])
            left_pupil_x = int((left_eye_coords[0][0] + left_eye_coords[2][0]) / 2 * frame.shape[1])
            left_pupil_y = int((left_eye_coords[1][1] + left_eye_coords[3][1]) / 2 * frame.shape[0])

            # Calculate the change in pupil position
            delta_x = right_pupil_x - prev_right_pupil_x
            delta_y = right_pupil_y - prev_right_pupil_y

            pyautogui.move(-delta_x * 50, delta_y * 50)

            cv2.drawMarker(frame, (right_pupil_x, right_pupil_y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=7,
                           thickness=1)
            cv2.drawMarker(frame, (left_pupil_x, left_pupil_y), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=7,
                           thickness=1)

            # Update the previous pupil position for the next iteration
            prev_right_pupil_x, prev_right_pupil_y = right_pupil_x, right_pupil_y

            # Display landmark indices
            for i, landmark_index in enumerate([469, 470, 471, 472, 474, 475, 476, 477]):
                landmark = face_landmarks.landmark[landmark_index]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Display the frame
        #cv2.imshow("Webcam Feed", frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
 """