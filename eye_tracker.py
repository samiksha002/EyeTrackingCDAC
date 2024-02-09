import cv2
import math
import mediapipe as mp
import numpy as np
import time
import pyautogui

RES_SCREEN = pyautogui.size()
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 1169

class GazeCalculator:
    def __init__(self, left_pupil, right_pupil, left_iris_radius, right_iris_radius,pitch,yaw,roll):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.left_iris_radius = left_iris_radius
        self.right_iris_radius = right_iris_radius
        self.pitch=pitch
        self.yaw=yaw
        self.roll=roll


    def pupil_coordinates_left(self):
        if self.left_pupil is not None:
            return self.left_pupil[0], self.left_pupil[1]
        else:
            # Handle the case where left_pupil is None
            return 0.0, 0.0
    def pupil_coordinates_right(self):
        if self.left_pupil is not None:
            return self.right_pupil[0],self.right_pupil[1]
        else:
            # Handle the case where left_pupil is None
            return 0.0, 0.0



    def calculate_gaze_vector(self):
        if self.left_pupil and self.right_pupil:
            # Calculate the gaze vector using the left and right pupil coordinates
            gaze_vector = (
                self.right_pupil[0] - self.left_pupil[0],
                self.right_pupil[1] - self.left_pupil[1],
            )
            return gaze_vector
        else:
            return None

class EyeTracker:
    def __init__(self):
        self.gaze_calculator = GazeCalculator(None,None,None,None,None,None,None)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.left_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_iris = [474, 475, 476, 477]
        self.left_iris = [469, 470, 471, 472]
        self.l_h_left = [33]
        self.l_h_right = [133]
        self.r_h_left = [362]
        self.r_h_right = [263]



        # Added: Pupil and Iris attributes
        self.left_pupil = None
        self.right_pupil = None
        self.left_iris_radius = None
        self.right_iris_radius = None
        # Added: Variables for Purkinje detection
        self.left_purkinje = None
        self.right_purkinje = None
        self.left_eye_bb = None  # Define the left eye bounding box
        self.right_eye_bb = None  # Define the right eye bounding box
        self.frame_gray = None  # Store the grayscale frame
        self.normalized_right_pupil_x=None
        self.normalized_right_pupil_y=None
        self.normalized_left_pupil_x=None
        self.normalized_left_pupil_y=None

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1.ravel()
        x2, y2 = point2.ravel()
        distance = math.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        return distance

    def iris_position(self, iris_center, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)
        ratio = center_to_right_dist / total_distance
        iris_position = ""
        if ratio <= 0.42:
            iris_position = "right"
        elif 0.42 < ratio <= 0.57:
            iris_position = "center"
        else:
            iris_position = "left"
        return iris_position, ratio

    def detect_pupil(self, eye_region):
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.GaussianBlur(gray_eye, (5, 5), 0)
        gray_eye = cv2.equalizeHist(gray_eye)

        _, threshold = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                return True, (cx, cy)

        return False, None


    def update(self, frame):
        self.frame = frame
        self._analyze()
        
    def pupil_left_coordinates(self,normalized_left_pupil_x,normalized_left_pupil_y):
        self.normalized_left_pupil_x=normalized_left_pupil_x
        self.normalized_left_pupil_y=normalized_left_pupil_y
        return self.normalized_left_pupil_x,self.normalized_left_pupil_y

    def pupil_right_coordinates(self,right_pupil_coordinates_x,right_pupil_coordinates_y):
        self.normalized_right_pupil_x=right_pupil_coordinates_x
        self.normalized_right_pupil_y=right_pupil_coordinates_y
        return self.normalized_right_pupil_x,self.normalized_right_pupil_y
    
    """ def is_point_inside_ellipse(self, point, center, axes, angle):
        x, y = point
        cx, cy = center
        a, b = axes

        # Translate the point to the ellipse's local coordinate system
        xp = np.cos(np.radians(angle)) * (x - cx) - np.sin(np.radians(angle)) * (y - cy)
        yp = np.sin(np.radians(angle)) * (x - cx) + np.cos(np.radians(angle)) * (y - cy)

        # Check if the point is inside the rotated ellipse equation
        return (xp / a) ** 2 + (yp / b) ** 2 <= 1 """
        
    def is_point_inside_ellipse(self, nose, chin, forehead, center, axes, angle):
        # Convert points to tuples if they are not
        nose = tuple(nose) if not isinstance(nose, tuple) else nose
        chin = tuple(chin) if not isinstance(chin, tuple) else chin
        forehead = tuple(forehead) if not isinstance(forehead, tuple) else forehead

        # Check if any of the points are inside the rotated ellipse
        is_nose_inside = self._is_single_point_inside_ellipse(nose, center, axes, angle)
        is_chin_inside = self._is_single_point_inside_ellipse(chin, center, axes, angle)
        is_forehead_inside = self._is_single_point_inside_ellipse(forehead, center, axes, angle)

        # Return True if any of the points are inside the ellipse
        print("nose",is_nose_inside)
        print("chin",is_chin_inside)
        print("forehead",is_forehead_inside)
        return is_nose_inside and is_chin_inside and is_forehead_inside

    def _is_single_point_inside_ellipse(self, point, center, axes, angle):
        x, y = point
        cx, cy = center
        a, b = axes

        # Translate the point to the ellipse's local coordinate system
        xp = np.cos(np.radians(angle)) * (x - cx) - np.sin(np.radians(angle)) * (y - cy)
        yp = np.sin(np.radians(angle)) * (x - cx) + np.cos(np.radians(angle)) * (y - cy)

        # Check if the point is inside the rotated ellipse equation
        return (xp / a) ** 2 + (yp / b) ** 2 <= 1


    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w,img_c = frame.shape
        rgb_frame.flags.writeable = False
        start = time.time()

        with self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_frame)


        rgb_frame.flags.writeable = True

        #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_2d = []
        face_3d = []
        left_pupil_x = 0
        left_pupil_y = 0
        right_pupil_x = 0
        right_pupil_y = 0
        pitch = 0
        yaw = 0
        roll = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 152 or idx == 10:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        if idx == 152:
                            chin_2d = (lm.x * img_w, lm.y * img_h)
                        if idx == 10:
                            forehead_2d = (lm.x * img_w, lm.y * img_h)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append(([x, y, lm.z]))

                # Get 2d Coord
                face_2d = np.array(face_2d, dtype=np.float64)

                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # getting rotational of face
                rmat, jac = cv2.Rodrigues(rotation_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # here based on axis rot angle is calculated
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                                 distortion_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                pitch= str(np.round(x,2))
                yaw= str(np.round(y,2))
                roll=str(np.round(z,2))

                cv2.line(frame, p1, p2, (255, 255, 255), 3)

                #cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                
                instructions = "Place your head inside the frame and stay still dont MOVE much"
                instructions_2 =  "PRESS 'c' to start calibration"
                cv2.putText(frame, instructions, (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                cv2.putText(frame, instructions_2, (183, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
                
                # Get nose 2D coordinates
                nose_2d = (int(nose_2d[0]), int(nose_2d[1]))
                chin_2d = (int(chin_2d[0]), int(chin_2d[1]))
                forehead_2d = (int(forehead_2d[0]), int(forehead_2d[1]))
                
                # Draw oval in the middle of the frame
                oval_center = (img_w // 2, img_h // 2)
                oval_axes = (int(img_w // 4.45), int(img_h // 4.45))  # You can adjust the size of the oval by changing the axes
                
                # Check if the nose is inside the ellipse
                is_nose_inside_ellipse = self.is_point_inside_ellipse(nose_2d,chin_2d,forehead_2d, oval_center, oval_axes, 90)

                # Draw rotated oval in the middle of the frame with different color based on the condition
                oval_color = (0, 255, 0)  # Default color (green)
                if not is_nose_inside_ellipse:
                    oval_color = (0, 0, 255)  # Change color to red
                
                
                cv2.ellipse(frame, oval_center, oval_axes, 90, 0, 360, oval_color, 2)

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            # print("FPS: ", fps)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

            # Draw 'X' marker on the pupils
            cv2.drawMarker(frame, (right_pupil_x, right_pupil_y), (0, 255, 0), markerType=cv2.MARKER_STAR,
                           markerSize=7,
                           thickness=1)
            cv2.drawMarker(frame, (left_pupil_x, left_pupil_y), (0, 255, 0), markerType=cv2.MARKER_STAR,
                           markerSize=7,
                           thickness=1)

        leftpupil = (left_pupil_x, left_pupil_y)
        rightpupil = (right_pupil_x, right_pupil_y)

        self.gaze_calculator.left_pupil = leftpupil
        self.gaze_calculator.right_pupil = rightpupil



        self.gaze_calculator.left_pupil = leftpupil
        self.gaze_calculator.right_pupil = rightpupil
        self.gaze_calculator.pitch=pitch
        self.gaze_calculator.yaw=yaw
        self.gaze_calculator.roll=roll

        return frame


if __name__ == "__main__":
    eye_tracker = EyeTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = eye_tracker.process_frame(frame)

        cv2.imshow("img", processed_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()