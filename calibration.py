import cv2
import pandas as pd
import numpy as np
import pyautogui
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer, StandardScaler
import matplotlib.pyplot as plt
import csv
import joblib

FRAME_WIDTH = 640
FRAME_HEIGHT =480
RES_SCREEN = pyautogui.size()

model = joblib.load('/Users/tanmayjain/Developer/cdac-2/gaze_2/randomForest.joblib')
# CSV file path
csv_file_path = "gauri_coordinates_data.csv"
# Writing to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["screen_coordinates", "left_x","left_y","right_x","right_y","predicted_x","predicted_y"])

class CalibrationSVR:
    def __init__(self):
        # Use SVR with a radial basis function (RBF) kernel for each dimension (x and y)
        self.reg_x = SVR(kernel='poly', C=150, degree=3, epsilon=0.1)
        self.reg_y = SVR(kernel='poly',C=150,degree=3,epsilon=0.1)

    def update(self, data):
        # Initialize arrays to store screen coordinates and pupil coordinates
        screen_coordinates = np.empty((0, 2))
        all_pupil_coords = np.empty((0, 4))  # Four independent variables (x and y coordinates of both pupils)

        # Collect data from known screen points
        for screen_point, (left_pupil, right_pupil) in data.items():
            if left_pupil is not None and right_pupil is not None:
                # Store screen coordinates
                screen_coordinates = np.concatenate((screen_coordinates, [screen_point]))

                # Store both left and right pupil coordinates
                all_pupil_coords = np.concatenate((all_pupil_coords, [left_pupil + right_pupil]))

        # Ensure both arrays have the same number of samples
        min_samples = min(screen_coordinates.shape[0], all_pupil_coords.shape[0])
        screen_coordinates = screen_coordinates[:min_samples]
        all_pupil_coords = all_pupil_coords[:min_samples]

        # Fit the calibration models for x and y coordinates separately
        self.reg_x.fit(all_pupil_coords, screen_coordinates[:, 0])  # 0 for x-coordinate
        self.reg_y.fit(all_pupil_coords, screen_coordinates[:, 1])  # 1 for y-coordinate

    def predict(self, left_pupil, right_pupil):
        # Use the trained models to predict screen coordinates for new left and right pupil coordinates
        x_coordinate = self.reg_x.predict([left_pupil + right_pupil])
        y_coordinate = self.reg_y.predict([left_pupil + right_pupil])
        new_pupil_point = ((left_pupil[0],left_pupil[1]),(right_pupil[0],right_pupil[1]))
        with open(csv_file_path, mode='a', newline='') as file_append:
            writer_append = csv.writer(file_append)
            writer_append.writerow(["", left_pupil[0],left_pupil[1],right_pupil[0],right_pupil[1],x_coordinate[0],y_coordinate[0]])
        return x_coordinate[0], y_coordinate[0]

    def plot_regression(self, all_p, all_coords):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(all_p[:, 0], all_p[:, 1], all_coords[:, 0], color='blue', label='Actual Coordinates (X)')
        ax.scatter(all_p[:, 0], all_p[:, 1], all_coords[:, 1], color='green', label='Actual Coordinates (Y)')

        predicted_x = self.reg_x.predict(all_p)
        predicted_y = self.reg_y.predict(all_p)

        ax.scatter(all_p[:, 0], all_p[:, 1], predicted_x, color='red', label='Predicted Coordinates (X)')
        ax.scatter(all_p[:, 0], all_p[:, 1], predicted_y, color='orange', label='Predicted Coordinates (Y)')

        ax.set_xlabel('Eye Coordinates X')
        ax.set_ylabel('Eye Coordinates Y')
        ax.set_zlabel('Screen Coordinates')

        plt.title('3D Regression Graph')
        plt.legend()
        plt.show()

""" def move_mouse(screen_coordinates):
    # Adjust the scaling factor based on your screen resolution
    scaling_factor = 1
    x, y = screen_coordinates

    # Disable PyAutoGUI fail-safe temporarily
    original_failsafe_setting = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False

    pyautogui.moveTo(x * scaling_factor, y * scaling_factor)

    # Restore the original fail-safe setting
    pyautogui.FAILSAFE = original_failsafe_setting """
    
def move_mouse(screen_x, screen_y):
    # Adjust the scaling factor based on your screen resolution
    scaling_factor = 1
    x = screen_x
    y = screen_y

    # Disable PyAutoGUI fail-safe temporarily
    original_failsafe_setting = pyautogui.FAILSAFE
    pyautogui.FAILSAFE = False



    pyautogui.moveTo(x * scaling_factor, y * scaling_factor)

    # Restore the original fail-safe setting
    pyautogui.FAILSAFE = original_failsafe_setting

def calibrate(camera, screen, eye_tracker):
    N_REQ_COORDINATES = 100
    N_SKIP_COORDINATES = 25

    screen.clean()

    calibration = CalibrationSVR()
    calibration_points = calculate_points(screen)

    coordinates = []

    completed = False
    enough = 0
    skip = 0
    point = calibration_points.pop(0)
    screen.draw(point)
    screen.show()
    while point:
        the_list = calculate_points(screen)
        key = the_list.index(point)
        screen.draw(point)
        screen.show()

        _, frame = camera.read()

        processed_frame = eye_tracker.process_frame(frame)
        right_pupil_coordinates = eye_tracker.gaze_calculator.right_pupil
        left_pupil_coordinates=eye_tracker.gaze_calculator.left_pupil
    

        cv2.namedWindow("frame")
        dec_frame = processed_frame
        dec_frame = cv2.resize(dec_frame, (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))
        cv2.moveWindow("frame", 0, 0)
        cv2.imshow('frame', dec_frame)
        
        coordinates_tuple = (left_pupil_coordinates,right_pupil_coordinates)
        with open(csv_file_path, mode='a', newline='') as file_append:
            writer_append = csv.writer(file_append)
            writer_append.writerow([point, left_pupil_coordinates[0],left_pupil_coordinates[1],right_pupil_coordinates[0],right_pupil_coordinates[1],"",""])
        print("COORDINATES: {}\tPOINT: {}".format(coordinates_tuple, point))

        if coordinates_tuple and skip < N_SKIP_COORDINATES:
            skip += 1
            continue

        if coordinates_tuple:
            coordinates.append((point, coordinates_tuple))
            enough += 1

        progress = len(coordinates) / N_REQ_COORDINATES
        print("progress in calibration",progress)
        screen.draw(point, progress=progress,point_index = key)#)
        screen.show()

        # next point condition
        if enough >= N_REQ_COORDINATES and len(calibration_points) > 0:
            point = calibration_points.pop(0)
            skip = 0
            enough = 0
            screen.draw(point)
            screen.show()

        # end calibration condition
        if enough >= N_REQ_COORDINATES and len(calibration_points) == 0:
            screen.clean()
            completed = True
            break

        k = cv2.waitKey(1) & 0xff

        if k == 1048603 or k == 27:  # esc to terminate calibration
            screen.mode = "normal"
            screen.clean()
            screen.show()
            break

    if completed:
        calibration.update(dict(coordinates))
        eye_tracker.calibration = calibration

        screen.mode = "calibrated"
        screen.show()

        # Evaluate accuracy metrics
        evaluate_metrics(calibration, coordinates)

        while True:
            _, frame = camera.read()

            processed_frame = eye_tracker.process_frame(frame)
            right_pupil_coordinates = eye_tracker.gaze_calculator.right_pupil
            left_pupil_coordinates=eye_tracker.gaze_calculator.left_pupil

            """ cv2.namedWindow("frame")
            dec_frame = processed_frame
            dec_frame = cv2.resize(dec_frame, (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))
            cv2.moveWindow("frame", 0, 0)  """
            #cv2.imshow('frame', dec_frame)

            print(f"Right Pupil Coordinates: {right_pupil_coordinates}")
            print(f"Left Pupil Coordinates: {left_pupil_coordinates}")

            if right_pupil_coordinates[0]!=0 and left_pupil_coordinates[0]!=0:
                # Prepare the real-time data for prediction
                real_time_data = pd.DataFrame({
                'left_x': [left_pupil_coordinates[0]],
                'left_y': [left_pupil_coordinates[1]],
                'right_x': [right_pupil_coordinates[0]],
                'right_y': [right_pupil_coordinates[1]]
                })
                print(real_time_data)
                scaler = StandardScaler()
                power_transformer = PowerTransformer()
                # Power Transformation
                power_transformer = PowerTransformer()
                real_time_data[['left_x', 'left_y', 'right_x', 'right_y']] = power_transformer.fit_transform(real_time_data[['left_x', 'left_y', 'right_x', 'right_y']])
                # Standardization
                scaler = StandardScaler()
                real_time_data[['left_x', 'left_y', 'right_x', 'right_y']] = scaler.fit_transform(real_time_data[['left_x', 'left_y', 'right_x', 'right_y']])
                predicted_coordinates = model.predict(real_time_data)
                screen_x = predicted_coordinates[0, 0]
                screen_y = predicted_coordinates[0, 1]
                screen_coordinates = calibration.predict(left_pupil_coordinates,right_pupil_coordinates)
                print(f"Continuous Coordinates of Screen OLD: {screen_coordinates}")
                print(f"Continuous Coordinates of Screen NEW: {screen_x,screen_y}")
                move_mouse(screen_x, screen_y)

            k = cv2.waitKey(1) & 0xff

            if k == 1048603 or k == 27:  # esc to exit
                break

def calculate_points(screen):
    points = []

    # center
    p = (int(0.5 * screen.width), int(0.5 * screen.height))
    points.append(p)

    # top left
    p = (int(0.05 * screen.width), int(0.05 * screen.height))
    points.append(p)

    # top
    p = (int(0.5 * screen.width), int(0.05 * screen.height))
    points.append(p)

    # top right
    p = (int(0.95 * screen.width), int(0.05 * screen.height))
    points.append(p)
    
    #2nd quad
    p=(int(0.25 * screen.width),int(0.25 * screen.height))
    points.append(p)
    
    #1st quad
    p = (int(0.75 * screen.width), int(0.25 * screen.height))
    points.append(p)

    # left
    p = (int(0.05 * screen.width), int(0.5 * screen.height))
    points.append(p)

    # right
    p = (int(0.95 * screen.width), int(0.5 * screen.height))
    points.append(p)
    
    #3rd quad
    p = (int(0.25 * screen.width), int(0.75 * screen.height))
    points.append(p)

    #4th quad
    p = (int(0.75 * screen.width), int(0.75 * screen.height))
    points.append(p)

    # bottom left
    p = (int(0.05 * screen.width), int(0.95 * screen.height))
    points.append(p)

    # bottom
    p = (int(0.5 * screen.width), int(0.95 * screen.height))
    points.append(p)

    # bottom right
    p = (int(0.95 * screen.width), int(0.95 * screen.height))
    points.append(p)



    return points

def evaluate_metrics(calibration, coordinates):
    true_coordinates = np.array([point for point, _ in coordinates])
    predicted_coordinates = np.array([calibration.predict(left_pupil,right_pupil) for _, (left_pupil,right_pupil) in coordinates])

    mse_x = mean_squared_error(true_coordinates[:, 0], predicted_coordinates[:, 0])
    rmse_x = np.sqrt(mse_x)

    mse_y = mean_squared_error(true_coordinates[:, 1], predicted_coordinates[:, 1])
    rmse_y = np.sqrt(mse_y)

    print(f"Mean Squared Error (X): {mse_x}")
    print(f"Root Mean Squared Error (X): {rmse_x}")

    print(f"Mean Squared Error (Y): {mse_y}")
    print(f"Root Mean Squared Error (Y): {rmse_y}")

 
