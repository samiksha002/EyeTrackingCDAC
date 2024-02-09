import numpy as np
import cv2
import pyautogui

RES_SCREEN = pyautogui.size() # RES_SCREEN[0] -> width
                              # RES_SCREEN[1] -> heigth

class Screen:

    def __init__(self, width=1800, height=1169):
        self.width = width
        self.height = height
        self.pointer = (0,0)
        self.mode = "normal"
        self.screen = np.ones((self.height, self.width, 3))
        self.clean()

    def refresh(self):
        self.clean()
        self.draw_pointer()
        self.show()

    def update(self, gaze):
        self.pointer = gaze

    def clean(self):
        self.screen = np.ones((self.height, self.width, 3))
        self.print_instructions()

    def draw(self, point, progress=0,point_index=0):
        x, y = point
        
        if progress-point_index == 1.0:
            cv2.circle(self.screen, (x, y), 9, (0, 255, 0), -1)
        else:
            cv2.circle(self.screen, (x, y), 9, (0, 0, 0), -1)

        if progress > 0:
            # Ellipse parameters
            radius = 11
            axes = (radius, radius)
            angle = 0
            start_angle = 0
            end_angle = 360 * (progress-point_index)
            cv2.ellipse(self.screen, (x, y), axes, angle, start_angle, end_angle, (0, 255, 0), 2)


    def draw_center(self):
        x, y = (int(0.5 * self.width), int(0.5 * self.height))
        cv2.circle(self.screen, (x, y), 9, (0, 0, 0), -1)

    def draw_pointer(self):
        x, y = self.pointer
        cv2.circle(self.screen, (x, y), 9, (0, 255, 0), -1)

    def print_instructions(self):
        x, y0, dy = int(0.03 * self.width), int(0.8 * self.height), 35

        if self.mode == "normal":
            instructions = "Press:\nESC to quit\nc to start calibration"
        if self.mode == "calibration":
#            instructions = "Press:\nESC to terminate\nn to next calibration step"
            instructions = "Press:\nESC to terminate calibration"

        for i, line in enumerate(instructions.split('\n')):
            y = y0 + i*dy
            cv2.putText(img=self.screen, text=line, org=(x, y),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,0), thickness=2)


    def print_message(self, msg):

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 2
        th = 3

        for i, line in enumerate(msg.split('\n')):
            textsize = cv2.getTextSize(line, font, fs, th)[0]
            x = (self.width - textsize[0]) // 2
            y0, dy = (self.height + textsize[1]) // 2, textsize[1] + 30

            y = y0 + i*dy
            cv2.putText(img=self.screen, text=line, org=(x, y),fontFace=font, fontScale=fs, color=(0,0,0), thickness=th)
            
    def show(self):
            cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
        
            # Calculate the x and y coordinates to center the window
            x_coordinate = int((RES_SCREEN[0] - self.width) // 2)
            y_coordinate = int((RES_SCREEN[1] - self.height) // 2)
        
            # Set the window position
            cv2.moveWindow("screen", x_coordinate+110, y_coordinate)
        
            # Set the window size to cover the full screen
            cv2.resizeWindow("screen", RES_SCREEN[0], RES_SCREEN[1])
        
            cv2.imshow("screen", self.screen)
    
"""def show(self):
        cv2.namedWindow("screen", cv2.WINDOW_NORMAL)  # Create the window with the WINDOW_NORMAL flag

        # Calculate the x and y coordinates to center the window
        x_coordinate = int((RES_SCREEN[0] - self.width) // 2)
        y_coordinate = int((RES_SCREEN[1] - self.height) // 2)

        cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        #cv2.setWindowProperty("screen", cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        cv2.imshow("screen", self.screen) """ 
        

        


