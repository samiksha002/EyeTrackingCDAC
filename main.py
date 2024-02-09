import time
import pyautogui
import cv2
import tkinter as tk
from eye_tracker import EyeTracker
from calibration import calibrate
from calibration import CalibrationSVR
from screen import Screen
from PIL import Image, ImageTk
import tkinter.messagebox as messagebox
from instruction_screen import show_instruction_screen
#imprort pygame

calibration = CalibrationSVR()
RES_SCREEN = pyautogui.size()
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 1169

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


""" def play_beep():
    pygame.mixer.init()
    pygame.mixer.music.load("/Users/tanmayjain/Developer/cdac-2/gaze_2/beep.wav")
    pygame.mixer.music.play() """
    
def on_button_click(row, col):
    categories = [
        "Fruits", "Vegetables", "Food", "Stationary Items",
    ]
    print(f"Button clicked at row {row}, column {col} - Category: {categories[row * 2 + col]}")
    
def show_instruction_dialog() :
    dialog = tk.Toplevel()
    dialog.title("Instructions")
    dialog.geometry("500x300")

    instructions_label = tk.Label(dialog, text="Follow the instructions and click OK when ready.")
    instructions_label.pack(pady=20)

    ok_button = tk.Button(dialog, text="OK", command=dialog.destroy)
    ok_button.pack(pady=10)


def main():
    # Show instruction screen before main implementation
    show_instruction_screen()
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Set up the eye tracker
    eye_tracker = EyeTracker()
    
    screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.namedWindow("frame")

    screen.clean()
    calibration_completed = False

    while True:
        _, frame = camera.read()

        processed_frame = eye_tracker.process_frame(frame)
        img_h, img_w,img_c = frame.shape
        x_coordinate = int((RES_SCREEN[0] - img_w) // 2)
        y_coordinate = int((RES_SCREEN[1] - img_h) // 3)
        cv2.moveWindow("frame", x_coordinate, y_coordinate)# Process the current frame
        cv2.imshow('frame', processed_frame)


        # Handle user input
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key to exit
            break
        elif k == ord('c'):  # 'c' key to start calibration
            #play_beep()
            screen.mode = "calibration"
            screen.draw_center()

            calibrate(camera, screen, eye_tracker)


            calibration_completed = True


    """ if calibration_completed:
        root = tk.Tk()
        root.title("2x2 Grid")
        root.geometry("1000x750")
        button_width = 400
        button_height = 350

        for i in range(2):
            for j in range(2):
                image_path = f"image{i * 2 + j + 1}.png"
                img = Image.open(image_path)
                new_width = int(img.width * 0.25)
                new_height = int(img.height * 0.25)
                img = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                x_coordinate = j * (button_width + 10)
                y_coordinate = i * (button_height + 10)

                button_text = [
                    "Fruits", "Vegetables", "Food", "Stationary Items",
                ][i * 2 + j]

                button = tk.Button(root, image=photo, text=button_text, compound=tk.TOP, width=button_width,
                                   height=button_height, border=True, borderwidth=1.5,
                                   command=lambda row=i, col=j: on_button_click(row, col))
                button.image = photo
                button.grid(row=i, column=j, padx=40, pady=20)

        root.after(50, move_mouse_continuous, eye_tracker, root)  # Continuously move mouse based on gaze vector
        root.mainloop()

        camera.release()
        cv2.destroyAllWindows() """

if __name__ == '__main__':
    main()
