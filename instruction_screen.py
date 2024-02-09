import tkinter as tk

def show_instruction_screen():
    root = tk.Toplevel()
    root.title("Calibration Instructions")

    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate the position to center the window
    x_coordinate = (screen_width - 800) // 2
    y_coordinate = (screen_height - 600) // 2

    # Set the geometry to center the window
    root.geometry(f"800x600+{x_coordinate}+{y_coordinate}")

    # Add instructions
    instrctions="\nFollow the instructions for calibration:\n\n\n\n\n\n\n\n1) Place your head inside the circular frame. DO NOT MOVE! \n\n 2) Keep your head as still as possible calibrate just by moving your eyes. \n\n 3) Press 'c' to start calibration."
    
    instructions_label = tk.Label(root,font=("Arial", 18), text = instrctions,fg = "white")
    instructions_label.pack(pady=20)

    # Add more instructions or elements as needed

     # Style the "OK" button
    ok_button = tk.Button(root, text="OK", command=root.destroy,width=10,height=2, font=("Arial", 14), bg="#0000ff", fg="black", padx=5, pady=5)
    ok_button.pack(pady=20)

    root.mainloop()
