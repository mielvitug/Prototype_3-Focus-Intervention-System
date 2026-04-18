import tkinter as tk #GUI dependencies from python

from config import WINDOW_WIDTH, WINDOW_HEIGHT # imports from config.py

def build_main_window():
    window = tk.Tk() # starts the window
    window.title("Focus Intervention System")
    window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    window.configure(bg="#111111")

    status_text = tk.StingVar(value="Status: Not started")
    debug_text = tk.StingVar(value="Debug: Waiting to start")

    title_label = tk.Label(
        window,
        text="Focus Intervention System",
        font=("Arial", 24, "bold"),
        bg="#111111",
        fg="white" # foreground color
    )
    title_label.pack(pady=(20, 10)) # inside the window for the title text

    status_label = tk.Label(
        window,
        textvariable=status_text, # Make this label display whatever text is stored inside 
        font=("Arial", 14),
        bg="#111111",
        fg="#00ffcc"
    )
    status_label.pack(pady=(0,6))

    debug_label = tk.Label(
        window,
        textvariable=debug_text, 
        font=("Arial", 14),
        width=60,
        height=20,
        bg="black",
        fg="white"
    )
    debug_label.pack(pady=(0,12))

    preview_label = tk.Label(
        window, 
        text="Camera preview will appear here",
        font=("Arial", 14),
        width=60,
        height=20,
        bg="black",
        fg="white"
    )
    preview_label.pack(pady=10)

    button_frame = tk.Frame(window, bg="#111111")
    button_frame.pack(pady=12)

    start_button = tk.button(
        button_frame,
        text="Start Monitoring",
        font=("Arial", 12, "bold"),
        width=18
    )
    start_button.pack(slide="left", padx=10)

    stop_button = tk.Button(
        button_frame,
        text="Stop Monitory",
        font=("Arial", 12, "bold"),
        width=18
    )
    stop_button.pack(side="left", padx=10)

    return { # Return call later
        "window": window,
        "status_text": status_text,
        "debug_text": debug_text,
        "preview_label": preview_label,
        "start_button": start_button,
        "stop_button": stop_button,
    }



