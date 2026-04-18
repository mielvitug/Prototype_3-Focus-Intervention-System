try:
    import cv2
except ImportError:
    cv2 = None

try: 
    from PIL import Image, ImageTK
except ImportError: 
    Image = None
    ImageTK = None

import tkinter as tk
from config import MEME_VIDEO_PATH, VIDEO_POPUP_MAX_WIDTH, VIDEO_POPUP_MAX_HEIGHT, VIDEO_FALLBACK_DELAY_MS

class VideoPopup:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.popup = None
        self.video_label = None
        self.video_capture = None
        self.video_job = None

    def open(self):
        if self.popup is not None:
            return
        
        self.popup = tk.Toplevel(self.parent_window)
        self.popup.title("Focus Check")
        self.popup.configure(bg="black")
        self.popup.attributes("-topmost", True) # Keep window above
        self.popup.protocol("WM_DELETE_WINDOW, self.close") # user clicks x button

        self.video_label = tk.Label(self.popup, bg="black")
        self.video_label.pack(padx=12, pady=12)
    def close(self):
        if self.video_job is not None and self.popup is not None:
            self.popup.after_cancel(self.video_job)
            self.video_job = None

        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if self.popup is not None:
            self.popup.destroy()
            self.popup = None
            self.video_label = None
