from pathlib import Path #helps build file paths directory

try:
    import cv2 # cv2 talks to the cam and reads video frames
except ImportError: # catches if there is no import module used
    cv2 = None # if it catches, it then sets this cv2 to None

# try:
#     from PIL import Image, ImageTk # PIL (Python Imaging Library) is for images
# except ImportError: # catches if there is no import module used
#     image = None
#     ImageTk = None

BASE_DIR = Path(__file__).resolve().parent # Get the folder that this Python file is inside, and store it

CAMERA_INDEX = 0 # Default camera
MEME_VIDEO_PATH = BASE_DIR / "meme_video.mp4"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" if cv2 else None
# Use OpenCV’s built-in folder of detector files, and choose the frontal-face detector XML

ATTENTION_STREAK_SECOND = 4.0 # if no face detected = play meme video
COOLDOWN_SECONDS = 8.0 # cooldown after the meme video triggers
FRAME_INTERVAL_MS = 80 # update webcame every 80 miliseconds.
 
WINDOW_WIDTH = 900 # window size
WINDOW_HEIGHT = 720

PREVIEW_MAX_WIDTH = 720 # size of the camera inside the window
PREVIEW_MAX_HEIGHT = 405

VIDEO_POPUP_MAX_WIDTH = 960 # maximum size of the video 
VIDEO_POPUP_MAX_HEIGHT = 540 

MIN_FACE_SIZE = (80, 80) # ignore anything other than 80x80p when looking for a face
FACE_SCALE_FACTOR = 1.1 # searches for faces of various sizes
FACE_MIN_NEIGHBORS = 5 # controls how sure OPENCV should check a face low=more detections but false-pos, high = least 5 supporting detections before accepting

USE_MIRRORED_PREVIEW = True # webcam mirror view like, flips it
VIDEO_FALLBACK_DELAY_MS = 33 # backup speed for playing the meme video, hence 30 fps
