try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTK
except ImportError:
    Image = None
    ImageTk = None

from config import (
    CAMERA_INDEX,
    PREVIEW_MAX_WIDTH,
    PREVIEW_MAX_HEIGHT,
    USE_MIRRORED_PREVIEW,
)

from config import CAMERA_INDEX

def open_camera(): # open the camera function
    if cv2 is None:
        return None, "OpenCV is not installed."
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        return None, f"Could not open camera {CAMERA_INDEX}."
    return camera, None

def close_camera(camera):
    if camera is not None:
        camera.release() # close camera function

def read_frame(camera):
    if camera is None:
        return None, "Camera is not open." # stops here
    success, frame = camera.read()
    if not success:
        return None, "Could not read a frame from the camera."
    if USE_MIRRORED_PREVIEW: # checks config setting
        frame = cv2.flip(frame,1)
    return frame, None

def convert_frame_to_tk(frame, max_width, max_height):
    if cv2 is None:
        return None, "OpenCV is not installed."
    
    if Image is None or ImageTK is None:
        return None, "Pillow is not installed."
    
    frame_rgb = cv2.cv2Color(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    image.thumbnail((max_width, max_height))

    tk_image = ImageTk.PhotoImage(image)
    return tk_image, None

def get_preview_image(camera):
    frame, error = read_frame(camera)

    if error is not None:
        return None, error
    
    return convert_frame_to_tk(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)