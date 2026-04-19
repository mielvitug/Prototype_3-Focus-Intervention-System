try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

from config import (
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_INDEX,
    CAMERA_TARGET_FPS,
    PREVIEW_MAX_HEIGHT,
    PREVIEW_MAX_WIDTH,
    USE_MIRRORED_PREVIEW,
)


def open_camera():
    if cv2 is None:
        return None, "OpenCV is not installed."

    if hasattr(cv2, "CAP_DSHOW"):
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if hasattr(cv2, "VideoWriter_fourcc"):
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, CAMERA_TARGET_FPS)

    if not camera.isOpened():
        return None, f"Could not open camera {CAMERA_INDEX}."

    return camera, None


def close_camera(camera):
    if camera is not None:
        camera.release()


def read_frame(camera):
    if camera is None:
        return None, "Camera is not open."

    success, frame = camera.read()

    if not success:
        return None, "Could not read a frame from the camera."

    if USE_MIRRORED_PREVIEW:
        frame = cv2.flip(frame, 1)

    return frame, None


def convert_frame_to_tk(frame, max_width, max_height):
    if cv2 is None:
        return None, "OpenCV is not installed."

    if Image is None or ImageTk is None:
        return None, "Pillow is not installed."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image.thumbnail((max_width, max_height))

    tk_image = ImageTk.PhotoImage(image)
    return tk_image, None


def get_preview_image_from_frame(frame):
    return convert_frame_to_tk(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)


def get_preview_image(camera):
    frame, error = read_frame(camera)

    if error is not None:
        return None, error

    return get_preview_image_from_frame(frame)
