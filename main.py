import time

from ui import build_main_window # imports from ui.py
from camera import open_camera, close_camera, get_preview_image, read_frame
from detection import load_face_detector, has_face
from config import FRAME_INTERVAL_MS, ATTENTION_STREAK_SECONDS

ui = build_main_window() # assigns the function from ui to this 
window = ui["window"] # pulls the main Tkinter window out of the dictionary

status_text = ui["status_text"]
debug_text = ui["debug_text"]
preview_label = ui["preview_label"]
start_button = ui["start_button"]
stop_button = ui["stop_button"]

is_monitoring = False
camera = None
preview_job = None
face_detector = None
distraction_start_time = None

def update_preview():
    global preview_job, distraction_start_time

    if not is_monitoring or camera is None:
        preview_job = None
        return

    frame, frame_error = read_frame(camera)

    if frame_error is not None:
        status_text.set("Status: Error")
        debug_text.set(f"Debug: {frame_error}")
        preview_job = None
        return

    face_found, detect_error = has_face(frame, face_detector)

    if detect_error is not None:
        status_text.set("Status: Error")
        debug_text.set(f"Debug: {detect_error}")
        preview_job = None
        return

    preview_image, preview_error = get_preview_image(camera)

    if preview_error is not None:
        status_text.set("Status: Error")
        debug_text.set(f"Debug: {preview_error}")
        preview_job = None
        return

    preview_label.config(image=preview_image, text="")
    preview_label.image = preview_image

    if face_found:
        distraction_start_time = None
        status_text.set("Status: Monitoring")
        debug_text.set("Debug: Face detected")
    else:
        if distraction_start_time is None:
            distraction_start_time = time.time()

        missing_seconds = time.time() - distraction_start_time
        debug_text.set(f"Debug: No face detected for {missing_seconds:.1f}s")

        if missing_seconds >= ATTENTION_STREAK_SECONDS:
            status_text.set("Status: Distracted")
        else:
            status_text.set("Status: Monitoring")

    preview_job = window.after(FRAME_INTERVAL_MS, update_preview)

def start_monitoring():
    global is_monitoring, camera, face_detector, distraction_start_time

    if is_monitoring: # guard checks, to not spam the button
        return
    
    detector, detector_error = load_face_detector()

    if detector_error is not None:
        status_text.set("Status: Error")
        debug_text.set(f"Debug: {detector_error}")
        return
    
    camera, camera_error = open_camera()

    if camera_error is not None:
        status_text.set("Status: Error")
        debug_text.set(f"Debug: {camera_error}")
        camera = None
        return

    face_detector = detector
    distraction_start_time = None
    is_monitoring = True
    status_text.set("Status: Monitoring")
    debug_text.set("Debug: Camera and detector ready")
    update_preview()

def stop_monitoring():
    global is_monitoring, camera, preview_job, face_detector, distraction_start_time

    if not is_monitoring:
        return
    
    if preview_job is not None:
        window.after_cancel(preview_job)
        preview_job = None
    
    close_camera(camera)
    camera = None
    face_detector = None
    distraction_start_time = None
    is_monitoring = False

    preview_label.config(image="", text="Camera preview stopped")
    preview_label.image = None

    status_text.set("Status: Stopped")
    debug_text.set("Debug: Camera released")

start_button.config(command=start_monitoring) # places the function on the start button.
stop_button.config(command=stop_monitoring) # places the function on the stop button

window.mainloop() # keeps the window running
