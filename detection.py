try:
    import cv2
except ImportError:
    cv2 = None

from config import (
    FACE_CASCADE_PATH,
    MIN_FACE_SIZE,
    FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
)

def load_face_detector():
    if cv2 is None:
        return None, "OpenCV is not installed."
    
    if FACE_CASCADE_PATH is None:
        return None, "Face cascade path is not available."
    
    detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    if detector.empty():
        return None, "Could not load the face detector."
    
    return detector, None

def detect_faces(frame, detector):
    if cv2 is None:
        return None, "OpenCV is not installed."
    
    if frame is None:
        return None, "Frame is missing."
    
    if detector is None:
        return None, "Face detector is not loaded."
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=MIN_FACE_SIZE
    )
    return faces, None

def has_face(frame, detector):
    faces, error = detect_faces(frame, detector)

    if error is not None:
        return False, error
    
    return len(faces) > 0, None