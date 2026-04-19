import threading
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

from camera import close_camera, get_preview_image_from_frame, open_camera, read_frame
from config import (
    ANALYSIS_FRAME_HEIGHT,
    ANALYSIS_FRAME_WIDTH,
    ANALYSIS_INTERVAL_MS,
    ATTENTION_STREAK_SECONDS,
    COOLDOWN_SECONDS,
    FACE_DETECTION_GRACE_SECONDS,
    FACE_MISS_CONFIRM_FRAMES,
    FRAME_INTERVAL_MS,
    INTERVENTION_FRAME_INTERVAL_MS,
    PHONE_DETECT_CONFIRM_FRAMES,
    PHONE_USAGE_MODEL_PATH,
    PHONE_USAGE_MIN_FACE_STABLE_SECONDS,
    PHONE_USAGE_STRONG_MARGIN,
    PHONE_SCORE_DECAY_PER_SECOND,
    PHONE_SCORE_GAIN_PER_SECOND,
    PHONE_SCORE_TRIGGER,
    PHONE_USAGE_STREAK_SECONDS,
    PHONE_USAGE_THRESHOLD,
    VIDEO_EXTRA_PLAY_SECONDS_ON_FACE_RETURN,
    VIDEO_STOP_ON_FACE_RETURN_SECONDS,
)
from detection import annotate_faces, detect_faces, load_face_detector
from phone_usage_model import PhoneUsageModel
from ui import build_main_window
from video_popup import VideoPopup


ui = build_main_window()
window = ui["window"]

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
phone_usage_start_time = None
phone_usage_score = 0.0
phone_detect_confirm_count = 0
face_present_start_time = 0.0
last_face_seen_at = 0.0
missed_face_frames = 0
face_return_start_time = 0.0
cooldown_until = 0.0
intervention_active = False
video_popup = VideoPopup(window)
phone_usage_model = None
next_analysis_at = 0.0
last_analysis_at = 0.0
cached_faces = ()
cached_face_found = False
cached_phone_probability = None
phone_probability_ema = 0.0
analysis_lock = threading.Lock()
analysis_in_flight = False
analysis_result_version = 0
last_consumed_analysis_version = 0
pending_analysis_error = None
analysis_session_id = 0

PHONE_CLEARANCE_BUFFER = 0.05
PHONE_PROBABILITY_RAW_WEIGHT = 0.45
PHONE_PROBABILITY_EMA_WEIGHT = 0.55
ANALYSIS_EMA_KEEP_WEIGHT = 0.65
ANALYSIS_EMA_NEW_WEIGHT = 0.35
ANALYSIS_EMA_DECAY_WEIGHT = 0.7
INTERVENTION_PHONE_STATUS = "Status: Intervention (Phone)"
INTERVENTION_ATTENTION_STATUS = "Status: Intervention (Attention)"
INTERVENTION_BASE_STATUS = "Status: Intervention"


def _set_preview_idle(text="Camera preview stopped"):
    preview_label.config(image="", text=text)
    preview_label.image = None


def _cancel_preview_job():
    global preview_job

    if preview_job is not None:
        window.after_cancel(preview_job)
        preview_job = None


def _reset_analysis_state(*, reset_consumed_version=True, advance_session=True):
    global next_analysis_at, last_analysis_at, cached_faces, cached_face_found, cached_phone_probability
    global phone_probability_ema, analysis_in_flight, analysis_result_version, last_consumed_analysis_version
    global pending_analysis_error, analysis_session_id

    next_analysis_at = 0.0
    last_analysis_at = 0.0
    cached_faces = ()
    cached_face_found = False
    cached_phone_probability = None
    phone_probability_ema = 0.0
    analysis_in_flight = False
    analysis_result_version = 0
    if reset_consumed_version:
        last_consumed_analysis_version = 0
    pending_analysis_error = None
    if advance_session:
        analysis_session_id += 1


def _reset_monitoring_state(*, keep_monitoring=False, last_seen_at=0.0):
    global is_monitoring, camera, face_detector, distraction_start_time, phone_usage_start_time
    global cooldown_until, intervention_active, phone_usage_model, last_face_seen_at, missed_face_frames
    global face_present_start_time, face_return_start_time, phone_usage_score, phone_detect_confirm_count

    camera = None
    face_detector = None
    distraction_start_time = None
    phone_usage_start_time = None
    phone_usage_score = 0.0
    phone_detect_confirm_count = 0
    face_present_start_time = 0.0
    face_return_start_time = 0.0
    last_face_seen_at = last_seen_at
    missed_face_frames = 0
    cooldown_until = 0.0
    intervention_active = False
    phone_usage_model = None
    is_monitoring = keep_monitoring


def _reset_distraction_tracking(*, now=None):
    global distraction_start_time, phone_usage_start_time, phone_usage_score, phone_detect_confirm_count
    global face_present_start_time, face_return_start_time, last_face_seen_at, missed_face_frames
    global next_analysis_at, last_analysis_at, phone_probability_ema, last_consumed_analysis_version

    distraction_start_time = None
    phone_usage_start_time = None
    phone_usage_score = 0.0
    phone_detect_confirm_count = 0
    face_present_start_time = now or 0.0
    face_return_start_time = 0.0
    last_face_seen_at = now or 0.0
    missed_face_frames = 0
    next_analysis_at = 0.0
    last_analysis_at = 0.0
    phone_probability_ema = 0.0
    last_consumed_analysis_version = analysis_result_version


def _effective_phone_probability(phone_probability, ema_probability):
    raw_probability = phone_probability or 0.0
    return (
        (raw_probability * PHONE_PROBABILITY_RAW_WEIGHT)
        + (ema_probability * PHONE_PROBABILITY_EMA_WEIGHT)
    )


def _phone_thresholds(model):
    if model is None:
        return None, None

    phone_threshold = max(
        PHONE_USAGE_THRESHOLD,
        getattr(model, "decision_threshold", PHONE_USAGE_THRESHOLD),
    )
    strong_phone_threshold = min(max(phone_threshold + PHONE_USAGE_STRONG_MARGIN, 0.58), 0.96)
    return phone_threshold, strong_phone_threshold


def _set_analysis_decay(ema_probability):
    global phone_probability_ema

    phone_probability_ema = max(ema_probability * ANALYSIS_EMA_DECAY_WEIGHT, 0.0)


def _decrease_phone_usage_score(amount):
    global phone_usage_score

    phone_usage_score = max(phone_usage_score - amount, 0.0)


def _increase_phone_usage_score(amount, upper_bound):
    global phone_usage_score

    phone_usage_score = min(phone_usage_score + amount, upper_bound)


def _clear_phone_usage_tracking():
    global phone_usage_start_time, phone_detect_confirm_count

    phone_usage_start_time = None
    phone_detect_confirm_count = 0


def _set_phone_usage_start(now):
    global phone_usage_start_time

    if phone_usage_start_time is None:
        phone_usage_start_time = now


def _queue_analysis_if_needed(frame, now, snapshot):
    should_queue_analysis = (
        (snapshot["next_analysis_at"] == 0.0 or now >= snapshot["next_analysis_at"])
        and not snapshot["analysis_in_flight"]
    )
    if should_queue_analysis:
        analyze_current_frame(frame, now)
        return get_analysis_snapshot()
    return snapshot


def _refresh_preview_image(frame, faces):
    annotated_frame = annotate_faces(frame, faces)
    preview_image, preview_error = get_preview_image_from_frame(annotated_frame)

    if preview_error is not None:
        fail_monitoring(preview_error)
        return False

    preview_label.config(image=preview_image, text="")
    preview_label.image = preview_image
    return True


def _format_playing_debug(reason, phone_probability=None, ema_probability=None, effective_probability=None):
    video_name = "meme video"
    if video_popup.current_video_path is not None:
        video_name = Path(video_popup.current_video_path).name

    if phone_probability is None or ema_probability is None or effective_probability is None:
        return f"Debug: {reason} | Playing {video_name}"

    return (
        f"Debug: Playing {video_name} | phone {phone_probability:.2f} "
        f"| smooth {ema_probability:.2f} | effective {effective_probability:.2f}"
    )


def _handle_intervention_preview(now, loop_started_at, face_found, phone_probability, ema_probability):
    global face_return_start_time

    if face_found:
        phone_threshold, strong_phone_threshold = _phone_thresholds(phone_usage_model)
        effective_probability = ema_probability
        total_face_return_stop_seconds = (
            VIDEO_STOP_ON_FACE_RETURN_SECONDS + VIDEO_EXTRA_PLAY_SECONDS_ON_FACE_RETURN
        )

        if phone_usage_model is not None and phone_probability is not None:
            effective_probability = _effective_phone_probability(phone_probability, ema_probability)

        if (
            phone_probability is not None
            and strong_phone_threshold is not None
            and phone_probability >= strong_phone_threshold
            and ema_probability >= strong_phone_threshold
        ):
            face_return_start_time = 0.0
            status_text.set(INTERVENTION_PHONE_STATUS)
            debug_text.set(
                _format_playing_debug(
                    "phone trigger",
                    phone_probability,
                    ema_probability,
                    effective_probability,
                )
            )
        else:
            if face_return_start_time == 0.0:
                face_return_start_time = now

            return_seconds = now - face_return_start_time
            remaining = max(total_face_return_stop_seconds - return_seconds, 0.0)
            status_text.set(INTERVENTION_BASE_STATUS)

            if phone_probability is not None and phone_threshold is not None:
                debug_text.set(
                    f"Debug: Face returned | phone clear {effective_probability:.2f}/{phone_threshold:.2f} "
                    f"| raw {phone_probability:.2f} | stopping video in {remaining:.1f}s"
                )
            else:
                debug_text.set(f"Debug: Face returned | stopping video in {remaining:.1f}s")

            if return_seconds >= total_face_return_stop_seconds:
                video_popup.close()
                return True
    else:
        face_return_start_time = 0.0

    if status_text.get() not in (INTERVENTION_PHONE_STATUS, INTERVENTION_ATTENTION_STATUS):
        status_text.set(INTERVENTION_BASE_STATUS)

    if not face_found:
        debug_text.set(_format_playing_debug("intervention"))

    schedule_next_preview(loop_started_at, INTERVENTION_FRAME_INTERVAL_MS)
    return True


def _handle_face_detected(now, analysis_updated, phone_probability, ema_probability, last_analysis_at_local):
    global face_present_start_time, last_face_seen_at, missed_face_frames, distraction_start_time
    global phone_detect_confirm_count, phone_usage_start_time

    if face_present_start_time == 0.0:
        face_present_start_time = now

    last_face_seen_at = now
    missed_face_frames = 0
    distraction_start_time = None
    status_text.set("Status: Monitoring")

    if phone_usage_model is None:
        _clear_phone_usage_tracking()
        if analysis_updated:
            _decrease_phone_usage_score((ANALYSIS_INTERVAL_MS / 1000.0) * PHONE_SCORE_DECAY_PER_SECOND)
            _set_analysis_decay(ema_probability)
        debug_text.set("Debug: Face detected")
        return False

    analysis_seconds = max(
        now - last_analysis_at_local if last_analysis_at_local else (ANALYSIS_INTERVAL_MS / 1000.0),
        FRAME_INTERVAL_MS / 1000.0,
    )
    face_stable_seconds = now - face_present_start_time
    phone_threshold, strong_phone_threshold = _phone_thresholds(phone_usage_model)
    phone_probability = phone_probability or 0.0
    effective_probability = _effective_phone_probability(phone_probability, ema_probability)

    if face_stable_seconds < PHONE_USAGE_MIN_FACE_STABLE_SECONDS:
        _clear_phone_usage_tracking()
        if analysis_updated:
            _decrease_phone_usage_score(analysis_seconds * PHONE_SCORE_DECAY_PER_SECOND)
        debug_text.set(
            f"Debug: Face detected | stabilizing {face_stable_seconds:.1f}/{PHONE_USAGE_MIN_FACE_STABLE_SECONDS:.1f}s "
            f"| score {phone_usage_score:.1f}/{PHONE_SCORE_TRIGGER:.1f}"
        )
        return False

    if phone_probability >= strong_phone_threshold and ema_probability >= strong_phone_threshold:
        if analysis_updated:
            phone_detect_confirm_count += 1

        if phone_detect_confirm_count < PHONE_DETECT_CONFIRM_FRAMES:
            phone_usage_start_time = None
            debug_text.set(
                f"Debug: Possible phone raw {phone_probability:.2f} | smooth {ema_probability:.2f} "
                f"| effective {effective_probability:.2f}/{strong_phone_threshold:.2f} "
                f"| confirm {phone_detect_confirm_count}/{PHONE_DETECT_CONFIRM_FRAMES}"
            )
            return False

        _set_phone_usage_start(now)
        phone_seconds = now - phone_usage_start_time
        if analysis_updated:
            _increase_phone_usage_score(analysis_seconds * PHONE_SCORE_GAIN_PER_SECOND, PHONE_USAGE_STREAK_SECONDS)
        status_text.set("Status: Phone Usage")
        debug_text.set(
            f"Debug: Face detected | raw {phone_probability:.2f} | smooth {ema_probability:.2f} "
            f"| effective {effective_probability:.2f}/{strong_phone_threshold:.2f} for {phone_seconds:.1f}s"
        )

        if phone_seconds >= PHONE_USAGE_STREAK_SECONDS:
            trigger_intervention(reason="phone")
            return True

        return False

    suspicious_threshold = max(phone_threshold - PHONE_CLEARANCE_BUFFER, 0.55)
    if phone_probability >= suspicious_threshold or ema_probability >= suspicious_threshold:
        phone_usage_start_time = None
        if analysis_updated:
            phone_detect_confirm_count = 0
            _decrease_phone_usage_score(analysis_seconds * PHONE_SCORE_DECAY_PER_SECOND)
        debug_text.set(
            f"Debug: Suspicious phone signal raw {phone_probability:.2f} | smooth {ema_probability:.2f} "
            f"| effective {effective_probability:.2f}/{phone_threshold:.2f} | not counting"
        )
        return False

    _clear_phone_usage_tracking()
    if analysis_updated:
        _decrease_phone_usage_score(analysis_seconds * PHONE_SCORE_DECAY_PER_SECOND)
    debug_text.set(
        f"Debug: Face detected | phone score {phone_probability:.2f}/{phone_threshold:.2f} "
        f"| smooth {ema_probability:.2f} | total {phone_usage_score:.1f}/{PHONE_SCORE_TRIGGER:.1f}"
    )
    return False


def _handle_missing_face(now, analysis_updated, ema_probability):
    global distraction_start_time, face_present_start_time, missed_face_frames

    _clear_phone_usage_tracking()
    if analysis_updated:
        _decrease_phone_usage_score((ANALYSIS_INTERVAL_MS / 1000.0) * PHONE_SCORE_DECAY_PER_SECOND)
        _set_analysis_decay(ema_probability)
        missed_face_frames += 1

    recently_saw_face = last_face_seen_at > 0 and (now - last_face_seen_at) <= FACE_DETECTION_GRACE_SECONDS
    if recently_saw_face or missed_face_frames < FACE_MISS_CONFIRM_FRAMES:
        distraction_start_time = None
        status_text.set("Status: Monitoring")
        grace_remaining = (
            max(FACE_DETECTION_GRACE_SECONDS - (now - last_face_seen_at), 0.0)
            if last_face_seen_at > 0
            else 0.0
        )
        debug_text.set(
            f"Debug: Face detection unstable | miss {missed_face_frames}/{FACE_MISS_CONFIRM_FRAMES} "
            f"| grace {grace_remaining:.1f}s"
        )
        return False

    face_present_start_time = 0.0
    if distraction_start_time is None:
        distraction_start_time = now

    missing_seconds = now - distraction_start_time
    debug_text.set(f"Debug: No face detected for {missing_seconds:.1f}s")

    if missing_seconds >= ATTENTION_STREAK_SECONDS:
        trigger_intervention(reason="attention")
        return True

    status_text.set("Status: Monitoring")
    return False


def refresh_controls():
    controls_active = intervention_active or is_monitoring
    start_button.config(state="disabled" if controls_active else "normal")
    stop_button.config(state="normal" if controls_active else "disabled")


def schedule_preview(delay_ms=FRAME_INTERVAL_MS):
    global preview_job

    preview_job = window.after(max(1, int(delay_ms)), update_preview)


def schedule_next_preview(loop_started_at, target_interval_ms=FRAME_INTERVAL_MS):
    elapsed_ms = (time.perf_counter() - loop_started_at) * 1000.0
    remaining_ms = max(1, int(round(target_interval_ms - elapsed_ms)))
    schedule_preview(remaining_ms)


def _prepare_analysis_frame(frame):
    if cv2 is None:
        return frame.copy(), frame.shape[:2], frame.shape[:2]

    frame_height, frame_width = frame.shape[:2]
    scale = min(
        ANALYSIS_FRAME_WIDTH / max(frame_width, 1),
        ANALYSIS_FRAME_HEIGHT / max(frame_height, 1),
        1.0,
    )

    if scale >= 0.999:
        return frame.copy(), frame.shape[:2], frame.shape[:2]

    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized_frame, (frame_height, frame_width), (resized_height, resized_width)


def _scale_faces_to_frame(faces, source_shape, target_shape):
    source_height, source_width = source_shape
    target_height, target_width = target_shape

    if source_height == target_height and source_width == target_width:
        return tuple(tuple(int(value) for value in face) for face in faces)

    scale_x = target_width / max(source_width, 1)
    scale_y = target_height / max(source_height, 1)
    scaled_faces = []

    for x, y, width, height in faces:
        scaled_faces.append(
            (
                int(round(x * scale_x)),
                int(round(y * scale_y)),
                int(round(width * scale_x)),
                int(round(height * scale_y)),
            )
        )

    return tuple(scaled_faces)


def _run_frame_analysis(frame, analyzed_at, session_id):
    global analysis_in_flight, analysis_result_version, cached_faces, cached_face_found, cached_phone_probability
    global last_analysis_at, phone_probability_ema, pending_analysis_error

    try:
        analysis_frame, original_shape, analysis_shape = _prepare_analysis_frame(frame)
        detector = face_detector
        model = phone_usage_model

        faces, detect_error = detect_faces(analysis_frame, detector)
        if detect_error is not None:
            with analysis_lock:
                if session_id == analysis_session_id and is_monitoring:
                    pending_analysis_error = detect_error
                analysis_in_flight = False
            return

        scaled_faces = _scale_faces_to_frame(faces, analysis_shape, original_shape)
        phone_probability = None

        if model is not None and scaled_faces:
            phone_probability = model.predict_frame_probability(frame, faces=scaled_faces)

        with analysis_lock:
            if session_id != analysis_session_id or not is_monitoring:
                analysis_in_flight = False
                return

            cached_faces = scaled_faces
            cached_face_found = len(cached_faces) > 0
            cached_phone_probability = phone_probability
            phone_probability_ema = (
                (phone_probability_ema * ANALYSIS_EMA_KEEP_WEIGHT)
                + ((phone_probability or 0.0) * ANALYSIS_EMA_NEW_WEIGHT)
            )
            last_analysis_at = analyzed_at
            analysis_result_version += 1
            analysis_in_flight = False
    except Exception as exc:
        with analysis_lock:
            if session_id == analysis_session_id and is_monitoring:
                pending_analysis_error = f"Background analysis crashed: {exc}"
            analysis_in_flight = False


def queue_frame_analysis(frame, now):
    global analysis_in_flight, next_analysis_at, pending_analysis_error

    with analysis_lock:
        if analysis_in_flight:
            return False

        analysis_in_flight = True
        next_analysis_at = now + (ANALYSIS_INTERVAL_MS / 1000.0)
        pending_analysis_error = None
        session_id = analysis_session_id

    worker = threading.Thread(
        target=_run_frame_analysis,
        args=(frame.copy(), now, session_id),
        daemon=True,
    )
    worker.start()
    return True


def get_analysis_snapshot():
    with analysis_lock:
        return {
            "error": pending_analysis_error,
            "version": analysis_result_version,
            "faces": tuple(cached_faces),
            "face_found": cached_face_found,
            "phone_probability": cached_phone_probability,
            "phone_probability_ema": phone_probability_ema,
            "last_analysis_at": last_analysis_at,
            "next_analysis_at": next_analysis_at,
            "analysis_in_flight": analysis_in_flight,
        }


def set_error_state(message):
    status_text.set("Status: Error")
    debug_text.set(f"Debug: {message}")


def fail_monitoring(message):
    _cancel_preview_job()
    video_popup.close(notify=False)
    close_camera(camera)
    _reset_monitoring_state()
    _reset_analysis_state()

    _set_preview_idle()
    refresh_controls()
    set_error_state(message)


def load_phone_usage_model():
    if not PHONE_USAGE_MODEL_PATH.exists():
        return None

    try:
        return PhoneUsageModel.load(PHONE_USAGE_MODEL_PATH)
    except Exception:
        return None


def handle_video_finished():
    global intervention_active, cooldown_until

    intervention_active = False
    now = time.time()
    _reset_distraction_tracking(now=now)
    refresh_controls()

    if not is_monitoring:
        return

    cooldown_until = now + COOLDOWN_SECONDS
    status_text.set("Status: Cooldown")
    debug_text.set(f"Debug: Cooldown for {COOLDOWN_SECONDS:.1f}s")

    if preview_job is None:
        schedule_preview()


def trigger_intervention(reason="attention"):
    global intervention_active

    if intervention_active or not is_monitoring:
        return

    intervention_active = True
    _clear_phone_usage_tracking()
    _reset_distraction_tracking()
    refresh_controls()

    if reason == "phone":
        status_text.set(INTERVENTION_PHONE_STATUS)
    else:
        status_text.set(INTERVENTION_ATTENTION_STATUS)

    popup_error = video_popup.play(on_complete=handle_video_finished)

    if popup_error is not None:
        intervention_active = False
        refresh_controls()
        set_error_state(popup_error)
        schedule_preview()
        return

    debug_text.set(_format_playing_debug(f"{reason} trigger"))

    if preview_job is None:
        schedule_preview()


def analyze_current_frame(frame, now):
    return queue_frame_analysis(frame, now)


def update_preview():
    global preview_job, last_consumed_analysis_version

    preview_job = None
    loop_started_at = time.perf_counter()

    if not is_monitoring or camera is None:
        return

    frame, frame_error = read_frame(camera)

    if frame_error is not None:
        fail_monitoring(frame_error)
        return

    now = time.time()
    snapshot = get_analysis_snapshot()

    if snapshot["error"] is not None:
        fail_monitoring(snapshot["error"])
        return

    snapshot = _queue_analysis_if_needed(frame, now, snapshot)

    analysis_updated = snapshot["version"] != last_consumed_analysis_version
    if analysis_updated:
        last_consumed_analysis_version = snapshot["version"]

    face_found = snapshot["face_found"]
    faces = snapshot["faces"]
    phone_probability = snapshot["phone_probability"]
    phone_probability_ema_local = snapshot["phone_probability_ema"]
    last_analysis_at_local = snapshot["last_analysis_at"]

    if not intervention_active:
        if not _refresh_preview_image(frame, faces):
            return

    if intervention_active:
        if _handle_intervention_preview(
            now,
            loop_started_at,
            face_found,
            phone_probability,
            phone_probability_ema_local,
        ):
            return
        return

    if now < cooldown_until:
        remaining = cooldown_until - now
        status_text.set("Status: Cooldown")
        debug_text.set(f"Debug: Cooldown {remaining:.1f}s remaining")
    elif face_found:
        if _handle_face_detected(
            now,
            analysis_updated,
            phone_probability,
            phone_probability_ema_local,
            last_analysis_at_local,
        ):
            return
    else:
        if _handle_missing_face(now, analysis_updated, phone_probability_ema_local):
            return

    schedule_next_preview(loop_started_at)


def start_monitoring():
    global is_monitoring, camera, face_detector, phone_usage_model

    if is_monitoring:
        return

    detector, detector_error = load_face_detector()

    if detector_error is not None:
        set_error_state(detector_error)
        return

    opened_camera, camera_error = open_camera()

    if camera_error is not None:
        set_error_state(camera_error)
        return

    now = time.time()
    _reset_monitoring_state(keep_monitoring=True, last_seen_at=now)
    camera = opened_camera
    face_detector = detector
    is_monitoring = True
    phone_usage_model = load_phone_usage_model()
    _reset_analysis_state()
    refresh_controls()

    status_text.set("Status: Monitoring")
    if phone_usage_model is not None:
        debug_text.set("Debug: Camera, detector, and phone model ready")
    else:
        debug_text.set("Debug: Camera and detector ready")
    update_preview()


def stop_monitoring():
    _cancel_preview_job()
    video_popup.close(notify=False)
    close_camera(camera)
    _reset_monitoring_state()
    _reset_analysis_state()
    refresh_controls()

    _set_preview_idle()
    status_text.set("Status: Stopped")
    debug_text.set("Debug: Camera released")


def on_window_close():
    stop_monitoring()
    window.destroy()


start_button.config(command=start_monitoring)
stop_button.config(command=stop_monitoring)
window.protocol("WM_DELETE_WINDOW", on_window_close)
refresh_controls()

window.mainloop()
