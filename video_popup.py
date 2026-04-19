import random
import sys
from pathlib import Path
import tkinter as tk
import vlc

from config import (
    MEME_VIDEO_DIR,
    SUPPORTED_VIDEO_EXTENSIONS,
    VIDEO_FALLBACK_DELAY_MS,
    VIDEO_POPUP_MAX_HEIGHT,
    VIDEO_POPUP_MAX_WIDTH,
)


class VideoPopup:
    def __init__(self, parent_window):
        self.parent_window = parent_window
        self.popup = None
        self.video_frame = None
        self.vlc_instance = None
        self.media_player = None
        self.video_job = None
        self.on_complete = None
        self.current_video_path = None
        self.playback_started = False
        self.end_reached = False
        self.error_seen = False

    def _block_manual_close(self, _event=None):
        return "break"

    def _safe_after_cancel(self, job):
        if job is None or self.popup is None:
            return
        try:
            self.popup.after_cancel(job)
        except tk.TclError:
            pass

    def _safe_destroy_popup(self):
        if self.popup is None:
            return
        try:
            if self.popup.winfo_exists():
                self.popup.destroy()
        except tk.TclError:
            pass

    def _create_vlc_instance(self):
        instance_args = [
            "--no-video-title-show",
            "--quiet",
            "--no-metadata-network-access",
        ]

        if sys.platform.startswith("win"):
            instance_args.extend(
                [
                    "--avcodec-hw=none",
                    "--vout=directx",
                ]
            )

        return vlc.Instance(*instance_args)

    def open(self):
        if self.popup is not None:
            return

        self.popup = tk.Toplevel(self.parent_window)
        self.popup.title("Focus Check")
        self.popup.configure(bg="black")
        self.popup.attributes("-topmost", True)
        self.popup.overrideredirect(True)
        self.popup.protocol("WM_DELETE_WINDOW", self._block_manual_close)
        self.popup.bind("<Escape>", self._block_manual_close)
        self.popup.bind("<Alt-F4>", self._block_manual_close)
        self.popup.resizable(False, False)
        self.popup.transient(self.parent_window)

        screen_width = self.popup.winfo_screenwidth()
        screen_height = self.popup.winfo_screenheight()
        popup_x = max((screen_width - VIDEO_POPUP_MAX_WIDTH) // 2, 0)
        popup_y = max((screen_height - VIDEO_POPUP_MAX_HEIGHT) // 2, 0)
        self.popup.geometry(f"{VIDEO_POPUP_MAX_WIDTH}x{VIDEO_POPUP_MAX_HEIGHT}+{popup_x}+{popup_y}")

        self.video_frame = tk.Frame(
            self.popup,
            bg="black",
        )
        self.video_frame.pack(fill="both", expand=True)
        self.popup.lift()
        self.popup.focus_force()

    def play(self, on_complete=None):
        video_path = self._choose_random_video()

        if video_path is None:
            return f"No supported videos found in: {MEME_VIDEO_DIR}"

        self.close(notify=False)
        self.open()

        self.current_video_path = video_path
        self.on_complete = on_complete
        self.playback_started = False
        self.end_reached = False
        self.error_seen = False
        self.popup.update_idletasks()

        self.vlc_instance = self._create_vlc_instance()
        self.media_player = self.vlc_instance.media_player_new()
        media = self.vlc_instance.media_new(str(video_path))
        self.media_player.set_media(media)

        handle = self.video_frame.winfo_id()
        if sys.platform.startswith("win"):
            self.media_player.set_hwnd(handle)
        elif sys.platform.startswith("linux"):
            self.media_player.set_xwindow(handle)
        elif sys.platform == "darwin":
            self.media_player.set_nsobject(handle)

        event_manager = self.media_player.event_manager()
        event_manager.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_media_end)
        event_manager.event_attach(vlc.EventType.MediaPlayerEncounteredError, self._on_media_error)

        start_result = self.media_player.play()
        if start_result == -1:
            self.close(notify=False)
            return "Could not start VLC playback."

        self.video_job = self.popup.after(max(VIDEO_FALLBACK_DELAY_MS, 200), self._poll_player_state)
        return None

    def _choose_random_video(self):
        video_dir = Path(MEME_VIDEO_DIR)

        if not video_dir.exists():
            return None

        videos = [
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

        if not videos:
            return None

        return random.choice(videos)

    def _on_media_end(self, _event):
        self.end_reached = True

    def _on_media_error(self, _event):
        self.error_seen = True

    def _poll_player_state(self):
        self.video_job = None

        if self.media_player is None or self.popup is None:
            return

        state = self.media_player.get_state()
        if state == vlc.State.Playing:
            self.playback_started = True

        if self.end_reached or state == vlc.State.Ended:
            self.close()
            return

        if self.error_seen or state == vlc.State.Error:
            self.close()
            return

        media_length = self.media_player.get_length()
        current_time = self.media_player.get_time()

        if self.playback_started and media_length > 0 and current_time >= media_length - 250:
            self.close()
            return

        self.video_job = self.popup.after(max(VIDEO_FALLBACK_DELAY_MS, 200), self._poll_player_state)

    def close(self, notify=True):
        callback = self.on_complete
        self.on_complete = None

        self._safe_after_cancel(self.video_job)
        self.video_job = None

        if self.media_player is not None:
            try:
                self.media_player.stop()
            except Exception:
                pass
            self.media_player = None

        self.vlc_instance = None
        self.current_video_path = None
        self.playback_started = False
        self.end_reached = False
        self.error_seen = False

        if self.popup is not None:
            self._safe_destroy_popup()
            self.popup = None
            self.video_frame = None

        if notify and callback is not None:
            callback()
