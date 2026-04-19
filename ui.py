import tkinter as tk

from config import PREVIEW_MAX_HEIGHT, PREVIEW_MAX_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH


BG_TOP = "#04171A"
BG_MID = "#072328"
BG_BOTTOM = "#021012"
PANEL_BG = "#08181C"
PANEL_BORDER = "#18C6BF"
PANEL_SHADOW = "#021012"
TITLE_COLOR = "#E8FFFE"
ACCENT_COLOR = "#46F6E9"
MUTED_COLOR = "#8FB9B7"
BUTTON_BG = "#0D2E33"
BUTTON_ACTIVE_BG = "#17454C"
BUTTON_FG = "#E7FFFF"
PREVIEW_BORDER = "#2EE3D8"
PREVIEW_BG = "#020A0C"


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _mix_color(start_color, end_color, factor):
    start_rgb = _hex_to_rgb(start_color)
    end_rgb = _hex_to_rgb(end_color)
    mixed = tuple(
        int(start + (end - start) * factor)
        for start, end in zip(start_rgb, end_rgb)
    )
    return _rgb_to_hex(mixed)


def draw_background(canvas, width, height):
    canvas.delete("bg")

    if width <= 1 or height <= 1:
        return

    bands = 80
    for index in range(bands):
        top_y = int((index / bands) * height)
        bottom_y = int(((index + 1) / bands) * height)

        if index < bands // 2:
            factor = index / max(bands // 2 - 1, 1)
            color = _mix_color(BG_TOP, BG_MID, factor)
        else:
            factor = (index - bands // 2) / max(bands // 2 - 1, 1)
            color = _mix_color(BG_MID, BG_BOTTOM, factor)

        canvas.create_rectangle(
            0,
            top_y,
            width,
            bottom_y,
            fill=color,
            outline="",
            tags="bg",
        )

    glow_specs = [
        (width * 0.18, height * 0.16, 260, "#0E545B"),
        (width * 0.82, height * 0.20, 240, "#0B4348"),
        (width * 0.72, height * 0.78, 300, "#09373C"),
    ]

    for center_x, center_y, radius, color in glow_specs:
        canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill=color,
            outline="",
            stipple="gray25",
            tags="bg",
        )

    for offset in range(0, width, 48):
        canvas.create_line(
            offset,
            0,
            offset - height * 0.18,
            height,
            fill="#0A2B2F",
            width=1,
            tags="bg",
        )


def build_main_window():
    window = tk.Tk()
    window.title("Focus Intervention System")
    window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    window.minsize(1360, 940)
    window.configure(bg=BG_BOTTOM)

    background_canvas = tk.Canvas(
        window,
        highlightthickness=0,
        bd=0,
        bg=BG_BOTTOM,
    )
    background_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    content_shell = tk.Frame(
        window,
        bg=PANEL_SHADOW,
        highlightthickness=0,
        bd=0,
    )
    content_shell.place(relx=0.5, rely=0.5, anchor="center")

    content_frame = tk.Frame(
        content_shell,
        bg=PANEL_BG,
        highlightbackground=PANEL_BORDER,
        highlightthickness=2,
        bd=0,
    )
    content_frame.pack(padx=(0, 6), pady=(0, 6))

    def refresh_layout(event=None):
        width = max(window.winfo_width(), 1)
        height = max(window.winfo_height(), 1)
        draw_background(background_canvas, width, height)

        panel_width = min(max(int(width * 0.88), 1280), 1380)
        panel_height = min(max(int(height * 0.88), 860), 980)
        content_shell.place_configure(width=panel_width + 10, height=panel_height + 10)
        content_frame.configure(width=panel_width, height=panel_height)
        content_frame.pack_propagate(False)
        reserved_height = 300
        preview_height = min(PREVIEW_MAX_HEIGHT, max(panel_height - reserved_height, 300))
        preview_frame.configure(height=preview_height)

    window.bind("<Configure>", refresh_layout)

    status_text = tk.StringVar(value="Status: Not started")
    debug_text = tk.StringVar(value="Debug: Waiting to start")

    top_bar = tk.Frame(content_frame, bg=PANEL_BG)
    top_bar.pack(fill="x", padx=34, pady=(26, 10))

    eyebrow_label = tk.Label(
        top_bar,
        text="VISUAL FOCUS INTERVENTION PROTOCOL",
        font=("Bahnschrift SemiBold", 11),
        bg=PANEL_BG,
        fg=ACCENT_COLOR,
        anchor="w",
    )
    eyebrow_label.pack(anchor="w")

    title_label = tk.Label(
        top_bar,
        text="Focus Intervention System",
        font=("Bahnschrift SemiBold", 28),
        bg=PANEL_BG,
        fg=TITLE_COLOR,
        anchor="w",
    )
    title_label.pack(anchor="w", pady=(6, 0))

    subtitle_label = tk.Label(
        top_bar,
        text="Phone usage and attention drift trigger a hard visual interruption.",
        font=("Segoe UI", 11),
        bg=PANEL_BG,
        fg=MUTED_COLOR,
        anchor="w",
    )
    subtitle_label.pack(anchor="w", pady=(8, 0))

    status_card = tk.Frame(
        content_frame,
        bg="#071216",
        highlightbackground="#114A4F",
        highlightthickness=1,
        bd=0,
    )
    status_card.pack(fill="x", padx=34, pady=(4, 16))

    status_label = tk.Label(
        status_card,
        textvariable=status_text,
        font=("Bahnschrift SemiBold", 15),
        bg="#071216",
        fg=ACCENT_COLOR,
        anchor="w",
    )
    status_label.pack(fill="x", padx=18, pady=(14, 4))

    debug_label = tk.Label(
        status_card,
        textvariable=debug_text,
        font=("Consolas", 10),
        bg="#071216",
        fg="#A5D7D4",
        anchor="w",
        justify="left",
    )
    debug_label.pack(fill="x", padx=18, pady=(0, 14))

    button_frame = tk.Frame(content_frame, bg=PANEL_BG)
    button_frame.pack(side="bottom", fill="x", padx=34, pady=(0, 28))

    button_row = tk.Frame(button_frame, bg=PANEL_BG)
    button_row.pack(anchor="center")

    preview_wrapper = tk.Frame(content_frame, bg=PANEL_BG)
    preview_wrapper.pack(fill="both", expand=True, padx=34, pady=(0, 18))

    preview_header = tk.Frame(preview_wrapper, bg=PANEL_BG)
    preview_header.pack(fill="x", pady=(0, 10))

    preview_title = tk.Label(
        preview_header,
        text="Live Camera Feed",
        font=("Bahnschrift SemiBold", 14),
        bg=PANEL_BG,
        fg=TITLE_COLOR,
        anchor="w",
    )
    preview_title.pack(side="left")

    preview_tag = tk.Label(
        preview_header,
        text="REAL-TIME",
        font=("Bahnschrift SemiBold", 9),
        bg="#0B2024",
        fg=ACCENT_COLOR,
        padx=10,
        pady=4,
    )
    preview_tag.pack(side="right")

    preview_frame = tk.Frame(
        preview_wrapper,
        width=PREVIEW_MAX_WIDTH,
        height=PREVIEW_MAX_HEIGHT,
        bg=PREVIEW_BORDER,
        bd=0,
        highlightthickness=0,
    )
    preview_frame.pack(expand=True)
    preview_frame.pack_propagate(False)

    preview_inner = tk.Frame(
        preview_frame,
        bg=PREVIEW_BG,
        bd=0,
        highlightthickness=0,
    )
    preview_inner.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.988, relheight=0.978)

    preview_label = tk.Label(
        preview_inner,
        text="Camera preview will appear here",
        font=("Bahnschrift", 16),
        bg=PREVIEW_BG,
        fg="#C6F7F3",
    )
    preview_label.pack(expand=True, fill="both")

    start_button = tk.Button(
        button_row,
        text="Start Monitoring",
        font=("Bahnschrift SemiBold", 13),
        width=18,
        padx=18,
        pady=12,
        bg=BUTTON_BG,
        fg=BUTTON_FG,
        activebackground=BUTTON_ACTIVE_BG,
        activeforeground=BUTTON_FG,
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground=PANEL_BORDER,
        cursor="hand2",
    )
    start_button.pack(side="left")

    stop_button = tk.Button(
        button_row,
        text="Stop Monitoring",
        font=("Bahnschrift SemiBold", 13),
        width=18,
        padx=18,
        pady=12,
        bg="#142326",
        fg="#D7F8F5",
        activebackground="#20363A",
        activeforeground="#D7F8F5",
        relief="flat",
        bd=0,
        highlightthickness=1,
        highlightbackground="#2C4C51",
        cursor="hand2",
        state="disabled",
    )
    stop_button.pack(side="left", padx=(14, 0))

    refresh_layout()

    return {
        "window": window,
        "status_text": status_text,
        "debug_text": debug_text,
        "preview_label": preview_label,
        "start_button": start_button,
        "stop_button": stop_button,
    }
