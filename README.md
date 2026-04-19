# Focus Intervention System

A Python desktop app that monitors a live webcam feed, looks for attention loss or possible phone usage, and responds with a forced video interruption.

At its core, this system is a focus-monitoring experiment built out of computer vision, shitposting, and memeing.

## What It Does

- opens your webcam and shows a live preview
- detects faces with OpenCV Haar cascades
- watches for sustained absence from the camera
- estimates possible phone usage with a custom NumPy model
- plays a meme/intervention video in a popup when a trigger condition is met

## Project Files

```text
Prototype_3 Focus_Intervention_System/
|-- main.py
|-- ui.py
|-- camera.py
|-- detection.py
|-- phone_usage_model.py
|-- train_phone_usage_model.py
|-- video_popup.py
|-- config.py
|-- dataset/
|-- models/
`-- meme_videos/
```

## Requirements

- Python 3
- Tkinter
- VLC installed on your machine

Install the Python packages with:

```powershell
pip install numpy pillow opencv-python python-vlc
```

## Run The App

From this folder:

```powershell
cd "D:\Programming\Python\Practice Python\Prototype_3 Focus_Intervention_System"
python main.py
```

## Train The Phone Model

If you want to retrain the phone-usage model using the images in `dataset/`:

```powershell
cd "D:\Programming\Python\Practice Python\Prototype_3 Focus_Intervention_System"
python train_phone_usage_model.py
```

This saves the trained model to `models/phone_usage_model.npz`.

## Dataset Notes

- `dataset/positive/` contains phone-related samples
- `dataset/negative/` contains non-phone samples
- `dataset/labels.csv` stores bounding boxes for labeled positive images

## Dataset Attribution

This project uses a locally bundled dataset stored in `dataset/` and annotated through `dataset/labels.csv`.

At the moment, the original external source, author, and license for these images are not documented anywhere in this repository. If this dataset came from a public dataset, website, Kaggle source, Roboflow project, or another creator, that original source should be credited here.

Suggested credit format once you know it:

```text
Dataset source: <name of dataset or creator>
Original link: <URL>
License: <license name>
Notes: locally relabeled / filtered / cropped for phone-usage detection
```

## Video Notes

- place your intervention or meme videos in `meme_videos/`
- supported formats are `.mp4`, `.mov`, `.avi`, `.mkv`, and `.webm`
- the app currently includes a placeholder file in that folder, so you can replace it with real clips

## How It Works

`main.py` coordinates the full monitoring loop. It opens the camera, schedules background frame analysis, updates the Tkinter UI, and triggers interventions.

`detection.py` handles face detection using multiple Haar cascade passes and merges likely face boxes.

`phone_usage_model.py` contains the custom feature pipeline, training logic, threshold selection, and frame scoring used for phone detection.

`video_popup.py` uses `python-vlc` to display a topmost popup that plays a randomly selected intervention video.

## Current Goal

This prototype looks like a practice project for combining:

- computer vision
- Tkinter UI work
- lightweight machine learning
- real-time behavioral feedback
