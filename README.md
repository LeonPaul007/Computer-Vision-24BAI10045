# Gesture Drawing - Computer Vision Virtual Drawing Application

An interactive gesture-controlled drawing application built in Python using **OpenCV**, **MediaPipe**, and **NumPy**. Draw in real time using hand gestures tracked through a webcam.

## Project Structure

```text
Computer-Vision-24BAI10045/
├── hand_tracker.py
├── canvas_ui.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Features

- 21-point hand landmark tracking with MediaPipe Hands.
- Selection mode using index + middle fingers.
- Drawing mode using only the index finger.
- Blue, green, red, and yellow drawing tools.
- Gesture-controlled eraser and clear screen.
- Live webcam mirror view.
- FPS and current tool/mode HUD.

## Requirements

- Python 3.8–3.12
- Functional webcam

## Installation

```bash
python -m venv venv
```

### Windows

```bash
.\venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Gesture Controls

| Gesture | Action |
|---|---|
| Index + Middle fingers up | Selection mode; hover over toolbar buttons |
| Only Index finger up | Draw or erase |
| Fingers lowered / fist | Standby / pause |

## Keyboard Controls

- `E` — activate eraser
- `C` — clear canvas
- `Q` or `ESC` — quit

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## Author

Leon Paul Malayil
