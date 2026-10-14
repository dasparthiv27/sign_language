# Sign Language Recognition

A webcam-based sign language recognition project using MediaPipe hand landmarks and a scikit-learn classifier.

## Setup

```bash
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Start webcam prediction:

```bash
python realtime_predict.py
```

Press `q` to close the camera window.

To test camera access independently:

```bash
python test_camera.py
```

## Training

The image dataset is stored under `dataset/`, with one subfolder per sign label. Extract hand landmarks and train a classifier with:

```bash
python extract_landmarks.py
python train_model.py
```

The extraction script writes `data/X.npy` and `data/y.npy`; training writes `sign_model.pkl`. The realtime script currently loads that model file.
