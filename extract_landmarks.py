import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

DATASET_PATH = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

X = []
y = []

total_images = 0
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(label_path):
        total_images += len(os.listdir(label_path))

with tqdm(total=total_images, desc="Extracting Landmarks") as pbar:
    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)

            if image is None:
                pbar.update(1)
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            landmarks = []

            if results.multi_hand_landmarks:
                # Sort hands by x-position for consistency
                hands_sorted = sorted(
                    results.multi_hand_landmarks,
                    key=lambda hand: hand.landmark[0].x
                )

                for hand_landmarks in hands_sorted[:2]:
                    wrist = hand_landmarks.landmark[0]
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([
                            lm.x - wrist.x,
                            lm.y - wrist.y,
                            lm.z - wrist.z
                        ])

            # If only 1 hand detected, pad second hand
            while len(landmarks) < 126:
                landmarks.append(0)

            if len(landmarks) == 126:
                X.append(landmarks)
                y.append(label)

            pbar.update(1)

X = np.array(X)
y = np.array(y)

os.makedirs("data", exist_ok=True)
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("\n✅ Landmarks extracted")
print("X shape:", X.shape)
print("y shape:", y.shape)
