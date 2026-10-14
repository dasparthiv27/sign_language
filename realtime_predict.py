import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = []

    if results.multi_hand_landmarks:

        hands_sorted = sorted(
            results.multi_hand_landmarks,
            key=lambda hand: hand.landmark[0].x
        )

        for hand_landmarks in hands_sorted[:2]:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_landmarks.landmark[0]
            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

    while len(landmarks) < 126:
        landmarks.append(0)

    if len(landmarks) == 126:
        prediction = model.predict([landmarks])[0]

        cv2.putText(
            frame,
            prediction,
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Sign Language to Text", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
