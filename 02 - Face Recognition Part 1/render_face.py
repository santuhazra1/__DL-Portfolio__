import cv2
import numpy as np


def draw_polyline(img, landmarks, start, end, is_closed=False):
    points = np.array(
        [
            [landmarks.part(i).x, landmarks.part(i).y]
            for i in range(start, end + 1)
        ],
        dtype=np.int32
    )
    cv2.polylines(img, [points], is_closed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)


# Use this function for 68-points facial landmark detector model
def render_face(img, landmarks):
    assert(landmarks.num_parts == 68)
    draw_polyline(img, landmarks, 0, 16)           # Jaw line
    draw_polyline(img, landmarks, 17, 21)          # Left eyebrow
    draw_polyline(img, landmarks, 22, 26)          # Right eyebrow
    draw_polyline(img, landmarks, 27, 30)          # Nose bridge
    draw_polyline(img, landmarks, 30, 35, True)    # Lower nose
    draw_polyline(img, landmarks, 36, 41, True)    # Left eye
    draw_polyline(img, landmarks, 42, 47, True)    # Right Eye
    draw_polyline(img, landmarks, 48, 59, True)    # Outer lip
    draw_polyline(img, landmarks, 60, 67, True)    # Inner lip


# Use this function for any model other than
# 68 points facial_landmark detector model
def render_face_2(img, landmarks, color=(0, 255, 0), radius=3):
    for part in landmarks.parts():
        cv2.circle(img, (part.x, part.y), radius, color, -1)
