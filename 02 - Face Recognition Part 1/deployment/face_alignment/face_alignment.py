import dlib
import cv2
import numpy as np
from PIL import Image

import face_blend_common as fbc


def load_image(image_bytes):
    print('Reading image')
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def load_models(predictor_path):
    print('Initializing the face detector instance')
    face_detector = dlib.get_frontal_face_detector()

    print('Loading the shape predictor model')
    landmark_detector = dlib.shape_predictor(predictor_path)

    return face_detector, landmark_detector


def normalize_image(img, points):
    print('Aligning Image')
    img_norm, points = fbc.normalizeImagesAndLandmarks((600, 600), img, points)
    img_norm = np.uint8(img_norm * 255)
    return img_norm


def align_image(predictor_path, image_bytes):
    # Load models and image
    face_detector, landmark_detector = load_models(predictor_path)
    img = load_image(image_bytes)

    # Detect landmarks
    status, result = fbc.getLandmarks(face_detector, landmark_detector, img)
    if status == 'fail':
        return status, result
    
    points = np.array(result)

    # Convert image to floating point in the range 0 to 1
    img = np.float32(img) / 255.0

    # Align image
    output = normalize_image(img, points)

    return status, Image.fromarray(output)
