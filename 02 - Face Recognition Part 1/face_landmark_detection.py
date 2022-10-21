import os
import argparse
import dlib
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from render_face import render_face


matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


def validate_params(predictor_path, image_filename, output_path):
    os.makedirs(output_path, exist_ok=True)
    if not os.path.isfile(image_filename):
        raise ValueError('Invalid image')
    if not os.path.isfile(predictor_path):
        raise ValueError('Invalid predictor')


def load_models_and_image(predictor_path, image_filename):
    print('Initializing the face detector instance')
    face_detector = dlib.get_frontal_face_detector()

    print('Loading the shape predictor model')
    landmark_detector = dlib.shape_predictor(predictor_path)

    print('Reading image')
    img = cv2.imread(image_filename)

    return face_detector, landmark_detector, img


def display_image(img, figsize=None, title=None):
    if not figsize is None:
        plt.figure(figsize=(15, 15))
    
    plt.imshow(img[:, :, ::-1])

    if not title is None:
        plt.title(title)
    
    plt.show()


def write_landmarks_to_file(landmarks, landmarks_filename):
    with open(landmarks_filename, 'w') as f:
        for part in landmarks.parts():
            f.write(f'{int(part.x)} {int(part.y)}\n')


def store_face_landmark(img, landmark_detector, face_rectangles, landmark_basename):
    # List to store landmarks of all detected faces
    landmarks_all = []

    # Loop over all detected face rectangles
    for i in range(len(face_rectangles)):
        new_rectangle = dlib.rectangle(
            int(face_rectangles[i].left()),
            int(face_rectangles[i].top()),
            int(face_rectangles[i].right()),
            int(face_rectangles[i].bottom()),
        )

        # For every face rectangle, run landmark detector
        landmarks = landmark_detector(img, new_rectangle)

        # Print number of landmarks
        if i == 0:
            print('Number of landmarks', len(landmarks.parts()))

        # Store landmarks for current face
        landmarks_all.append(landmarks)

        # Render outline of the face using detected landmarks
        render_face(img, landmarks)

        # Save the landmarks
        landmarks_filename = f'{landmark_basename}_{i}.txt'
        print('Saving landmarks to', landmarks_filename)
        write_landmarks_to_file(landmarks, landmarks_filename)  # save to disk


def detect_landmark(predictor_path, image_filename, output_path, display=False):
    # Validation checks
    validate_params(predictor_path, image_filename, output_path)

    # Load models and image
    face_detector, landmark_detector, img = load_models_and_image(predictor_path, image_filename)

    # Base name of output file
    landmark_basename = os.path.join(output_path, os.path.splitext(os.path.basename(image_filename))[0])

    if display:
        display_image(img)
    
    # Detect faces
    # 0 means no-upscaling has to be done (for small faces, upscaling is required)
    face_rectangles = face_detector(img, 0)
    print('Number of faces detected:', len(face_rectangles))

    # Detect and store face landmarks
    store_face_landmark(img, landmark_detector, face_rectangles, landmark_basename)

    # Save output image
    output_filename = f'{landmark_basename}_landmarks.jpg'
    print('Saving output image to', output_filename)
    cv2.imwrite(output_filename, img)

    if display:
        display_image(img, figsize=(15, 15), title='Facial Landmark Detector')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predictor', required=True, help='Predictor model')
    parser.add_argument('-i', '--image', required=True, help='Image filename')
    parser.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
        help='Output directory name'
    )
    parser.add_argument('--display', action='store_true', help='Display images')
    args = parser.parse_args()

    detect_landmark(args.predictor, args.image, args.output, args.display)
