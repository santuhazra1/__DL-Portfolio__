import os
import argparse
import cv2
import numpy as np

import face_blend_common as fbc
from face_landmark_detection import load_models_and_image, validate_params, display_image


def align_face(img, points, output_dim):
    print('Aligning Image')
    img_norm, points = fbc.normalizeImagesAndLandmarks(output_dim, img, points)
    img_norm = np.uint8(img_norm * 255)
    return img_norm


def main(predictor_path, image_filename, output_dim, output_path, display=False):
    # Validation checks
    validate_params(predictor_path, image_filename, output_path)

    # Load models and image
    face_detector, landmark_detector, img = load_models_and_image(predictor_path, image_filename)

    if display:
        display_image(img)

    # Detect landmarks
    points = np.array(fbc.getLandmarks(face_detector, landmark_detector, img))

    # Convert image to floating point in the range 0 to 1
    img = np.float32(img) / 255.0

    # Align image
    img_align = align_face(img, points, output_dim)

    # Save image
    output_filename = os.path.join(
        output_path,
        '_aligned'.join(os.path.splitext(os.path.basename(image_filename)))
    )
    cv2.imwrite(output_filename, img_align)
    print('Output image saved to', output_filename)

    if display:
        display_image(img_align, title='Aligned Image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predictor', required=True, help='Predictor model')
    parser.add_argument('-i', '--image', required=True, help='Image filename')
    parser.add_argument(
        '--output',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
        help='Output directory name'
    )
    parser.add_argument('--height', type=int, default=600, help='Output image height')
    parser.add_argument('--width', type=int, default=600, help='Output image width')
    parser.add_argument('--display', action='store_true', help='Display images')
    args = parser.parse_args()

    main(args.predictor, args.image, (args.height, args.width), args.output, args.display)
