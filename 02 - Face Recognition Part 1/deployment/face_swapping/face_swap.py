import cv2
import dlib
import numpy as np
from PIL import Image

import face_blend_common as fbc


def load_image(image_bytes):
    print('Reading image')
    img_display = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    img = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
    return img, img_display


def load_models(predictor_path):
    print('Initializing the face detector instance')
    detector = dlib.get_frontal_face_detector()

    print('Loading the shape predictor model')
    predictor = dlib.shape_predictor(predictor_path)

    return detector, predictor


def get_convex_hull(hull_index, points_source, points_target):
    print('Creating convex hull lists')
    hull_source = []
    hull_target = []
    for i in range(0, len(hull_index)):
        hull_source.append(points_source[hull_index[i][0]])
        hull_target.append(points_target[hull_index[i][0]])
    
    return hull_source, hull_target


def find_centroid(img_target, hull_target):
    print('Calculating mask for seamless cloning')
    hull8U = []
    for i in range(0, len(hull_target)):
        hull8U.append((hull_target[i][0], hull_target[i][1]))

    mask = np.zeros(img_target.shape, dtype=img_target.dtype) 
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    print('Finding centroid')
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    return center, mask


def calculate_triangulation(img_target, hull_source, hull_target, img_source_display, img_target_display):
    print('Finding Delaunay traingulation for convex hull points')
    img_target_size = img_target.shape    
    rect = (0, 0, img_target_size[1], img_target_size[0])

    dt = fbc.calculateDelaunayTriangles(rect, hull_target)

    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        quit()
    
    print('Applying triangulation to images')
    img_source_temp = img_source_display.copy()
    img_target_temp = img_target_display.copy()

    tris_source = []
    tris_target = []
    for i in range(0, len(dt)):
        tri_source = []
        tri_target = []
        for j in range(0, 3):
            tri_source.append(hull_source[dt[i][j]])
            tri_target.append(hull_target[dt[i][j]])

        tris_source.append(tri_source)
        tris_target.append(tri_target)

    cv2.polylines(img_source_temp, np.array(tris_source), True, (0, 0, 255), 2)
    cv2.polylines(img_target_temp, np.array(tris_target), True, (0, 0, 255), 2)

    return tris_source, tris_target


def swap_image(img_source_bytes, img_target_bytes, predictor_path):
    detector, predictor = load_models(predictor_path)

    img_source, img_source_display = load_image(img_source_bytes)
    img_target, img_target_display = load_image(img_target_bytes)
    img_source_warped = np.copy(img_target)

    # Read array of corresponding points
    status, results_source = fbc.getLandmarks(detector, predictor, img_source)
    if status == 'fail':
        return status, results_source
    points_source = results_source

    status, results_target = fbc.getLandmarks(detector, predictor, img_target)
    if status == 'fail':
        return status, results_target
    points_target = results_target

    # Convex hull
    hull_index = cv2.convexHull(np.array(points_target), returnPoints=False)
    hull_source, hull_target = get_convex_hull(hull_index, points_source, points_target)
    
    # Calculate Mask and Find Centroid
    center, mask = find_centroid(img_target, hull_target)

    # Calculate triangulation
    tris_source, tris_target = calculate_triangulation(
        img_target, hull_source, hull_target, img_source_display, img_target_display
    )

    # Simple Alpha Blending
    print('Applying affine transformation to Delaunay triangles')
    for i in range(0, len(tris_source)):
        fbc.warpTriangle(img_source, img_source_warped, tris_source[i], tris_target[i])

    print('Cloning seamlessly')
    output = cv2.seamlessClone(np.uint8(img_source_warped), img_target, mask, center, cv2.NORMAL_CLONE)

    return status, Image.fromarray(output)
