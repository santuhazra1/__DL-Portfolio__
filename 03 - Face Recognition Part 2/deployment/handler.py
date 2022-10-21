try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
import torch
from torchvision import transforms
from PIL import Image
from requests_toolbelt.multipart import decoder

from classes import class_names


# Define environment variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'face_recognition.pt'

print('Downloading model...')
s3 = boto3.client('s3')

try:
    if not os.path.isfile(MODEL_PATH):
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print('Creating bytestream...')
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading model...')
        model = torch.jit.load(bytestream)
        print('Model loaded.')
except Exception as e:
    print(repr(e))
    raise(e)


def fetch_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    
    print('Content-Type', content_type_header)
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')
    
    return picture.content


def transform_image(image_bytes):
    """Apply transformations to an input image."""
    try:
        transformations = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=2),
            transforms.ToTensor(),
        ])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)


def get_prediction(image_bytes):
    """Get predictions for an image."""
    tensor = transform_image(image_bytes)
    print('Picture transformed')
    prediction_idx = model(tensor).argmax().item()
    if prediction_idx > 9:
        return prediction_idx, 'Unknown Face'
    return prediction_idx, class_names[prediction_idx]


def recognize_face(event, context):
    """Recognize the Input Image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)

        # Predict face
        prediction_idx, prediction = get_prediction(picture)
        print(f'Prediction: {prediction_idx}\tPrediction Name: {prediction}')

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({
                'result': 'success',
                'data': {
                    'predicted': prediction_idx,
                    'predicted_name': prediction
                }
            })
        }
    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'error': repr(e)})
        }
