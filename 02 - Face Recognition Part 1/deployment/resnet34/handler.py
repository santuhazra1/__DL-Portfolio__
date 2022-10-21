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
import torchvision
from torchvision import transforms
from PIL import Image
from requests_toolbelt.multipart import decoder

from imagenet_classes import imagenet_class_names


# Define environment variables
S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4'
MODEL_PATH = os.environ['MODEL_PATH'] if 'MODEL_PATH' in os.environ else 'resnet34.pt'

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


def transform_image(image_bytes):
    """Apply transformations to an input image."""
    try:
        transformations = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    return model(tensor).argmax().item()


def classify_image(event, context):
    """Take input image from API and classify it."""
    try:
        # Get image from the request
        if 'Content-Type' in event['headers']:
            content_type_header = event['headers']['Content-Type']
        else:
            content_type_header = event['headers']['content-type']
        body = base64.b64decode(event['body'])
        print('Body loaded')

        # Obtain the final picture that will be used by the model
        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        print('Picture obtained')

        # Get predictions
        prediction = get_prediction(image_bytes=picture.content)
        prediction_name = imagenet_class_names[prediction]
        print(f'Prediction: {prediction}\tPrediction Name: {prediction_name}')

        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({
                'file': filename.replace('"', ''),
                'predicted': prediction,
                'predicted_name': prediction_name,
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
