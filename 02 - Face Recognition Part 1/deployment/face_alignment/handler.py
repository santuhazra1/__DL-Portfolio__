try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
from requests_toolbelt.multipart import decoder

from face_alignment import align_image


# Define environment variables
MODEL_PATH = 'shape_predictor_5_face_landmarks.dat'


def fetch_input_image(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Picture obtained')
    
    return picture.content


def align_face(event, context):
    """Align the Input Image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)

        # Align image
        status, output = align_image(MODEL_PATH, picture)
        print('status', status)
        if status == 'success':
            buffer = io.BytesIO()
            output.save(buffer, format="JPEG")
            output_bytes = base64.b64encode(buffer.getvalue())

            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': json.dumps({'result': status, 'data': output_bytes.decode('ascii')})
            }
        else:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': json.dumps({'result': status, 'data': output})
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
