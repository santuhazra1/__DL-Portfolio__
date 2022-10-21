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

from face_swap import swap_image


# Define environment variables
MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'


def fetch_input_images(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final picture that will be used by the model
    decoded = decoder.MultipartDecoder(body, content_type_header).parts
    picture_source, picture_target = decoded[0], decoded[1]
    print('Picture obtained')
    
    return picture_source.content, picture_target.content


def swap_face(event, context):
    """Swap Faces."""
    try:
        # Get images from the request
        picture_source, picture_target = fetch_input_images(event)

        # Swap images
        status, output = swap_image(picture_source, picture_target, MODEL_PATH)
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
