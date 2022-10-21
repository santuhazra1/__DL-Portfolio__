try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import base64
import boto3
from PIL import Image
from requests_toolbelt.multipart import decoder

from super_resolution import upscale


MODEL_PATH = 'srgan.pt'


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
    
    return Image.open(io.BytesIO(picture.content))


def srgan(event, context):
    """Super Resolution"""
    try:
        # Get image from the request
        image = fetch_input_image(event)

        # Upscale the image
        output = upscale(image, MODEL_PATH)

        # Convert output to bytes
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
            'body': json.dumps({'data': output_bytes.decode('ascii')})
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
