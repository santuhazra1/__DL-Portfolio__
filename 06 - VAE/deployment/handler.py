try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import base64
import boto3
from PIL import Image
from requests_toolbelt.multipart import decoder

from model import Model

MODEL_PATH = 'vae.onnx'


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


def reconstruct(event, context):
    """Reconstruct the Input Image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)
        image = Image.open(io.BytesIO(picture))

        print('Loading model')
        model = Model(MODEL_PATH)

        # Reconstruct image
        output = model(image)

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
