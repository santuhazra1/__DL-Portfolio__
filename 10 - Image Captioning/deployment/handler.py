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

from image_captioning import caption_image


MODEL_PATH = 'model.pth.tar'
WORDMAP_PATH = 'wordmap.json'


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


def caption(event, context):
    """Caption input image."""
    try:
        # Get image from the request
        picture = fetch_input_image(event)
        image = Image.open(io.BytesIO(picture))

        # Get Caption
        print('Getting caption')
        output = caption_image(image, MODEL_PATH, WORDMAP_PATH)
        print('Caption:', output)

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': output})
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
