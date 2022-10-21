try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import base64
import boto3
import torch

print('torch version:',torch.__version__)

from requests_toolbelt.multipart import decoder

from german_to_english import translate_sentence

MODEL_PATH = 'model.pt'
METADATA_PATH = 'model_metadata.pkl'


def fetch_inputs(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final input that will be used by the model
    input_text = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Input obtained')
    
    return input_text.content.decode('utf-8')


def translate(event, context):
    """Style the content image."""
    try:
        input_text = fetch_inputs(event)
        print(input_text)

        # Output Sentiment
        output = translate_sentence(input_text, MODEL_PATH, METADATA_PATH)
        print(output)

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
