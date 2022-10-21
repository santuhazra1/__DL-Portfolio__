try:
    import unzip_requirements
except ImportError:
    pass

import os
import json
import base64
from requests_toolbelt.multipart import decoder

from asr import predict_text

# Define global variables
MODEL_PATH = 'model.pt'


def fetch_input(event):
    print('Fetching Content-Type')
    if 'Content-Type' in event['headers']:
        content_type_header = event['headers']['Content-Type']
    else:
        content_type_header = event['headers']['content-type']
    print('Loading body...')
    body = base64.b64decode(event['body'])
    print('Body loaded')

    # Obtain the final input that will be used by the model
    audio = decoder.MultipartDecoder(body, content_type_header).parts[0]
    print('Audio obtained')
    
    return audio.content


def stt(event, context):
    try:
        # Get audio from the request
        audio_file = '/tmp/temp.wav'
        audio = fetch_input(event)
        print('Type:', type(audio))

        # Do some pre-processing
        with open(audio_file, 'wb') as f:
            f.write(audio)
        print('File written')

        # Get Text
        print('Getting text')
        output = predict_text(audio_file, MODEL_PATH)
        print('Text:', output)

        # Remove file
        os.remove(audio_file)
        print('processed file removed')

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
