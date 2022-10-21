try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import base64
import boto3
from requests_toolbelt.multipart import decoder

from sentiment_analysis import predict_sentiment

MODEL_PATH = 'sentiment_analysis.pt'
METADATA_PATH = 'sentiment_analysis_metadata.pkl'


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


def sentiment(event, context):
    """Style the content image."""
    try:
        input_text = fetch_inputs(event)
        print(input_text)

        # Output Sentiment
        output = predict_sentiment(input_text, MODEL_PATH, METADATA_PATH)
        print(output)
        if output == 'pos':
            final_output = 'Positive'
        elif output == 'neg':
            final_output = 'Negative'

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': final_output})
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
