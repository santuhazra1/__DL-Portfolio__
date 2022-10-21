try:
    import unzip_requirements
except ImportError:
    pass

import io
import json
import random
import numpy as np
import base64
import boto3
import torch
from PIL import Image
from requests_toolbelt.multipart import decoder


MODEL_PATH = 'gan.pt'


def generate(event, context):
    try:
        # Generate Image
        model = torch.jit.load(MODEL_PATH)
        fixed_noise = torch.randn(64, 100, 1, 1, device='cpu')
        with torch.no_grad():
            fake = model(fixed_noise).detach().cpu()
        generated_image = random.choice(fake).permute(1, 2, 0).numpy().copy() * 255
        generated_image = Image.fromarray(generated_image.astype(np.uint8))

        print('Loading output to buffer')
        buffer = io.BytesIO()
        generated_image.save(buffer, format="JPEG")
        generated_image_bytes = base64.b64encode(buffer.getvalue())

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'data': generated_image_bytes.decode('ascii')})
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
