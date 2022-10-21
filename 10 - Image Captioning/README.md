# Image Captioning

[![Website](https://img.shields.io/badge/Website-blue.svg)](http://orionai.s3-website.ap-south-1.amazonaws.com/imagecaptioning)


The goal of this project is to train and deploy an image caption generation model. The code for deployment can be found [here](deployment).

### Parameters and Hyperparameters

- Loss Function: NLLLoss
- Bleu Score: 14.0
- Epochs: 120
- Encoder: Pre-trained ResNet-18 on ImageNet dataset
- Decoder Learning Rate: 4e-4
- Optimizer: Adam
- Batch Size: 32
- Embedding dimension: 128
- Attention dimension: 128
- Decoder dimension: 128
- Dropout: 0.5

## Results

|                             Input Image                              |                    Output Caption                    |
| :------------------------------------------------------------------: | :--------------------------------------------------: |
| <img src="./images/input1.jpg" width="300px" alt="centered image" /> |     a man in a wetsuit is surfing on a surfboard     |
| <img src="./images/input2.jpg" width="300px" alt="centered image" /> |      a group of people sit on a snowy mountain       |
| <img src="./images/input3.jpg" width="300px" alt="centered image" /> | a young boy in a red shirt is riding on a tire swing |

## Architecture

We used an Encoder-Decoder architecture. The encoder is 18-layered Residual Network pre-trained on the ImageNet classification task and the layers are not fine tuned. Decoder has attention incorporated into it as it will help to look at different parts of the image to generate prediction for the sequence.

<p align='center'>
    <img src="./images/architecture.JPG" width="430px" alt="architecture" />  
</p>

## Attention Mechanism

A neural network is considered to be an effort to mimic human brain actions in a simplified manner. Attention Mechanism is also an attempt to implement the same action of selectively concentrating on a few relevant things, while ignoring others in deep neural networks.

An attention mechanism allows the model to focus on the currently most relevant part of the source sentence. In this project we implemented additive attention that was used in Bahdanau et al.

<p align='center'>
    <img src="./images/attention.jpeg" width="430px" alt="attention mechanism" />  
</p>

The code present here has been referenced from [this](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) repository.
