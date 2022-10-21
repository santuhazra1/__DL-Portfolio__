# Generative Adversarial Networks

[![Website](https://img.shields.io/badge/Website-green.svg)](http://orionai.s3-website.ap-south-1.amazonaws.com/dcgan)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1j1Y9I9QpVk-kn2os3cKVfk59RKu3XcDy?usp=sharing) -->

The goal of this project is to create an interactive website that generates Indian cars.

All the files and the models have to be deployed to AWS Lambda. The code to deploy them can be found [here](deployment/).

### Parameters and Hyperparameters

- Loss Function: Binary Cross Entropy Loss
- Epochs: 1000
- Optimizer: Adam
- Learning Rate: 0.0002
- Batch Size: 128

## Results

|                               Real cars                               |                          Generated Fake Cars                          |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
| <img src="./images/realCar.jpg" width="300px" alt="centered image" /> | <img src="./images/fakeCar.jpg" width="300px" alt="centered image" /> |

## Some of the Generated Fake Cars

|                                                                    |                                                                    |
| :----------------------------------------------------------------: | :----------------------------------------------------------------: |
| <img src="./images/car1.jpg" width="150px" alt="centered image" /> | <img src="./images/car2.jpg" width="150px" alt="centered image" /> |
| <img src="./images/car3.jpg" width="150px" alt="centered image" /> | <img src="./images/car4.jpg" width="150px" alt="centered image" /> |
|                                                                    |

## Generator and Discriminator Loss During Training

<p align='center'>
    <img src="./images/loss.png" width="350px" alt="centered image" />
</p>

## Dataset Preparation

[![Dataset](https://img.shields.io/badge/Dataset-blue.svg)](https://drive.google.com/file/d/1AGvCVOlW224M8mG8i7IOFmUF0LyZaflK/view?usp=sharing)

For the dataset, we downloaded **Indian car** images from various sources such as Flicker and Google Images. The images of the car are facing the left side.

- Image Size: 64x64x3
- Number of Images: 1124
