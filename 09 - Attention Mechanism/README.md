# GRU, Attention Mechanism & Transformers

[![Website](https://img.shields.io/badge/Website-blue.svg)](http://orionai.s3-website.ap-south-1.amazonaws.com/machinetranslation)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ogPvV4JI3Gr0-kWsjysmoDsZLuRtTbU-?usp=sharing) -->

The goal of this project is to train and deploy a model which can translate sentence from German to English.

All the code for deployment can be found in the [deployment](deployment) directory.

### Parameters and Hyperparameters

- Loss Function: NLLLoss
- Epochs: 10
- Learning Rate: 0.0003
- Optimizer: Adam
- Batch Size: 320

## Results

Since the model was trained on a very small dataset (due to limitations on google colab), the results are not very accurate.

### Input Sentence

```
jeden tag nahmen wir einen anderen weg, sodass niemand erraten konnte, wohin wir gingen.
```

### Output Sentence

```
and every day every day , we took another one that could n't go anywhere . where we went .
```

## Attention Mechanism

A neural network is considered to be an effort to mimic human brain actions in a simplified manner. Attention Mechanism is also an attempt to implement the same action of selectively concentrating on a few relevant things, while ignoring others in deep neural networks.

An attention mechanism allows the model to focus on the currently most relevant part of the source sentence. In this project we implemented additive attention that was used in _Bahdanau et al._

<p align='center'>
    <img src="./images/attention.jpeg" width="430px" alt="attention mechanism" />  
</p>
