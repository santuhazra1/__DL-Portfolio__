# Neural Embedding

[![Website](https://img.shields.io/badge/Website-blue.svg)](http://orionai.s3-website.ap-south-1.amazonaws.com/sentimentanalysis)

The goal of this project is to perform sentiment analysis on movie reviews i.e. for a given review, the model will predict whether it is a positive review or a negative review. The model is deployed is aws lambda..

## Different Approaches to Sentiment Analysis

We first try out different approaches and then choose the best one among them for deployment. We use this [repository](https://github.com/bentrevett/pytorch-sentiment-analysis) as reference for testing different approaches.

### 1 - Simple Sentiment Analysis

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Kf7ZhReErFUVRJCyKkUeWK_vcz7pSV1j?usp=sharing) -->

This is a simple model where words are encoded as one-hot vector and fed to a single RNN layer. The model does not perform very well and obtains a test accuracy of just **45.94%**.

### 2 - Upgraded Sentiment Analysis

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xv_rI2CRxNHj_l-SzrtD5SJe1wxXdWKT?usp=sharing) -->

This approach encodes the words using 100-dimensional GloVe embeddings and feeds them to a LSTM network. The model obtains a test accuracy of **87.74%**.

### 3 - Faster Sentiment Analysis

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16arM9jpqDPvxeOLUbTwd9CQejHU9Guw_?usp=sharing) -->

In this approach, the following steps are performed:

1. Generate bigrams for each input sentence and append it to the end of the tokenized list.
2. Encode the words using 100-dimensional GloVe embeddings.
3. Use 2D average pooling with a filter size of `sentence_length x 1` on the embedding matrix.
4. Feed the output of the above step to a Fully Connected layer.

The model trains very fast and obtains a test accuracy of **85.24%**.

### 4 - Convolutional Sentiment Analysis

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JCHvU0JOAPjeXfPnwX4obqd2pzJEI2m6?usp=sharing) -->

This approach uses Convolutional Neural Networks (CNNs) on word embeddings. CNNs can help the model to look at bi-grams (a 1x2 filter), tri-grams (a 1x3 filter) and/or n-grams (a 1x*n* filter) within the text. The intuition here is that the appearance of certain bi-grams, tri-grams and n-grams within the review will be a good indication of the final sentiment. The model obtains a test accuracy of **85.69%**.

### 5 - Transformers Sentiment Analysis

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bd2Q8Yhs80IRRKnRs0dQe_2uLP4WFtot?usp=sharing) -->

In this approach, we use a pre-trained transformer model, specifically the BERT (Bidirectional Encoder Representations from Transformers) model to provide embeddings for the text. These embeddings are then fed to a GRU for making predictions. The model obtains a test accuracy of **91.88%**. Since the model is huge in size, it cannot be deployed on AWS lambda.

## Results

We choose the `Upgraded Sentiment Analysis` model for depoyment. The code for deployment can be found [here](deployment).

Model output samples
| Input Text | Prediction |
| :----: | :----: |
| This film is great | positive |
| This film is terrible | negative |
