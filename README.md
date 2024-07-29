# Sentiment Analysis with Attention Mechanism

This repository contains the code for a sentiment analysis model that uses an attention mechanism to improve the performance and interpretability of predictions. The project aims to classify text data into positive or negative sentiment using a bidirectional LSTM with an attention layer.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Sentiment analysis is a crucial task in natural language processing (NLP) that involves determining the sentiment expressed in a piece of text. This project leverages deep learning techniques, specifically LSTM and attention mechanisms, to enhance sentiment classification accuracy and interpretability.

## Dataset
We used the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive or negative sentiment. Due to computational constraints, subsets of the dataset (25% and 50% of each class) were used for training and validation.

## Preprocessing
The preprocessing steps included:
1. **Loading the dataset**: Import the data and load it into a pandas DataFrame.
2. **Sampling the data**: To manage dataset size, subsets of each class were sampled (25% and 50%).
3. **Cleaning the text**: Remove URLs, special characters, and convert text to lowercase.
4. **Tokenization**: Convert text into sequences of tokens.
5. **Padding**: Ensure all sequences have uniform length by padding shorter sequences.

## Model Architecture
The model consists of:
- **Embedding Layer**: Converts input sequences into dense vectors of fixed size.
- **Bidirectional LSTM Layer**: Processes the sequence of embeddings.
- **Attention Layer**: Computes attention weights and generates a context vector.
- **Dense Layers**: Perform the final classification.

## Training
The model was trained using the Adam optimizer with binary cross-entropy loss. Early stopping and dropout were used to prevent overfitting. The training process included monitoring validation accuracy and loss.

### Adjusted Learning Rate and Regularization
The learning rate was adjusted to 0.0001, and L2 regularization was applied to the LSTM and Dense layers. Early stopping was used to prevent overfitting, with a patience of 5 epochs.

### Simplified Model
A simpler version of the model without attention was also trained to compare performance.

## Results
Both models showed improvement in training and validation accuracy. The model with the adjusted learning rate and regularization achieved higher validation accuracy and lower loss, indicating better generalization.

### Training and Validation Accuracy for 25% Data
![Training and Validation Accuracy for 25% Data](images/accuracy_25.png)

### Training and Validation Loss for 25% Data
![Training and Validation Loss for 25% Data](images/loss_25.png)

### Training and Validation Accuracy for 50% Data
![Training and Validation Accuracy for 50% Data](images/accuracy_50.png)

### Training and Validation Loss for 50% Data
![Training and Validation Loss for 50% Data](images/loss_50.png)

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-attention.git
   cd sentiment-analysis-attention
