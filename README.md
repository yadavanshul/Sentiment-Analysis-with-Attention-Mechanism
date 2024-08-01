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
We used the Sentiment140 dataset, which contains 1.6 million tweets labeled as positive or negative sentiment. Due to computational constraints, subsets of the dataset (10% and 50% of each class) were used for training and validation.

## Preprocessing
The preprocessing steps included:
1. **Loading the dataset**: Import the data and load it into a pandas DataFrame.
2. **Sampling the data**: To manage dataset size, subsets of each class were sampled (10% and 50%).
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

### Regularization
Regularization techniques such as L2 regularization and dropout were applied to prevent overfitting.

### Adjusted Learning Rate
The learning rate was adjusted to 0.0001 for better convergence.

### Simplified Model
A simpler version of the model without attention was also trained to compare performance.

## Results
The following results were obtained using 10% of the data:

### Without Regularization
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 56.27%   | 0.6622 | 76.16%       | 0.4998   |
| 2     | 77.96%   | 0.4747 | 76.40%       | 0.4870   |
| 3     | 79.67%   | 0.4392 | 76.29%       | 0.4980   |
| 4     | 81.39%   | 0.4035 | 75.93%       | 0.5182   |
| 5     | 83.04%   | 0.3668 | 75.57%       | 0.5575   |
| 6     | 84.61%   | 0.3332 | 74.88%       | 0.6264   |
| 7     | 86.10%   | 0.3017 | 74.44%       | 0.7153   |
| 8     | 87.39%   | 0.2731 | 73.82%       | 0.7880   |
| 9     | 88.72%   | 0.2476 | 73.85%       | 0.8964   |
| 10    | 89.94%   | 0.2242 | 73.00%       | 0.9659   |

### With Regularization
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 50.13%   | 1.2773 | 49.88%       | 0.6932   |
| 2     | 50.12%   | 0.6932 | 50.12%       | 0.6932   |
| 3     | 50.11%   | 0.6932 | 50.12%       | 0.6931   |
| 4     | 49.88%   | 0.6932 | 50.12%       | 0.6931   |
| 5     | 50.07%   | 0.6932 | 49.88%       | 0.6932   |
| 6     | 50.20%   | 0.6932 | 49.88%       | 0.6932   |

### Adjusted Learning Rate
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 50.52%   | 0.6929 | 71.83%       | 0.5785   |
| 2     | 73.81%   | 0.5490 | 75.49%       | 0.5108   |
| 3     | 76.85%   | 0.4930 | 75.91%       | 0.5028   |
| 4     | 78.25%   | 0.4720 | 74.07%       | 0.5108   |
| 5     | 78.91%   | 0.4605 | 76.27%       | 0.4933   |
| 6     | 79.36%   | 0.4539 | 75.13%       | 0.5049   |
| 7     | 79.29%   | 0.4529 | 76.22%       | 0.4956   |
| 8     | 79.66%   | 0.4442 | 76.10%       | 0.4976   |

The following results were obtained using 50% of the data:

### Training without Regularization
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 71.04%   | 0.5385 | 78.34%       | 0.4551   |
| 2     | 79.09%   | 0.4481 | 78.46%       | 0.4537   |
| 3     | 80.04%   | 0.4290 | 78.62%       | 0.4534   |
| 4     | 81.04%   | 0.4103 | 78.33%       | 0.4607   |
| 5     | 82.01%   | 0.3909 | 78.15%       | 0.4754   |
| 6     | 83.12%   | 0.3699 | 77.71%       | 0.4953   |
| 7     | 84.23%   | 0.3467 | 77.36%       | 0.5219   |
| 8     | 85.31%   | 0.3245 | 77.09%       | 0.5651   |
| 9     | 86.36%   | 0.3011 | 76.57%       | 0.6041   |
| 10    | 87.32%   | 0.2804 | 76.13%       | 0.6562   |

### Adjusting Learning Rate and Simplifying Model
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 63.40%   | 0.6123 | 76.14%       | 0.4902   |
| 2     | 77.56%   | 0.4801 | 77.73%       | 0.4691   |
| 3     | 77.99%   | 0.4700 | 77.70%       | 0.4660   |
| 4     | 78.18%   | 0.4635 | 77.82%       | 0.4711   |
| 5     | 78.54%   | 0.4578 | 77.49%       | 0.4660   |
| 6     | 78.78%   | 0.4519 | 78.10%       | 0.4608   |
| 7     | 78.96%   | 0.4462 | 77.96%       | 0.4630   |
| 8     | 79.17%   | 0.4423 | 78.08%       | 0.4645   |
| 9     | 79.23%   | 0.4382 | 78.04%       | 0.4657   |

### Adding Batch Normalization
| Epoch | Accuracy | Loss   | Val Accuracy | Val Loss |
|-------|----------|--------|--------------|----------|
| 1     | 70.88%   | 0.5753 | 62.38%       | 0.8090   |
| 2     | 77.49%   | 0.4923 | 77.54%       | 0.4806   |
| 3     | 77.72%   | 0.4843 | 77.61%       | 0.4852   |
| 4     | 77.93%   | 0.4776 | 75.64%       | 0.4992   |
| 5     | 78.21%   | 0.4703 | 76.71%       | 0.4868   |
| 6     | 78.43%   | 0.4655 | 77.13%       | 0.4827   |
| 7     | 78.73%   | 0.4600 | 77.89%       | 0.4742   |
| 8     | 78.94%   | 0.4554 | 71.72%       | 0.5589   |
| 9     | 79.04%   | 0.4531 | 77.29%       | 0.4811   |
| 10    | 79.17%   | 0.4497 | 77.14%       | 0.4891   |

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-attention.git
   cd sentiment-analysis-attention
