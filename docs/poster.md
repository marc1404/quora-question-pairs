# Poster

## Data Analysis
- Total number of question pairs for training: 323164
- Duplicate pairs: 36.88%

## Feature Extraction
- word2vec
- Google News Embedding
- Once we have the vector representations:
  - Find the longest question
  - Use zero padding to normalise question length

## Training & Validation
**Validation Split:**  
We decided to use 90/10 split for validation as the training set contains > 400k entries.  
10% of that yields 40k validation entries already which is enough.

**Loss function:**  
*binary cross-entropy* because the output is a probability between 0 and 1.  
This is a natural fit for our goal as we are only trying to classify between one class; is it a duplicate or not.  
(If there is still time I will try hinge-loss as it can also be used for classification)

**Optimization function:**  
We've compared *Adadelta* and *Adam* as optimisation functions for our task.  
Adadelta was the first choice because it requires no manual tuning of the learning rate and appears to be robust when used in different scenarios.  
Using it for our models had good results but we tried comparing alternatives to see if we could further optimise.  
Adam is similar to Adadelta and also used by many other successful implementations of this task.  
Both functions lead to similar good results in our case.

**Early stopping:**  
We use early stopping on the validation loss metric to avoid overfitting.

## Dense Concatenation
- Evaluated the simplest model we could think of
- Inspired by the blog post of Quora engineers about their first approaches
- Process:
  - Take vector representation of both questions and concatenate them
  - Feed this single representation into a fully connected dense network
- Goal: Network learns what a single network of similar questions looks like
- Best accuracy ~0.7
- Any dropout or regularisation layer decreased performance
- Early stopping on validation accuracy

## Siamese Manhattan LSTM
- Each the embedding layers and the LSTM layers share the same weights
- LSTMs output a 50-dimensional similarity vector
- Manhattan distance is calculated between these vectors
- Early stopping on validation loss

## Interesting Findings
- We tested pre, post and mirrored zero padding for the dense model without noticing any difference in performance
- However, the Sia MaLSTM performs best with pre-zero padding, any other configuration leads to a far worse result
- The simple dense model converges surprisingly fast (even with CPU training)

## Technical Info
We could optionally include this info to show that we invested time in thinking about development and training of our NN.  
The technical training info is a little bit more interesting than the development info.  

Trained in the following environment:
- MSI GeForce GTX970 4G
- 16GB RAM
- Python 3.6
- Keras with TensorFlow 1.4 backend
- NVIDIA CUDA 8

Development:
- Anaconda with Python 3.6
- GitHub's Atom Editor
- Hydrogen plugin for Jupyter kernel features
- Source code on GitHub
