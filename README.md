# Movie Review Sentiment Analysis
## Sentiment analysis of Movie Reviews
It is a natural language processing problem where text is understood and the underlying intent is predicted. Here, you need to  predict the sentiment of movie reviews as either positive or negative in Python using the Keras deep learning library.

## Data description
The dataset is the Large Movie Review Dataset often referred to as the IMDB dataset.

The [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) (often referred to as the IMDB dataset) contains 25,000 highly polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given moving review has a positive or negative sentiment.  Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers).

## Multi-hot encode the input data

All sequences are of different length and our vocabulory size is 10K.  
1) Intialize vector of dimension 10,000 with value 0. 
2) For those tokens in a sequence which are present in Vocabulary make that position as 1 and keep all other positions filled with 0. <br>
For example, lets take Vocabulary = ['I': 0, ':1, 'eat: 2:' mango: 3, 'fruit':4, 'happy':5, 'you':6] 
We have two sequnces and Multi-hot encoding of both sequences will be of dimension:  7 (vocab size).
1) *Mango is my favourite fruit* becomes *Mango ? ? ? fruit* after removing words which are not in my vocabulary. Hence multi hot encoding will have two 1's corresponding to mango and fruit i.e, [0, 0, 0, 1, 1, 0, 0] <br>
Similarly, 2) *I love to eat mango*  = *I ? ? eat mango*  =  [1, 1, 0, 1, 0, 0, 0]

