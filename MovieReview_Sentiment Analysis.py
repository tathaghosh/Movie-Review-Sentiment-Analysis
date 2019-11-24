
## Loading dataset
import numpy as np
from matplotlib import pyplot
import numpy
import keras
from keras import regularizers,layers
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# np.load is used inside imdb.load_data. But imdb.load_data still assumes the default 
# values of an older version of numpy. So necessary changes to np.load are made

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load Numpy
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print(X.shape)
print(X_train.shape)

"""## **Example reviews.**"""

word_to_id = keras.datasets.imdb.get_word_index()
id_to_word = {value:key for key,value in word_to_id.items()}
for i in range(15,20):
  print("********************************************")
  print(' '.join(id_to_word.get(id - 3, '?')for id in X_train[i] ))

"""## Summarize the data
1) Find out the number of classes in label (*y* array)? <br>
2) Find out number of unique words in dataset *X*?  <br>
3) Calculate the list of review length , report mean and standard deviation. <br>
"""

def summarize_data():
  """
  Output:
                    classes: list, list of unique classes in y  
                no_of_words: int, number of unique words in dataset x 
     list_of_review_lengths: list,  list of lengths of each review 
         mean_review_length: float, mean(list_of_review_lengths), a single floating point value
          std_review_length: float, standard_deviation(list_of_review_lengths), a single floating point value
  """
  
  classes = np.unique(y_train)
  words_set=set();
  list_of_review_lengths=[]
  for review in X:
    list_of_review_lengths.append(len(review))
    words_set.update(review)
  no_of_words=len(words_set)
  mean_review_length = np.mean(list_of_review_lengths)
  std_review_length = np.std(list_of_review_lengths)
  return classes, no_of_words, list_of_review_lengths, mean_review_length, std_review_length


classes, no_of_words, list_of_review_lengths, mean_review_length, std_review_length = summarize_data()

'''test for summarize_data'''
def test_summarize_data():
  assert classes.tolist() == [0,1]
  assert no_of_words == 9998
  assert np.isclose(mean_review_length, 234.75892, atol = 0.001)
  assert np.isclose(std_review_length, 172.91149458735703, atol = 0.001)
  print('Test passed', '\U0001F44D')
test_summarize_data()

type(y_train)

"""## One hot encode the output data"""

def one_hot(y):
  """
  Inputs:
    y: numpy array with class labels
  Outputs:
    y_oh: numpy array with corresponding one-hot encodings
  """
  
  y_oh = np.zeros((y.shape[0], classes.shape[0]))
  for i in range(y.shape[0]):
    y_oh[i,y[i]]=1
  return y_oh
y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Multi-hot encode the input data

def multi_hot_encode(sequences, dimension):
  """
    Input:
          sequences: list of sequences in X_train or X_test

    Output:
          results: mult numpy matrix of shape(len(sequences), dimension)
                  
  """
  
  results = np.zeros((len(sequences), dimension))
  for review_no in range(len(sequences)):
    for word in sequences[review_no]:
      results[review_no, word] = 1
  return results

x_train = multi_hot_encode(X_train, 10000)
x_test = multi_hot_encode(X_test, 10000)

print("x_train ", x_train.shape)
print("x_test ", x_test.shape)

'''test for pad_sequences'''
def test_multi_hot_encode():
  assert np.sum(x_train[1]) == 121.0
  print('Test passed', '\U0001F44D')
test_multi_hot_encode()

"""## Split the data into train and validation"""

from sklearn.model_selection import train_test_split
x_strat, x_dev, y_strat, y_dev = train_test_split(x_train, y_train,test_size=0.40,random_state=0, stratify=y_train)

"""## Build Model
Build a multi layered feed forward network in keras.

### Create the model
"""

def create_model():
    """
    Output:
        model: A compiled keras model
    """
    
    from keras.layers import Dense
    from keras.models import Model
    from keras.optimizers import Adam
    model = keras.Sequential()
    model.add(Dense(100, activation="relu", input_dim=x_train.shape[1]))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    opt=Adam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
    return model
  
model = create_model()
print(model.summary())

"""### Fit the Model"""

import matplotlib.pyplot as plt
def fit(model):
    """
    Action:
        Fit the model created above using training data as x_strat and y_strat
        and validation_data as x_dev and y_dev, verbose=2 and store it in 'history' variable.
        
        evaluate the model using x_test, y_test, verbose=0 and store it in 'scores' list
    Output:
        scores: list of length 2
        history_dict: output of history.history where history is output of model.fit()
    """
    
    history = model.fit(x_strat, y_strat, verbose=2, validation_data=(x_dev, y_dev), epochs=20, batch_size=20)
    scores = model.evaluate(x_test, y_test, verbose=0)
    history_dict = history.history

    return scores,history_dict
    
scores,history_dict = fit(model)

Accuracy=scores[1]*100
print('Accuracy of your model is')
print(scores[1]*100)

history_dict['loss']

# Verify whether training in converged or not

import matplotlib.pyplot as plt
plt.clf()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, (len(history_dict['acc']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""### Advanced Testing
1. Some reviews where the model fails to predict the sentiment correctly.
2. 5 reviews with at least 20 words to see if the model correctly predicts the sentiment on these reviews
"""

m=0
n=50
pred = model.predict(x_test[m:n])
pred_class = [np.argmax(i) for i in pred]
true = y_test[m:n]
true_class = [np.argmax(i) for i in true]
print("Some wrongly classified reviews are:\n")
for i in range(m,n):
  if true_class[i] != pred_class[i]:
    print(' '.join(id_to_word.get(id - 3, '')for id in X_test[i] ))
    print('')

eng_reviews = [
    'this was one of the best movie i have ever watched i recommend that everyone must watch this at least once',
    'most pathetic movie of the century do not bother to waste your time or money to watch this piece of garbage',
    'useless nonsense junk',
    'most fantastic film of this year',
    'this movie is terribly fantastic',
]
eng_reviews_split = [review.split(' ') for review in eng_reviews]
num_review=[]
for review in eng_reviews_split:
  x=[int(word_to_id.get(word, -2))+3 for word in review]
  x.insert(0,1)
  num_review.append(x)
mh_reviews = multi_hot_encode(num_review, 10000)

pred=model.predict(mh_reviews)
pred_class = [np.argmax(i) for i in pred]
for i in pred_class:
  if i==1:
    print("Positive review")
  else:
    print("Negative review")
