from keras import layers
from keras.models import  Sequential
from keras.models import Model

from keras.layers import Embedding,Conv1D,MaxPooling1D,Flatten,Dense,Input, GlobalMaxPooling1D
import  numpy as np
import pandas as pd
from keras.utils import  np_utils
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

def load_train_test_data(train_name='../data/train.csv',test_name='../data/test.csv'):
    dt = pd.read_csv(train_name)
    dt = dt.sample(frac=1)
    train_len = dt.shape[0]
    dt2 = pd.read_csv(test_name)
    # dt2 = dt2.sample(frac=1)
    print dt.iloc[:, 1].shape,dt2.iloc[:,1].shape
    pd.concat([dt.iloc[:, 1], dt2.iloc[:, 1]], axis=0)
    texts = pd.concat([dt.iloc[:, 1],dt2.iloc[:,1]],axis=0).values
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    train_X = data[0:train_len]
    test_X = data[train_len:]
    train_Y = dt.iloc[:, 2:]
    return train_X,train_Y,test_X,tokenizer.word_index

def load_data(train_name='../data/train.csv'):
    dt = pd.read_csv(train_name)
    dt = dt.sample(frac=1)
    data_len = dt.shape[0]
    texts =dt.iloc[:, 1].values

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    dev_X = data[0:data_len/10]
    dev_Y = dt.iloc[0:data_len/10,2:]
    train_X= data[data_len/10:]
    train_Y = dt.iloc[data_len/10:,2:]
    return train_X,train_Y,dev_X,dev_Y,tokenizer.word_index

MAX_NB_WORDS = 1000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
labels_index = 6
# train_X,train_Y,dev_X,dev_Y,word_index = load_data('../data/train.csv')
train_X,train_Y,test_X,word_index = load_train_test_data()

# print dev_Y.shape
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(35)(x)  # global max pooling
x = GlobalMaxPooling1D()(x)
# x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(6, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(train_X, train_Y.values#, validation_data=(dev_X, dev_Y.values)
          ,epochs=2, batch_size=128)

pred_y = model.predict(test_X,batch_size=128)
df = pd.DataFrame(pred_y)
df.to_csv('toxic_cnn1d.csv',index=False,sep=',')
#
# labels = np_utils.to_categorical(np.asarray(labels))
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

# # split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
# x_train = data[:-nb_validation_samples]
# y_train = labels[:-nb_validation_samples]
# x_val = data[-nb_validation_samples:]
# y_val = labels[-nb_validation_samples:]

#
# model = Sequential()
# model.add(Embedding(1000, 64, input_length=10))
# # the model will take as input an integer matrix of size (batch, input_length).
# # the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# # now model.output_shape == (None, 10, 64), where None is the batch dimension.
#
# input_array = np.random.randint(1000, size=(32, 10))
#
# model.compile('rmsprop', 'mse')
# output_array = model.predict(input_array)
# assert output_array.shape == (32, 10, 64)