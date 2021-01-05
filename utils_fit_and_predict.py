import datetime
import numpy as np
import tensorflow as tf
import pandas as pd

from keras import layers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D

from keras.layers.core import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from psycopg2.extras import execute_values
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


np.random.seed(42)


def get_model(maxlen, params):
    '''
    Define the model according to its parameters
    Args:
    - maxlen
    - params - namedtuple object
    '''
    model = Sequential()
    model.add(Embedding(params.vocab_size+1, params.embed_size, input_length=maxlen))
    model.add(SpatialDropout1D(0.2)) 
    model.add(Conv1D(filters=params.n_filters, kernel_size=params.n_words, activation="relu"))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])
    return model


def fit_model(prepped_data, params, domain, model):
    
    mcp_save_name = (f'{str(datetime.datetime.now().strftime("%Y%m%d"))}'
                     f'{domain} ALL PRINCIPLE_PROB CNN v{params.vocab_size} em{params.embed_size}'
                     f'{params.n_filters} w{params.n_words} b{params.batch_size} ep{params.n_epochs} '
                     f'using {prepped_data.word2index_Name}.hdf5')
    print(mcp_save_name)
    mcp_save = ModelCheckpoint(mcp_save_name, save_best_only=True, monitor='val_accuracy', mode='max')
    
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='max')
    
    history = model.fit(prepped_data.X_train_numbers,
                    prepped_data.y_train,
                    batch_size=params.batch_size,
                    epochs=params.n_epochs,
                    callbacks=[earlyStopping, mcp_save],
                    validation_data = (prepped_data.X_test_numbers, prepped_data.y_test)
                    )    
    return model, history, mcp_save_name


def get_predictions(mcp_save_Name, X_test_numbers):
    loaded_model = tf.keras.models.load_model(mcp_save_Name)
    predictions = pd.DataFrame(loaded_model.predict([X_test_numbers]))
    predictions.columns = ['probN','probY']
    return predictions


def make_results_df(test, predictions, domain, probY_threshold):
    results = pd.merge(test.reset_index(drop=True), predictions, left_index=True, right_index=True)
    results.rename(columns={domain:'actual'}, inplace=True)
    print(results.head())
    results['predicted'] = 'n'
    results['predicted'][results['probY'] >= probY_threshold] = 'y'
    results['actual'][results['actual'] == 0] = 'n'
    results['actual'][results['actual'] != 'n'] = 'y'
    results = results[['commentID','comment','actual','predicted','probN','probY']]
    return results


def fit_make_crosstab(prepped_data, model, params, domain, probY_threshold):
    model, history, mcp_save_name = fit_model(prepped_data, params, domain, model)
    predictions = get_predictions(mcp_save_name, prepped_data.X_test_numbers)
    results = make_results_df(prepped_data.test, predictions, domain, probY_threshold)
    print('\nCross-tab for validation data (actual coding reads across, predictions downwards)')
    print('* accuracy = '+str((len(results) - len(results[results['actual'] != results['predicted']]))/len(results)*100)[:5] + '% *')
    print(results.pivot_table(index=['actual'],columns=['predicted'], values='comment', aggfunc=[len])            )
    return predictions, results
