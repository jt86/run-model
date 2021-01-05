##############################################################################
# Import the necessary libraries

import collections
import datetime
import math
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import re
import string

from collections import namedtuple 
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from nltk.tokenize import word_tokenize

##################### Prep the comments for modelling #########################

# Define the functions for prepping the data (either developing a new model - prepCommentForTraining -
# or prepping data for use with an existing model - prepCommentForScoring)


def prepCommentForTraining(df,                             # Name of the dataframe being prepped
                           existingCommentColumnName,      # Name of the column with the comments in not yet prepped
                           newCommentColumnName,           # Name of the new column to receive the prepped data
                           dependentVariableColumnName,    # IF training the data, the name of the column containing the dependent variable                     
                           dependentVariableType,          # either 'categorical' or 'binary'
                           trained_maxlen,                 # the maximum number of words you want the model to be able to cope with in a comment
                           VOCAB_SIZE,                     # The vocab size you wish to use for the model (the number of most frequent words to include in the model)
                           training_proportion,            # Number between 0 and 1 you wish to use to train the data, e.g. 0.75
                           existing_word2index_model,      # If using an existing word2index model (e.g. if training multiple models and don't want each to have a different word2index) 
                           word2index_name_details):       # The name of the word2index model being generated (prefixed by date, suffixed by maxlen)

    # Import the empty arrays in case they are needed
    global X_train_numbers
    global y_train
    global X_test_numbers
    global y_test
    
    print('testing')

    # Set the comment to lower case
    df[newCommentColumnName] = df[existingCommentColumnName].str.lower()
    print('case set to lower')

    # Sort out all the fucking my name is John.I live in london bollock
    df[newCommentColumnName] = df[newCommentColumnName].replace('\.\.\.',' ', regex=True)
    df[newCommentColumnName] = df[newCommentColumnName].replace('\.','. ', regex=True)
    df[newCommentColumnName] = df[newCommentColumnName].replace('  ',' ', regex=True)
    df[newCommentColumnName] = df[newCommentColumnName].replace('  ',' ', regex=True)

    # Change '+' for '&' to help with a+e / a&e recognition
    df[newCommentColumnName] = df[newCommentColumnName].replace('\+','&', regex=True)
    print('+ replaced with &')
    # replace back and forward slashes with a space
    df[newCommentColumnName] = df[newCommentColumnName].replace('\/', ' ', regex=True)
    print('forward slashes replaced with space')

    # Clear out unidentified unicode, e.g u2026
    df[newCommentColumnName] = df[newCommentColumnName].replace('u\d[\w]+', '', regex=True)
    print('u200ef etc removed')

    # convert variants of a&e for the tokenizer
    df[newCommentColumnName] = df[newCommentColumnName].replace('a & e ','a&e ', regex=True)
    df[newCommentColumnName] = df[newCommentColumnName].replace(' a & e',' a&e ', regex=True)
    print('a & e correct to a&e')
    df[newCommentColumnName] = df[newCommentColumnName].replace('a&e','ae', regex=True)
    print('a&e changed to ae to deal with tokenizing nightmare!')

    #### Remove unwanted characters
    # List the characters you wish to keep
    wantedCharacters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
                        'p','q','r','s','t','u','v','w','x','y','z','&','.',',','!',
                        '?',':','_','-',"'",'@','#',' ','*', '"']


    # Keep only the desired wanted characters
    df[newCommentColumnName] = df[newCommentColumnName].apply(lambda x : ''.join([i for i in x if i in wantedCharacters]))
    print('non-standard characters removed')



    lengthBeforeEmptyStringCommentsRemoved = len(df)
    
    df = df[df[newCommentColumnName] != ''].reset_index(drop=True)
    df = df[df[newCommentColumnName].notnull()].reset_index(drop=True)
    
    lengthAfterEmptyStringCommentsRemoved = len(df)
    
    if lengthBeforeEmptyStringCommentsRemoved != lengthAfterEmptyStringCommentsRemoved:
        print(str(lengthBeforeEmptyStringCommentsRemoved - lengthAfterEmptyStringCommentsRemoved) + ' newly empty/blank comments removed from trainingData')
   
    trainProportion = math.ceil(len(df)*training_proportion)
    print('trainProportion = math.ceil(len(df)*training_proportion)') 
   
    if existing_word2index_model != 'None':
        
        print('Using existing word2index model')
        word2index_Name = existing_word2index_model
        
        # Load the dictionaries
        with open(existing_word2index_model, 'rb') as handle:
             word2index = pickle.load(handle)
     
        # Derive current trained_maxlen from index2word dictionary name
        maxlen = int(((existing_word2index_model).split('maxLen ')[1]).split('.p')[0])
         
        xs, ys = [], []
        
        for row in range(len(df)):
            
            ys.append(df[dependentVariableColumnName][row])
            
            words = [x.lower() for x in nltk.word_tokenize(df['comment'][row])]
            wids = [word2index[word] for word in words]
            xs.append(wids)
                
        X_numbers = pad_sequences(xs, maxlen=maxlen, padding='post')
            
        X_train_numbers = X_numbers[:trainProportion]
        X_test_numbers = X_numbers[trainProportion:]
        print('X number sorted')
        
        # If the dependent variable is categorical, the first column appears to always be zeroes,
        # and hence redundant.
        
        y = np_utils.to_categorical(ys)
        
        if dependentVariableType == 'categorical':
            y = y[:,1:]    
        elif dependentVariableType == 'binary':
            y = y
        else:
            print('dependentVariableType not recognised')
        
        y_train = y[:trainProportion]
        y_test = y[trainProportion:]
        print('Y number sorted')
        
        train = df[:trainProportion]
        test = df[trainProportion:]
        
    else:

        print('Generating new word2index model')
        
        # Calculate the maximum number of words in any comment
        counter = collections.Counter()
        maxlen = 0
        for row in range(len(df[newCommentColumnName])):
            words = [x.lower() for x in nltk.word_tokenize(df[newCommentColumnName][row])]
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                counter[word] += 1
        print('\nmaxlen in training data: '+str(maxlen))
        
        if maxlen > trained_maxlen:
            print('** New comment longer than previous maxlen comment **')
        else:
            maxlen = trained_maxlen
        
        print('maxlen set to: '+str(maxlen))
        
        word2index = collections.defaultdict(int)
        print('word2index = collections.defaultdict(int)')
    
        for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
            word2index[word[0]] = wid + 1
        #vocab_sz = len(word2index) + 1
        index2word = {v:k for k, v in word2index.items()}
        print("""for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
                    word2index[word[0]] = wid + 1
                #vocab_sz = len(word2index) + 1
                index2word = {v:k for k, v in word2index.items()}""")
    
        # Save the word2index dictionary just developed for future scoring
        word2index_Name = str(datetime.datetime.now().strftime("%Y%m%d"))+' trainingData ' + word2index_name_details + ' word2index maxLen '+str(maxlen)+'.pickle'
        print("word2index_Name: "+str(word2index_Name))
            
        with open(word2index_Name, 'wb') as handle:
            pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('\nword2index pickle dumped, name: '+str(word2index_Name))
    
        # Save the index2word dictionary just developed for future scoring
        index2word_Name = str(datetime.datetime.now().strftime("%Y%m%d"))+' trainingData index2word maxLen '+str(maxlen)+'.pickle'
        print("index2word_Name: "+str(index2word_Name))
    
        # with open(index2word_Name, 'wb') as handle:
        #     pickle.dump(index2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('index2word pickle dumped, name: '+str(index2word_Name))
        
        xs, ys = [], []
        
        for row in range(len(df)):
            
            ys.append(df[dependentVariableColumnName][row])
            
            words = [x.lower() for x in nltk.word_tokenize(df['comment'][row])]
            wids = [word2index[word] for word in words]
            xs.append(wids)
                
        X_numbers = pad_sequences(xs, maxlen=maxlen, padding='post')
            
        X_train_numbers = X_numbers[:trainProportion]
        X_test_numbers = X_numbers[trainProportion:]
        print('X number sorted')
        
        # If the dependent variable is categorical, the first column appears to always be zeroes,
        # and hence redundant.
        
        y = np_utils.to_categorical(ys)
        
        if dependentVariableType == 'categorical':
            y = y[:,1:]    
        elif dependentVariableType == 'binary':
            y = y
        else:
            print('dependentVariableType not recognised')
        
        y_train = y[:trainProportion]
        y_test = y[trainProportion:]
        print('Y number sorted')
            
        train = df[:trainProportion]
        test = df[trainProportion:]
        
    PreppedData = namedtuple('Prepped',["X_train_numbers", "y_train", "X_test_numbers", "y_test", 
                                    "train", "test", "maxlen", "word2index_Name"])
    prepped_data = PreppedData(X_train_numbers, y_train, X_test_numbers, y_test,
                      train, test, maxlen, word2index_Name)
    return prepped_data
#     return(X_train_numbers, y_train, X_test_numbers, y_test, train, test, maxlen, word2index_Name)
    

# prepCommentForTraining(training_tweets,                             # Name of the dataframe being prepped
#                            'cleanedComment',      # Name of the column with the comments in not yet prepped
#                            'comment',           # Name of the new column to receive the prepped data
#                            'recommend',    # IF training the data, the name of the column containing the dependent variable                     
#                            100,                 # the maximum number of words you want the model to be able to cope with in a comment
#                            5000,                     # The vocab size you wish to use for the model (the number of most frequent words to include in the model)
#                            0.75)           # Number between 0 and 1 you wish to use to train the data, e.g. 0.75

