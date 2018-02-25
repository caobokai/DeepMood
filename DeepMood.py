'''
A demo of DeepMood on synthetic data.
[bib] DeepMood: Modeling Mobile Phone Typing Dynamics for Mood Detection, B. Cao et al., 2017.
'''

from __future__ import print_function
import sys
import os.path
import random
import pandas as pd
import numpy as np
import cPickle as pickle
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import *

def load_data(modes, params, n_samples, max_value = 1000):
    data = {}
    label = []
    for mode in modes:
        data[mode] = []
    for i in range(n_samples):
        if random.random() < .5:
            for mode in modes:
                # Random sequence length
                len = random.randint(params['seq_min_len'], params['seq_max_len'])
                # Generate a linear sequence
                rand_start = random.randint(0, max_value - len)
                s = [[float(i) / max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(params['seq_max_len'] - len)]
                data[mode].append(s)
            label.append(1)
        else:
            for mode in modes:
                # Random sequence length
                len = random.randint(params['seq_min_len'], params['seq_max_len'])
                # Generate a random sequence
                s = [[float(random.randint(0, max_value)) / max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(params['seq_max_len'] - len)]
                data[mode].append(s)
            label.append(0)
    for mode in modes:
        data[mode] = np.array(data[mode])
    label = np.array(label)
    return data, label

def evaluate(y_test, y_pred, params):
    res = {}
    if params['is_clf']:
        res['accuracy'] = float(sum(y_test == y_pred)) / len(y_test)
        res['precision'] = float(sum(y_test & y_pred) + 1) / (sum(y_pred) + 1)
        res['recall'] = float(sum(y_test & y_pred) + 1) / (sum(y_test) + 1)
        res['f_score'] = 2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    else:
        res['rmse'] = np.sqrt(np.mean(np.square(y_test - y_pred)))
        res['mae'] = np.mean(np.abs(y_test - y_pred))
        res['explained_variance_score'] = 1 - np.square(np.std(y_test - y_pred)) / np.square(np.std(y_test))
        res['r2_score'] = 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))
    print(' '.join(["%s: %.4f" % (i, res[i]) for i in res]))
    return res

def mvm_decision_function(arg):
    n_modes = len(arg)
    latentx = arg
    y = K.concatenate([K.sum(K.prod(K.stack([latentx[j][:, i * params['n_latent'] : (i + 1) * params['n_latent']] for j in range(n_modes)]), axis = 0), \
                             axis = -1, keepdims = True) for i in range(params['n_classes'])])
    return y

def fm_decision_function(arg):
    latentx, bias = arg[0], arg[1]
    pairwise = K.concatenate([K.sum(K.square(latentx[:, i * params['n_latent'] : (i + 1) * params['n_latent']]), \
                             axis = -1, keepdims = True) for i in range(params['n_classes'])])
    y = K.sum(K.tf.pack([pairwise, bias]), axis = 0)
    return y

def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def create_model(train_data, test_data, params):
    input_list = []
    output_list = []
    train_list = []
    test_list = []
    if params['idx'] == 1:
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape = (params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value = 0, input_shape = (params['seq_max_len'], train_data[mode][0].shape[1]))(sub_input)
                sub_output = Bidirectional(GRU(output_dim = params['n_hidden'], return_sequences = False, consume_less = 'mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                output_list.append(drop_output)
        x = merge(output_list, mode = 'concat') if len(output_list) > 1 else output_list[0]
        latentx = Dense(params['n_classes'] * params['n_latent'], activation = 'relu', bias = True if params['bias'] else False)(x)
        y = Dense(params['n_classes'], bias = False)(latentx)
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    if params['idx'] == 2:
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape = (params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value = 0, input_shape = (params['seq_max_len'], train_data[mode][0].shape[1]))(sub_input)
                sub_output = Bidirectional(GRU(output_dim = params['n_hidden'], return_sequences = False, consume_less = 'mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                output_list.append(drop_output)
        x = merge(output_list, mode = 'concat') if len(output_list) > 1 else output_list[0]
        latentx = Dense(params['n_latent'] * params['n_classes'], bias = False)(x)
        bias = Dense(params['n_classes'], bias = True if params['bias'] else False)(x)
        y = merge([latentx, bias], mode = fm_decision_function, output_shape = (params['n_classes'], ))
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    if params['idx'] == 3:
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape = (params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value = 0, input_shape = (params['seq_max_len'], train_data[mode][0].shape[1]))(sub_input)
                sub_output = Bidirectional(GRU(output_dim = params['n_hidden'], return_sequences = False, consume_less = 'mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                latentx = Dense(params['n_latent'] * params['n_classes'], bias = True if params['bias'] else False)(drop_output)
                output_list.append(latentx)
        y = merge(output_list, mode = mvm_decision_function, output_shape = (params['n_classes'], ))
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    model = Model(input = input_list, output = y_act)
    model.compile(loss = objective, optimizer = RMSprop(lr = params['lr']), metrics = metric)
    return model, train_list, test_list

def run_model(train_data, test_data, y_train, y_test, params):
    model, X_train, X_test = create_model(train_data, test_data, params)
    hist = model.fit(x = X_train, y = y_train, batch_size = params['batch_size'], verbose = 2, \
        nb_epoch = params['n_epochs'], validation_data = (X_test, y_test))
    y_score = model.predict(X_test, batch_size = params['batch_size'], verbose = 0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32') if params['is_clf'] else np.ravel(y_score)
    return y_pred, hist.history

modes = ['alphanum', 'special', 'accel']
params = {'seq_min_len': 10,
          'seq_max_len': 100,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.1,
          'n_epochs': 500,
          'n_hidden': 8,
          'n_latent': 8,
          'n_classes': 1,
          'bias': 1,
          'is_clf': 1,
          'idx': 3, # 1: dnn, 2: dfm, 3: dmvm
          'includes_alphanum': 1,
          'includes_special': 1,
          'includes_accel': 1}
train_data, y_train = load_data(modes, params, 800)
test_data, y_test = load_data(modes, params, 200)
y_pred, hist = run_model(train_data, test_data, y_train, y_test, params)
res = evaluate(y_test, y_pred, params)