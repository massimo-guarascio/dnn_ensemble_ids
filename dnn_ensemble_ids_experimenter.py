#import libraries
from __future__ import print_function

# WORK ONLY WITH sklearn <= 0.21
import warnings
warnings.filterwarnings("ignore")

from timeit import default_timer as timer
import os
import pandas as pd
import numpy as np
import random as rn
import tensorflow as tf
os.environ['KERAS_BACKEND'] = "tensorflow"
import keras
import math
from keras import backend as K, optimizers, metrics
import sys

try:
    # py3
    from configparser import ConfigParser
except:
    from ConfigParser import SafeConfigParser as ConfigParser

#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
#from keras.engine.saving import load_model
from keras.layers.noise import GaussianNoise

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics.classification import confusion_matrix, classification_report
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.data import StandardScaler, Normalizer, minmax_scale,OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

#from losses import binary_focal_loss
#from ensemble_factory import *

from imblearn.under_sampling import RandomUnderSampler

import pickle as pk
from collections import Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

#SEED
seed = 56

#CONSTANTs
FOCAL_LOSS = "focal_loss"
COST_SENSITIVE_LOSS = "cost_sensitive_loss"
ENSEMBLE_MAX = "ensemble_max"
ENSEMBLE_AVG = "ensemble_avg"
ENSEMBLE_STACK = "ensemble_stack"
ENSEMBLE_F_STACK = "ensemble_f_stack"
ENSEMBLE_F_STACK_V2 = "ensemble_f_stack_v2"
ENSEMBLE_MOE = "ensemble_moe"

#################
#UTILITY METHODs#
#################

#CREATE EXTENDED INPUT
def create_extended_input(raw_input_layer):

    extended_features = []
    range_values = [2, 4, 8, 16]
    
    extended_features.append(raw_input_layer)
    
    one_minus_i = Lambda(lambda x: 1 - K.clip(x, 0, 1))(raw_input_layer)
    extended_features.append(one_minus_i)
        
    #power
    for v in range_values:
        power_i = Lambda(lambda x: x**v)(raw_input_layer)
        extended_features.append(power_i)

    #root
    for v in range_values:
        root_i = Lambda(lambda x: K.clip(x, 0, 1) ** (1/v))(raw_input_layer)
        extended_features.append(root_i)
    
    #sin and 1-cos
    sin_i = Lambda(lambda x: K.sin(math.pi * K.clip(x, 0, 1)))(raw_input_layer)
    extended_features.append(sin_i)
    one_minus_cos_i = Lambda(lambda x: 1 - K.cos(math.pi * K.clip(x, 0, 1)))(raw_input_layer)
    extended_features.append(one_minus_cos_i)
    
    #other extensions
    log_i = Lambda(lambda x: K.log(K.clip(x, 0, 1) + 1)/math.log(2))(raw_input_layer)
    extended_features.append(log_i)
    one_minus_inv_log_i = Lambda(lambda x: 1 - K.log(K.clip(-x, 0, 1) + 2)/math.log(2))(raw_input_layer)    
    extended_features.append(one_minus_inv_log_i)    
    exp_i = Lambda(lambda x: K.exp(x - 1))(raw_input_layer)
    extended_features.append(exp_i)
    one_minus_exp_i = Lambda(lambda x: 1- K.exp(-x))(raw_input_layer)
    extended_features.append(one_minus_exp_i)

    # improved input
    return Concatenate()(extended_features)


#CREATE A SINGLE RESIDUAL BLOCK INCLUDING 2 BUILDING BLOCKs
def create_single_building_block(output, input, factors, dropout_pcg):

    # building block
    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(output)
    out = Concatenate()([input, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)
    
    #res
    add = Add()([out, output])
    
    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(add)
    out = Concatenate()([input, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)
    
    return out
    
#CREATE A NUMBER OF RESIDUAL BLOCKs    
def create_multiple_building_block(out, input, factors, dropout_pcg, depth):

    current_out = create_single_building_block(out, input, factors, dropout_pcg)
    for i in range(1, depth):
        current_out = create_single_building_block(current_out, input, factors, dropout_pcg)
    return current_out
    
#CREATE THE BASE MODEL ARCHITECTURE
def create_dnn_tf_func(dimensions, base_learner_parameters):

    #parameters
    dropout_pcg = base_learner_parameters["dropout_pcg"]
    embedding_size = base_learner_parameters["embedding_size"]

    #latent factors
    factors = base_learner_parameters["factors"]

    #init input
    raw_input_layer = keras.layers.Input(shape=(dimensions,))

    #improved layer
    extended_input_layer = create_extended_input(raw_input_layer)

    #feature embedding
    input_layer = Dense(embedding_size, kernel_initializer=glorot_normal(seed), activation="tanh")(extended_input_layer)
    
    #depth and width factors
    depth = base_learner_parameters["depth"] - 1

    # l1
    l = Dense(factors, kernel_initializer=glorot_normal(seed), activation="tanh")(input_layer)
    out = Concatenate()([input_layer, l])
    out = BatchNormalization()(out)
    out = Dropout(dropout_pcg)(out)

    #add deep BB
    out = create_multiple_building_block(out, input_layer, factors, dropout_pcg, depth)    
    
    #l-1
    decision_layer = Dense(factors, kernel_initializer=glorot_normal(seed), activation="sigmoid")(out)
    out = BatchNormalization()(decision_layer)
     
    # lp
    out = Dense(1, kernel_initializer=glorot_normal(seed), activation="sigmoid")(out)

    model = keras.models.Model(inputs=[raw_input_layer], outputs=out)

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-14)

    if loss_type == base_learner_parameters["loss_type"]:
        model.compile(loss=cost_sensitive_loss(base_learner_parameters["fn_weight"], base_learner_parameters["fp_weight"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        #model.compile(loss="mse", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])

    else:
        if loss_type == base_learner_parameters["loss_type"]:
            model.compile(loss=binary_focal_loss(gamma=base_learner_parameters["gamma"], alpha=base_learner_parameters["alpha"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("error: Unknown loss")
            sys.exit(-1)

    return model, raw_input_layer, out, decision_layer

#COST SENSITIVE LOSS
def cost_sensitive_loss(fn_weight=1. , fp_weight=1.):
    def inner_cost_sensitive_loss(y_true, y_pred):
        mask_fn = K.clip(K.round(y_true-y_pred), 0, 1)
        w_fn = mask_fn * fn_weight
        mask_fp = K.clip(K.round(y_pred-y_true), 0, 1)
        w_fp = mask_fp * fp_weight
        mask_other = K.clip(1-K.round(K.abs(y_true-y_pred)), 0, 1)
        w = w_fn + w_fp + mask_other
        return K.mean(K.square(y_pred - y_true) * w) 
    return inner_cost_sensitive_loss

#COMPUTE MACRO RECALL FOR BINARY PROBLEMS
def macro_recall(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def rec(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        total_per_class = K.cast(K.equal(K.round(y_true), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        class_acc = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(total_per_class), 1),
                                                                         K.floatx())
        return class_acc

    v = 0
    for i in range(num_classes):
        v = v + rec(y_true, y_pred)
        class_id = i

    return v / num_classes

#COMPUTE MACRO PRECISION FOR BINARY PROBLEMS
def macro_precision(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def prec(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        class_acc = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(accuracy_mask), 1), K.floatx())
        return class_acc

    v = 0
    for i in range(num_classes):
        v = v + prec(y_true, y_pred)
        class_id = i

    return v / num_classes

#COMPUTE MACRO F1-SCORE FOR BINARY PROBLEMS
def macro_f1(y_true, y_pred):
    # HARDCODED
    num_classes = 2
    class_id = 0

    def f1(y_true, y_pred):
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        prec = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(accuracy_mask), 1), K.floatx())
        accuracy_mask = K.cast(K.equal(K.round(y_pred), class_id), 'int32')
        total_per_class = K.cast(K.equal(K.round(y_true), class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(y_true, K.round(y_pred)), 'int32') * accuracy_mask
        rec = K.cast(K.sum(class_acc_tensor), K.floatx()) / K.cast(K.maximum(K.sum(total_per_class), 1), K.floatx())
        return 2 * K.cast(K.sum(rec), K.floatx()) * K.cast(K.sum(prec), K.floatx()) / K.cast(
            K.maximum(K.sum(prec + rec), 1), K.floatx())

    v = 0
    for i in range(num_classes):
        v = v + f1(y_true, y_pred)
        class_id = i

    return v / num_classes

# def convert_string_to_float(value):
#     return float(value.decode('ascii'))

#CREATE DICTIONARY FOR CATEGORICAL ATTRIBUTE ENCODING
def create_dictionary(dataset_paths, delim, decimal, load_from_file, preprocesser_folder_path, suffix, to_remove_list_parameter, categorical_feature_list_parameter):

    if load_from_file:
        #DO SOMETHING
        # STORING LabelEncoderMap
        column_dict = pk.load(open(preprocesser_folder_path + "clm_dict_"+suffix+".sav", 'rb'))

        # STORING ohe
        x_ohe = pk.load(open(preprocesser_folder_path + "ohe_"+suffix+".sav", 'rb'))
        if debug:
            print("LOADED")

        return (column_dict, x_ohe)

    data_list = []
    for i in range(0, len(dataset_paths)):

        # --- LOADING DATASET ---
        data = readData(dataset_paths[i], delim, decimal)

        # REMOVING ID
        # to_remove = ["fc_id","fc_tstamp","fc_src_port", "fc_dst_port"]
        to_remove = to_remove_list_parameter #["fc_id", "fc_tstamp", "fc_src_port", "fc_dst_port", "fc_src_addr", "fc_dst_addr", "lpi_category", "lpi_proto", "crl_group", "crl_name" ]
        data = data.drop(to_remove, 1)

        # CLEAN STRING
        #data["class"] = data["class"].apply(convert_string_to_float)

        data_list.append(data)
        if debug:
            print("READ")

    # , "lpi_category", "lpi_proto", "crl_group", "crl_name"
    categorical_feature_list = categorical_feature_list_parameter #["fc_proto"]

    data_list = pd.concat(data_list)
    # mapping
    column_dict = {}

    # CREATE LABEL ENCODER
    for column_name in categorical_feature_list:
        #~ column_encoder = sklearn.preprocessing.label.LabelEncoder()
        column_encoder = sklearn.preprocessing.LabelEncoder()
        # column_encoder.fit(preprocessed_training[column_name])
        column_encoder.fit(data_list[column_name])
        column_dict[column_name] = column_encoder

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        data_list[column_name] = current_encoder.transform(data_list[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(data_list.columns.get_loc(v))

    if debug :
        print("Features indexes: ", indexes_to_encode)

    x_ohe = OneHotEncoder(categorical_features=indexes_to_encode, sparse=False)

    x_ohe.fit(data_list)

    #STORING LabelEncoderMap
    pk.dump(column_dict, open(preprocesser_folder_path + "clm_dict_"+suffix+".sav", 'wb') ,  protocol =2)

    #STORING ohe
    pk.dump(x_ohe, open(preprocesser_folder_path+"ohe_"+suffix+".sav", 'wb'), protocol =2 )



    return (column_dict, x_ohe)

#READ A SINGLE CHUNCK
def readData(path, delimiter, decimal):
    "READ SINGLE FILE"
    df=pd.read_csv(path, delimiter=delimiter, decimal=decimal, engine="c", header = 0)
    return df
    
#
def extract_preprocessed_data(dataset_path, delim, decimal, train_perc, test_perc, column_dict, x_ohe):

    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)

    if (debug) :
        print("--- LOADED DATASET ---", dataset_path)

    #REMOVING id-like useless features
    to_remove = to_remove_list_parameter 
    data = data.drop(to_remove,1)

    if (debug) :
        print("--- USELESS ATTRIBUTES REMOVED ---")

    #PRINT CLASS DISTRIBUTION
    if (debug):
        print(data["class"].value_counts())

    ## --- PREPROCESSING ---

    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    #create X and y for training set and test set
    X = data.drop("class",1)
    
    y = data["class"]
    
    #STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_perc,test_size = test_perc, random_state = seed)
    if debug:
    	print("--- CREATED TRAINING AND TEST SET ---")

    #PRINT DISTRIBUTION
    if (debug) :
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())

    # categorical features conversion via one-hot encoding

    #categorical feature list
    categorical_feature_list = categorical_feature_list_parameter 

    preprocessed_training = X_train  

    #SCALING NUMERICAL FEATURES
    scaler_x = MinMaxScaler(feature_range=(0, 1)) 

    to_scale = preprocessed_training.columns.difference(categorical_feature_list)

    scaler_x.fit(preprocessed_training[to_scale])

    preprocessed_training[to_scale] = scaler_x.transform(preprocessed_training[to_scale])

    if(debug) :
        print("--- SCALING APPLIED ---")

    # CREATE AND APPLY OHE

    #CREATE LABEL ENCODER

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        preprocessed_training[column_name] = current_encoder.transform(preprocessed_training[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(preprocessed_training.columns.get_loc(v))

    preprocessed_training = x_ohe.transform(preprocessed_training)

    if (debug) :
        print("--- CATEGORICAL FEATURE ENCODED ---")

    #preparing test

    #create copy
    preprocessed_test = X_test.copy()
    preprocessed_test[to_scale] = scaler_x.transform(preprocessed_test[to_scale])
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        preprocessed_test[column_name] = current_encoder.transform(preprocessed_test[column_name])
    preprocessed_test = x_ohe.transform(preprocessed_test)

    return (preprocessed_training,y_train,preprocessed_test,y_test)

#utility method
def first_extract_preprocessed_data(dataset_path, delim, decimal, train_perc,test_perc, column_dict, x_ohe,out_dataset):

    # --- LOADING DATASET ---
    data = readData(dataset_path, delim, decimal)

    if (debug) :
        print("--- LOADED DATASET ---")

    #PRINT COLUMN NAMES
    if (debug):
        print(data.columns)

    #PRINT HEAD OF THE DATASET
    if (debug):
        print(data.head(3))

    #REMOVING ID
    #to_remove = ["fc_id","fc_tstamp","fc_src_port", "fc_dst_port"]
    to_remove = to_remove_list_parameter #["fc_id","fc_tstamp","fc_src_port", "fc_dst_port","fc_src_addr","fc_dst_addr", "lpi_category", "lpi_proto", "crl_group", "crl_name" ]
    data = data.drop(to_remove,1)

    if (debug) :
        print("--- USELESS ATTRIBUTES REMOVED ---")

    #PRINT COLUMN NAMES
    if (debug) :
        print("FEATURES AFTER USELESSS ATTRIBUTES REMOVED:")
        print(data.columns)

    # PRINT HEAD OF THE DATASET
    if (debug):
        print(data.head(3))

    #PRINT CLASS DISTRIBUTION
    if (debug):
        print(data["class"].value_counts())

    ## --- PREPROCESSING ---

    

    # categorical features conversion via one-hot encoding

    categorical_feature_list = categorical_feature_list_parameter
    
    #SCALING NUMERICAL FEATURES
    scaler_x = MinMaxScaler(feature_range=(0, 1)) 

    to_scale = data.columns.difference(categorical_feature_list)

    scaler_x.fit(data[to_scale])

    data[to_scale] = scaler_x.transform(data[to_scale])

    if(debug) :
        print("--- SCALING APPLIED ---")

    # CREATE AND APPLY OHE

    # APPLY ENCODING
    for column_name in categorical_feature_list:
        current_encoder = column_dict[column_name]
        data[column_name] = current_encoder.transform(data[column_name])

    indexes_to_encode = []
    for v in categorical_feature_list:
        indexes_to_encode.append(data.columns.get_loc(v))

    if debug :
        print("Features indexes: ", indexes_to_encode)

    tmp_X = data.drop("class",1)
    tmp_y = data["class"]
    tmpdata = x_ohe.transform(tmp_X)
    data=pd.DataFrame(tmpdata, columns = ["Attr_"+str(int(i)) for i in range(tmpdata.shape[1])])
    data['class']=tmp_y    

    if (debug) :
        print("--- CATEGORICAL FEATURE ENCODED ---")

    #preparing test
    
    #saving data
    data.to_pickle(out_dataset)
    
    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    #CLEAN STRING
    #data["class"] = data["class"].apply(convert_string_to_float)

    #create X and y for training set and test set
    X = data.drop("class",1)
    
    y = data["class"]
    
    #STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc,test_size = test_perc, random_state = seed)
    if debug:
    	print("--- CREATED TRAINING AND TEST SET ---")

    #PRINT DISTRIBUTION
    if (debug) :
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())
        print ("Type xtrain" , type(X_train), " Type Y ", type(y_train))
    return (np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test))



#only first time load and preprocess all data

def eff_extract_preprocessed_data(out_dataset, delim, decimal, train_perc, test_perc, column_dict, x_ohe):

    #loading data
    data=pd.read_pickle(out_dataset)
    # STEP 1: SPLIT THE DATASET IN TRAINING SET AND TEST SET
    # STEP 2: CATEGORICAL ATTRIBUTES ARE CONVERTED IN NUMERICAL BY USING ONE-HOT ENCODIING

    #CLEAN STRING
    #data["class"] = data["class"].apply(convert_string_to_float)

    #create X and y for training set and test set
    X = data.drop("class",1)
    
    y = data["class"]
    
    #STEP 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_perc,test_size = test_perc, random_state = seed)
    if debug:
    	print("--- CREATED TRAINING AND TEST SET ---")

    #PRINT DISTRIBUTION
    if (debug) :
        print("Training distribution")
        print(y_train.value_counts())
        print("Test distribution")
        print(y_test.value_counts())

    return (np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test))


#METHOD FOR TRAINING A BASE MODEL
def build_base_model(X_growing, X_validation, y_growing, y_validation, batch_size, num_epoch,  verbose_fit, verbose_model_check, old_model_name, new_model_name, load_from_file, base_learner_parameters, model_folder_path, to_remove_list_parameter, categorical_feature_list_parameter):
    
    
    model_init = create_dnn_tf_func(X_growing.shape[1], base_learner_parameters)

    model = model_init[0]
    model_input = model_init[1]
    model_output = model_init[2]
    model_features = model_init[3]

    if model == "not_init":
        print("Classifier not init, exit")
        exit(-1)

    if(debug) :
        print("--- INIT COMPLETED ---")

    # building

    # callback list
    best_model_path =  ''.join([model_folder_path, new_model_name , "_" ,loss_type , ".hdf5"])
    #save_weights_only = False,
    checkpoint = ModelCheckpoint(best_model_path, monitor='val_macro_f1', verbose=verbose_model_check, save_best_only=True,
                                 save_weights_only=True, mode='max')
    opt = ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
    #lrm = LearningRateMonitor()
    cb_list = [checkpoint, opt]

    #LOAD OLD MODEL IF EXIST
    if old_model_name is not None :
        # load old model
        old_model_path = ''.join([model_folder_path, old_model_name, "_" ,loss_type , ".hdf5"])
        #model = load_model(old_model_path, custom_objects={'binary_class_weighted_loss_tf': binary_class_weighted_loss_tf})

        #TRANSFER LEARNING
        #UNCOMMENT TO ENABLE TRANSFER LEARNING
        #model.load_weights(old_model_path)
        if debug:
        	print("### OLD MODEL LOADED ###")

    # fit
    if not load_from_file:
        if debug:
            print("### FITTING ###")
        start = timer();             
        model.fit(X_growing, y_growing, batch_size=batch_size, epochs=num_epoch, callbacks=cb_list, validation_data=(X_validation, y_validation), verbose=verbose_fit)
        end = timer(); total_time=end-start
    else:
        total_time=0
        if(debug):
            print("no training phase: loaded model from file")

    if debug :
        print("base model created")

    # load best model
    model.load_weights(best_model_path)

    return (model, model_features, total_time)


#FACTORY FOR ENSEMBLE MODELs
def create_ensemble(models, parameters):

    if parameters["ensemble_type"] == ENSEMBLE_MAX:
        return ensemble_max(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_AVG:
        return ensemble_avg(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_STACK:
        return ensemble_stacking(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_F_STACK:
        return ensemble_stacking_feature(models, parameters)

    if parameters["ensemble_type"] == ENSEMBLE_F_STACK_V2:
        return ensemble_stacking_feature_V2(models, parameters)
        
    if parameters["ensemble_type"] == ENSEMBLE_MOE:
        return ensemble_moe(models, parameters)

    print("ERROR: No ensemble specified")
    sys.exit(-1)


#ENSEMBLE STATEGY: MAX SCORE (NO TRAINABLE)
def ensemble_max(models, parameters, ensemble_model_name=None):

    def compute_strongest_pred(x):
        thr = tf.fill(tf.shape(x), 0.5)
        x1 = x - thr
        pos = K.relu(x1)
        neg = K.relu(-x1)
        max_pos_abs = K.max(pos, axis=1)
        max_neg_abs = K.max(neg, axis=1)
        bool_idx = K.greater(max_pos_abs, max_neg_abs)
        float_idx = K.cast(bool_idx, dtype=K.floatx())
        thr1 = tf.fill(tf.shape(max_pos_abs), 0.5)
        mask = float_idx * max_pos_abs - (1-float_idx) * max_neg_abs + thr1
        return K.reshape(mask, (-1, 1))

    def compute_strongest_pred_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] = 1
        return tuple(shape)

    freeze = parameters["do_freeze"]
    if freeze:
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m,parameters["freeze_base_models_partly"])

    ensemble_input = [m.input for m in models]
    base_preds = [m.output for m in models]
 
    x = Concatenate()(base_preds)
    
    y = Lambda(compute_strongest_pred, output_shape=compute_strongest_pred_shape)(x)

    ensemble_model = Model(ensemble_input, y, name='ensembleMax')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        ensemble_model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,  metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            ensemble_model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            ensemble_model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            #sys.exit(-1)

    return ensemble_model

#ENSEMBLE STATEGY: AVERAGE SCORE (NO TRAINABLE)
def ensemble_avg(models, parameters):

    freeze = parameters["do_freeze"]
    if freeze :
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model,parameters["freeze_base_models_partly"])
    inputs = [model.input for model in models]
    outputs = [model.output for model in models]
    y = Average()(outputs)
    model = Model(inputs, y, name='ensembleAvg')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-14)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,  metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model


#ENSEMBLE STRATEGY: DEEP STACKING (TRAINABLE)
def ensemble_stacking(models, parameters):

    freeze = parameters["do_freeze"]
    if freeze :
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m,parameters["freeze_base_models_partly"])

    inputs = [m.input for m in models]
    outputs = [m.output for m in models]

    #ADD default features
    #outputs.append(models[0].input)

    x = Concatenate()(outputs)
    factors = parameters["factors"]

    err_x = Lambda(lambda v: abs(v - 0.5))(x)

    #concatenate , models[0].input
    x = Concatenate()([x, err_x])

    #path
    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, x, name='ensembleStacking')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,  metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            #sys.exit(-1)

    return model


#ENSEMBLE STRATEGY: MIXTURE OF EXPERTS(TRAINABLE)
def ensemble_moe(models, parameters):
    freeze = parameters["do_freeze"]
    if freeze :
        for m in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(m,parameters["freeze_base_models_partly"])

    ensemble_input = [m.input for m in models]
    models_outputs = [m.output for m in models]

    # Gating network
    g = Dense(128, activation="tanh", kernel_initializer=glorot_normal(seed))(models[0].input)
    #g = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(inputs)
    #g = BatchNormalization()(g)
    #g = Dropout(0.2)(g)
    g = Dense(len(models_outputs), activation='softmax')(g)

    # Weighted combination
    p = Concatenate()(models_outputs)
    weighted_p = Multiply()([p, g])
    shape_list = models_outputs[0].get_shape().as_list()
    y = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), output_shape=tuple(shape_list[1:]))(weighted_p)

    model = Model(ensemble_input, y, name='ensemble_moe')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,  metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss: using default")
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy', 'mse', macro_f1])
            #sys.exit(-1)

    return model

#ENSEMBLE STRATEGY: DEEP STACKING WITH HIGH LEVEL FEATURES EXTRACTED FROM BASE MODELS (TRAINABLE)
def ensemble_stacking_feature(models, parameters):

    freeze = parameters["do_freeze"]
    if freeze :
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model,parameters["freeze_base_models_partly"])


    factors = parameters["factors"]

    #input declaration
    w_ensemble_input = [model.input for model in models]

    pred_list = [model.output for model in models]
    #print("len:",len(pred_list))

    #print("base_preds_tensor:", base_preds_tensor.shape)
    context_list = [model.layers[-4].output for model in models]

    pred_tensor = Concatenate()(pred_list)
    err_tensor = Lambda(lambda v: abs(v - 0.5))(pred_tensor)
    context_tensor = Concatenate()(context_list)

    #ADD default features
    #pred_list.append(models[0].input)

    #merged_context = Concatenate()(pred_list+context_list)

    merged_context = Concatenate()([pred_tensor, err_tensor, context_tensor])

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(merged_context)
    x = BatchNormalization()(x)
    
    x = Dense(1, activation='sigmoid')(x)

    model = Model(w_ensemble_input, x, name='ensemble_stacking_feature')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-9)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model

#ENSEMBLE STRATEGY: DEEP STACKING WITH HIGH LEVEL FEATURES EXTRACTED FROM BASE MODELS - variant 2 (TRAINABLE)
def ensemble_stacking_feature_V2(models, parameters):

    freeze = parameters["do_freeze"]
    if freeze :
        for model in models:
            if parameters["freeze_base_models_partly"] or parameters["freeze_base_models"]:
                freeze_model(model,parameters["freeze_base_models_partly"])

    w_ensemble_input = [model.input for model in models]

    pred_list = [model.output for model in models]
    #print("len:",len(pred_list))

    #print("base_preds_tensor:", base_preds_tensor.shape)
    context_list = [model.layers[-4].output for model in models]

    #ADD default features
    #pred_list.append(models[0].input)

    add_context = Add()(context_list)
    pred_list.append(add_context)
    merged_context = Concatenate()(pred_list)

    factors = parameters["factors"]

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(merged_context)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(factors, activation="tanh", kernel_initializer=glorot_normal(seed))(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(w_ensemble_input, x, name='ensemble_stacking_feature')

    opt = optimizers.rmsprop(lr=0.001, epsilon=1e-14)

    if parameters["loss_type"] == COST_SENSITIVE_LOSS:
        model.compile(loss=cost_sensitive_loss(parameters["fn_weight"], parameters["fp_weight"]), optimizer=opt,
                      metrics=['accuracy', 'mse', macro_f1])
    else:
        if parameters["loss_type"] == FOCAL_LOSS:
            model.compile(loss=binary_focal_loss(gamma=parameters["gamma"], alpha=parameters["alpha"]), optimizer=opt,
                          metrics=['accuracy', 'mse', macro_f1])
        else:
            print("Unknown loss")
            sys.exit(-1)

    return model

#FREEZE THE WEIGHT OF A MODEL
def freeze_model(model,partly=False):
    if not partly:
        model.trainable = False
    for layer in model.layers:
        if not partly or ("features_0" not in layer.name) and ("features_1" not in layer.name):
            layer.trainable = False


#MAIN

if __name__ == "__main__":
	
	# command line arguments
    if len (sys.argv) < 3:
     	print("Usage: %s <random seed> <file_output> <file_params (optional default.ini)>" % sys.argv[0])
     	sys.exit(-1)
    seed= int(sys.argv[1])
    file_output=sys.argv[2]
    if len (sys.argv) == 3:
    	file_params='default.ini'
    else:
        file_params=sys.argv[3]
    
    # seed initialization     
    os.environ['PYTHONHASHSEED'] = '0'
    #numpy seed
    np.random.seed(seed)
    #rn seed
    rn.seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    #sess
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    #tf seed
    tf.set_random_seed(seed)
    
    # --- PARAMETERS ---
    
    #other parameters
    delim = ","
    decimal = "."
    train_perc = 0.1    #implicitly make an undersampling if train_perc+test_perc <1.0
    test_perc = 0.33
    batch_size = 512
    #suppose training-set (after splitting train and test) is composed of 1000 tuples, the new training set for training DNn will be training_growing_dnn=1000*(1-training_perc)
    # and for training dnn you use  training_grow_dnn * (1-growing_perc) and for growing dnn, you use growing_perc * training_growing_dnn
    
    growing_perc = 0.30        #used to define the percentage of training set to grow the DNN
    validation_perc = 0.20     #used to define the percentage of training set to train the combining function of the ensemble
    
   

    #read params from file
    config = ConfigParser()
    config.read(file_params)


    #MANAGEMENT PARAMETERS read from argv[2] or 'default.in
    verbose_fit = config.getint('management', 'verbose_fit')
    verbose_model_check = config.getint('management', 'verbose_model_check')
    debug = config.getboolean('management', 'debug')
    load_preprocesser_from_file = config.getboolean('management', 'load_preprocesser_from_file') #First time set to False to create the preprocessers
    load_from_file = config.getboolean('management', 'load_from_file')     
    load_datasets_from_file = config.getboolean('management', 'load_datasets_from_file')
    if debug:
    	print ("verbose fit", verbose_fit," verbose model check", verbose_model_check," debug ",debug)
    	print ("load_from_preprocesser", load_preprocesser_from_file, " load_from_file", load_from_file," load_dataset_from_file ",load_datasets_from_file)
    #parameters running or not algorithms
    running_ensemble = config.getboolean('running', 'running_ensemble')  
    running_competitor=config.getboolean('running', 'running_competitor')  
    evaluate_base_models=config.getboolean('running', 'evaluate_base_models')  
    if debug:
    	print ("running_ensemble", running_ensemble, " running_competitor",running_competitor," evaluate_base_models",evaluate_base_models)
    train_perc=config.getfloat('running','train_perc')
    test_perc=config.getfloat('running','test_perc')
    growing_perc=config.getfloat('running','growing_perc')
    validation_perc=config.getfloat('running','validation_perc')
    if debug:
    	print ("Trainining ", train_perc," Testing ", test_perc," Growing ", growing_perc, " Validation ", validation_perc);
    
    #ENSEMBLE PARAMETERs
    ensemble_types = [ENSEMBLE_STACK, ENSEMBLE_F_STACK, ENSEMBLE_MOE, ENSEMBLE_MAX]
    
    do_freeze_choose = [False] 	#do_freeze_choose = [True, False] enable/disable weight freezing 
    ens_factors = 64 			#number for neurons for the ensemble layers

    #DNN ARCHITECTURE PARAMETERs
    dropout_pcg = 0.25			#dropout pcg
    num_epoch = 16				#training epochs

    factors = 32				#number for neurons for the base model layers

    #LOSSES PARAMETERs
    # available values: COST_SENSITIVE_LOSS, FOCAL_LOSS
    loss_type = config.get('architecture', 'loss_type') 
    add_competitor_to_ensemble = config.getboolean('architecture', 'add_competitor')
    num_epoch = config.getint('architecture','num_epochs')
    dropout_pcg = config.getfloat('architecture','dropout_perc')
     
    if debug:
    	print ("Loss= ",loss_type)
    gamma_loss_parameter = 2.
    alpha_loss_parameter = 0.5
    fn_weight = 4.		#false negative weight
    fp_weight = 4.		#false positive weight
    
    if debug:
	    print ("loss type ", loss_type, " add competitor ",add_competitor_to_ensemble," num epochs ", num_epoch, " dropout ", dropout_pcg)


    #FIXED PARAMETERs
    
    model_folder_path = "../res/output/model/"
    preprocesser_folder_path = "../res/output/preprocesser/"
    #data_path="../res/input/data_preprocessed_2/disjoint_samples"
    
    #chunk datasets for iscx
    #data_path="../res/input/data_preprocessed_2/"
    #data_names=["testbed12_preprocessed.csv","testbed13_preprocessed.csv","testbed14_preprocessed.csv","testbed15_preprocessed.csv","testbed17_preprocessed.csv"]
    #predata_names=["testbed12_preprocessed.pkl","testbed13_preprocessed.pkl","testbed14_preprocessed.pkl","testbed15_preprocessed.pkl","testbed17_preprocessed.pkl"]
    
    #model_names = ["model_day_2","model_day_3","model_day_4","model_day_5","model_day_7"]
    #dataset_names = ["dataset_day_2", "dataset_day_3", "dataset_day_4", "dataset_day_5", "dataset_day_7"]
    
    
    #chunk datasets for cicids 2017
    data_path="../res/input/data_preprocessed_2/id_dataset_2017"
    data_names=["Tuesday.csv","Wednesday.csv","Thursday.csv","Friday.csv"]
    predata_names=["Tuesday.pkl","Wednesday.pkl","Thursday.pkl","Friday.pkl"]
    
    model_names = ["model_Tuesday","model_Wednesday","model_Thursday","model_Friday"]
    dataset_names = ["dataset_Tuesday", "dataset_Wednesday", "dataset_Thursday", "dataset_Friday"]

    
    dataset_paths=[os.path.join(data_path, x) for x in data_names]
    predataset_paths=[os.path.join(data_path, x) for x in predata_names]
    if debug:
    	print ("datasets ", dataset_paths)

  
    #iscx categorical encoder and useless attributes
    #enc_suffix = "extended"
    #to_remove_list_parameter = ["fc_id", "fc_tstamp", "fc_src_port", "fc_dst_port", "fc_src_addr", "fc_dst_addr","lpi_proto", "lpi_category", "crl_group", "crl_name"]
    #categorical_feature_list_parameter = ["fc_proto"]
    
    
    #cicids categorical encoder and useless attributes
    enc_suffix = "extended_2017"
    to_remove_list_parameter = ["Destination_Port","Flow_Bytes_s","Flow_Packets_s"]    
    categorical_feature_list_parameter = ["port_type"]

    #Variables
    base_model_list = [ ]
    x_ensemble = [ ]
    y_ensemble = [ ]
    x_tests = [ ]
    y_tests = [ ]

    #base learner parameters
    base_learner_parameters = {}
    base_learner_parameters["loss_type"] = loss_type
    base_learner_parameters["factors"] = factors
    base_learner_parameters["dropout_pcg"] = dropout_pcg
    base_learner_parameters["gamma"] = gamma_loss_parameter
    base_learner_parameters["alpha"] = alpha_loss_parameter
    base_learner_parameters["fn_weight"] = fn_weight
    base_learner_parameters["fp_weight"] = fp_weight
    base_learner_parameters["embedding_size"] = 96  
    base_learner_parameters["depth"] = 3


    #dataset dictionary creation
    cat_pre = create_dictionary(dataset_paths, delim, decimal, load_preprocesser_from_file, preprocesser_folder_path, enc_suffix, to_remove_list_parameter, categorical_feature_list_parameter)
    column_dict = cat_pre[0]
    x_ohe = cat_pre[1]


    if debug:
    	print("LOAD/TRAIN BASE MODEL")

    for i in range(0, len(dataset_paths)):

        #EXTRACT DATASET        
        preprocessed_data = extract_preprocessed_data(dataset_paths[i], delim, decimal, train_perc,test_perc, column_dict, x_ohe)
        

        #TEMPORARY VARs
        preprocessed_training = preprocessed_data[0]
        y_train = preprocessed_data[1]
        preprocessed_test = preprocessed_data[2]
        y_test = preprocessed_data[3]

        #MODELS
        if i == 0 :
            old_model_name = None
        else :
            old_model_name = model_names[i-1]
        new_model_name = model_names[i]
        
        # split the training set into 2 datasets also stratifying uniformly for each class    (x_growing would be used to build the base model and X_validation for the ensemble 
        X_train_base, X_train_ensemble, y_train_base, y_train_ensemble = train_test_split(preprocessed_training, y_train, stratify=y_train, test_size=validation_perc, random_state=seed)
        
        if debug:
            #print("Num dimensions:", len(X_growing[0]))
            print("Num dimensions:", X_train_base.shape[1])
        X_growing, X_validation, y_growing, y_validation= train_test_split(X_train_base, y_train_base, stratify=y_train_base, test_size=growing_perc, random_state=seed)
        #CREATING MODELs
        result = build_base_model(X_growing, X_validation, y_growing, y_validation , batch_size, num_epoch, verbose_fit, verbose_model_check, old_model_name, new_model_name, load_from_file, base_learner_parameters, model_folder_path, to_remove_list_parameter, categorical_feature_list_parameter)

        base_model = result[0]
        
        x_ensemble.append(X_train_ensemble)
        y_ensemble.append(y_train_ensemble)
        
        if debug:
            print (dataset_names[i], " set for the training of the ensemble")
            print (y_train_ensemble.value_counts())
        
        total_time=result[2]
        
        #STORING FOR TESTING ENSEMBLE
        base_model_list.append(base_model)
        x_tests.append(preprocessed_test)
        y_tests.append(y_test)

        if (debug):

            print("RESULT TEST")

            pred_y_prob = base_model.predict(preprocessed_test, verbose=0)
            pred_y = np.around(pred_y_prob, 0)

            print(str(confusion_matrix(y_test, pred_y)).replace("[", " ").replace("]", " "))

            print(classification_report(y_test, pred_y))

            report_map = classification_report(y_test, pred_y, output_dict=True)
            auc_score = roc_auc_score(y_test, pred_y_prob)
            pr1, rec1, thr1 = precision_recall_curve(y_test, pred_y_prob)
            auc_score_pr = auc(rec1,pr1)
            print(seed, "\t", dataset_names[i], "\t", model_names[i]+"_"+loss_type,"\t", report_map['1.0']['precision'], "\t", report_map['1.0']['recall'], "\t", report_map['1.0']['f1-score'],"\t",auc_score,"\t",auc_score_pr,"\t",total_time)

    #merge test set
    overall_test_x=x_tests[0].copy()
    overall_test_y=y_tests[0].copy()
    for j in range(1, len(dataset_paths)):
        overall_test_x = np.vstack((overall_test_x, x_tests[i]))
        overall_test_y = np.concatenate((overall_test_y, y_tests[i]), axis=0)
        
        
    if evaluate_base_models:
        #Evaluate base models
        #create the input model
        #print("Building model input...")
        ensemble_data_x = x_ensemble[0].copy()
        ensemble_data_y = y_ensemble[0].copy()
        for j in range(0, len(dataset_paths)):

            base_model_path = ''.join([model_folder_path, model_names[j], "_", loss_type, ".hdf5"])
            b_model=create_dnn_tf_func(len(ensemble_data_x[0]), base_learner_parameters)
            b_model[0].load_weights(base_model_path)

            # evaluation for day - competitor
            for i in range(0, len(dataset_paths)):
                if debug:
                    print("RESULT TEST")

                current_test_x = x_tests[i]
                current_test_y = y_tests[i]

                b_pred_y_prob = b_model[0].predict(current_test_x, verbose=0)
                b_pred_y = np.around(b_pred_y_prob, 0)

                if (debug):
                     print(str(confusion_matrix(current_test_y, b_pred_y)).replace("[", " ").replace("]", " "))
                if (debug):
                     print(classification_report(current_test_y, b_pred_y))

                report_map = classification_report(current_test_y, b_pred_y, output_dict=True)
                auc_score = roc_auc_score(current_test_y, b_pred_y_prob)
                pr1, rec1, thr1 = precision_recall_curve(current_test_y, b_pred_y_prob)
                auc_score_pr = auc(rec1,pr1)
                with open(file_output,'a') as f:
                    result_string=str(seed)+"\t"+dataset_names[i]+"\t"+model_names[j]+"_"+str(loss_type)+"\t"+ str(report_map['1.0']['precision'])+"\t"+str(report_map['1.0']['recall'])+"\t"+str(report_map['1.0']['f1-score'])+"\t"+str(auc_score)+"\t"+str(auc_score_pr)+"\t"+str(total_time)
                    f.write(result_string+'\n')
                print(seed, "\t", dataset_names[i], "\t", model_names[j]+"_"+loss_type,"\t", report_map['1.0']['precision'], "\t", report_map['1.0']['recall'], "\t", report_map['1.0']['f1-score'],"\t",auc_score,"\t",auc_score_pr,"\t",total_time)
            
            
            #evaluate on all the test set    
            b_pred_y_prob = b_model[0].predict(overall_test_x, verbose=0)
            b_pred_y = np.around(b_pred_y_prob, 0)
            if (debug):
                print(str(confusion_matrix(overall_test_y, b_pred_y)).replace("[", " ").replace("]", " "))
            if (debug):
                print(classification_report(overall_test_y, b_pred_y))

            report_map = classification_report(overall_test_y, b_pred_y, output_dict=True)
            auc_score = roc_auc_score(overall_test_y, b_pred_y_prob)
            pr1, rec1, thr1 = precision_recall_curve(overall_test_y, b_pred_y_prob)
            auc_score_pr = auc(rec1,pr1)
            with open(file_output,'a') as f:
                result_string=str(seed)+"\t"+"overall_data"+"\t"+model_names[j]+"_"+str(loss_type)+"\t"+ str(report_map['1.0']['precision'])+"\t"+str(report_map['1.0']['recall'])+"\t"+str(report_map['1.0']['f1-score'])+"\t"+str(auc_score)+"\t"+str(auc_score_pr)+"\t"+str(total_time)
                f.write(result_string+'\n')
            print(seed, "\t","overall_data", "\t", model_names[j]+"_"+loss_type,"\t", report_map['1.0']['precision'], "\t", report_map['1.0']['recall'], "\t", report_map['1.0']['f1-score'],"\t",auc_score,"\t",auc_score_pr,"\t",total_time)

    #create dataset for the validation of the ensemble and of the competitor
    ensemble_data_x = x_ensemble[0].copy()
    ensemble_data_y = y_ensemble[0].copy()
    for i in range(1,len(dataset_paths)):
        ensemble_data_x = np.vstack((ensemble_data_x, x_ensemble[i]))
        ensemble_data_y = np.concatenate((ensemble_data_y, y_ensemble[i]), axis=0)
        if debug:
            print("size ensemble data", len(ensemble_data_x))
            print("size ensemble data y", len(ensemble_data_y))


    if running_competitor:
        #Competitor Model

        #base learner parameters
        competitor_parameters = {}
        competitor_parameters["loss_type"] = loss_type
        competitor_parameters["factors"] = factors
        competitor_parameters["dropout_pcg"] = dropout_pcg
        competitor_parameters["gamma"] = gamma_loss_parameter
        competitor_parameters["alpha"] = alpha_loss_parameter
        competitor_parameters["fn_weight"] = fn_weight
        competitor_parameters["fp_weight"] = fp_weight
        competitor_parameters["embedding_size"] = 48
        competitor_parameters["depth"] = 3

        #for other ensemble-based competitors
        random_forest_size = 5
        boosting_size = 5


        #DNN competitor
        
        # callback list
        competitor_model_path =  ''.join([model_folder_path, "competitor_" ,loss_type , ".hdf5"])
        #save_weights_only = False,
        checkpoint = ModelCheckpoint(competitor_model_path, monitor='macro_f1', verbose=verbose_model_check, save_best_only=True,
                                 save_weights_only=True, mode='max')
        opt = ReduceLROnPlateau(monitor='val_loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
        callbacks_list = [checkpoint, opt]

        #init competitor
        competitor_init = create_dnn_tf_func(len(ensemble_data_x[0]), competitor_parameters)

        #fit
        start=timer()
        competitor_init[0].fit(ensemble_data_x, ensemble_data_y, batch_size=batch_size, epochs=num_epoch, callbacks=callbacks_list,
            verbose=verbose_fit)
        end = timer(); total_time=end-start

        #load best competitor weights
        competitor_init[0].load_weights(competitor_model_path)

        # evaluation for day - competitor
        for i in range(0, len(dataset_paths)):
            if debug:
                print("RESULT TEST")

            current_test_x = x_tests[i]
            current_test_y = y_tests[i]

            competitor_pred_y_prob = competitor_init[0].predict(current_test_x, verbose=0)
            competitor_pred_y = np.around(competitor_pred_y_prob, 0)

            if (debug):
                print(str(confusion_matrix(current_test_y, competitor_pred_y)).replace("[", " ").replace("]", " "))
            if (debug):
                print(classification_report(current_test_y, competitor_pred_y))

            report_map = classification_report(current_test_y, competitor_pred_y, output_dict=True)
            auc_score = roc_auc_score(current_test_y, competitor_pred_y_prob)
            pr1, rec1, thr1 = precision_recall_curve(current_test_y, competitor_pred_y_prob)
            auc_score_pr = auc(rec1,pr1)
            with open(file_output,'a') as f:
                result_string=str(seed)+"\t"+dataset_names[i]+"\t"+"competitor_" + str(loss_type)+ "\t"+str(report_map['1.0']['precision'])+ "\t"+ str(report_map['1.0']['recall'])+ "\t"+str(report_map['1.0']['f1-score'])+"\t"+str( auc_score)+"\t"+str( auc_score_pr)+"\t"+str(total_time)
                f.write(result_string+'\n')                    
                     
            print(seed, "\t", dataset_names[i], "\t",  "competitor_" + loss_type, "\t", report_map['1.0']['precision'], "\t", report_map['1.0']['recall'], "\t", report_map['1.0']['f1-score'], "\t", auc_score,"\t",auc_score_pr,"\t",total_time)
        
        #evaluate on all the test set
        current_test_x = overall_test_x
        current_test_y = overall_test_y

        competitor_pred_y_prob = competitor_init[0].predict(current_test_x, verbose=0)
        competitor_pred_y = np.around(competitor_pred_y_prob, 0)

        if (debug):
            print(str(confusion_matrix(current_test_y, competitor_pred_y)).replace("[", " ").replace("]", " "))
        if (debug):
            print(classification_report(current_test_y, competitor_pred_y))

        report_map = classification_report(current_test_y, competitor_pred_y, output_dict=True)
        auc_score = roc_auc_score(current_test_y, competitor_pred_y_prob)
        pr1, rec1, thr1 = precision_recall_curve(current_test_y, competitor_pred_y_prob)
        auc_score_pr = auc(rec1,pr1)
        with open(file_output,'a') as f:
            result_string=str(seed)+"\t"+"overall_data"+"\t"+"competitor_" + str(loss_type)+ "\t"+str(report_map['1.0']['precision'])+ "\t"+ str(report_map['1.0']['recall'])+ "\t"+str(report_map['1.0']['f1-score'])+"\t"+str( auc_score)+"\t"+str( auc_score_pr)+"\t"+str(total_time)
            f.write(result_string+'\n')                    
                     
        print(seed, "\t", "overall_data", "\t",  "competitor_" + loss_type, "\t", report_map['1.0']['precision'], "\t", report_map['1.0']['recall'], "\t", report_map['1.0']['f1-score'], "\t", auc_score,"\t",auc_score_pr,"\t",total_time)
        
    #adding competitor to the ensemble
    if add_competitor_to_ensemble:
        base_model_list.append(competitor_init[0])


    if running_ensemble:
        #ENSEMBLE CREATION AND EVALUATION

        for ensemble_type in ensemble_types:
            for do_freeze in do_freeze_choose:

                # creating parameters for ensemble
                parameters = {}
                parameters["freeze_base_models"] = True
                parameters["freeze_base_models_partly"] = False
                parameters["ensemble_type"] = ensemble_type
                parameters["factors"] = ens_factors
                parameters["dropout_pcg"] = dropout_pcg
                parameters["loss_type"] = COST_SENSITIVE_LOSS #"OTHER"
                parameters["gamma"] = gamma_loss_parameter
                parameters["alpha"] = alpha_loss_parameter
                parameters["fn_weight"] = fn_weight
                parameters["fp_weight"] = fp_weight
                parameters["do_freeze"] = do_freeze

                #create the input model
                #print("Building model input...")
                
                input_shape = ensemble_data_x.shape[1]
                name_model_input = 'input_ensemble'

                ensemble_input = keras.layers.Input(shape=(input_shape,), name=name_model_input)

                #init ensemble model
                ensemble_model = create_ensemble(base_model_list, parameters)

                #init callback
                # callback list
                if do_freeze:
            	    parfreeze="_Freeze"
                else:
            	    parfreeze="_NOFreeze"
                ensemble_model_path = ''.join([model_folder_path, parameters["ensemble_type"],"_",parameters["loss_type"],parfreeze,".hdf5"])
            
                checkpoint = ModelCheckpoint(ensemble_model_path, monitor='macro_f1', verbose=verbose_model_check, save_best_only=True,
                                         save_weights_only=True, mode='max')
                opt = ReduceLROnPlateau(monitor='loss', mode='min', min_lr=1e-15, patience=3, factor=0.001, verbose=0)
                #lrm = LearningRateMonitor()
                callbacks_list = [checkpoint, opt]

                #fill input
                ensemble_data_inputs_x = [ensemble_data_x for i in range(0,len(base_model_list))]
                
                #fit
                start = timer()
                ensemble_model.fit(ensemble_data_inputs_x, ensemble_data_y, batch_size=batch_size, epochs=num_epoch, callbacks=callbacks_list, verbose=verbose_fit)
                end = timer(); total_time=end-start
                #load weight
                ensemble_model.load_weights(ensemble_model_path)

                if debug:
                    print("*** ENSEMBLE EVALUATION ***")

                #evaluation for day
                for i in range(0, len(dataset_paths)):
                    if debug:
                        print("RESULT TEST")

                    current_test_x = x_tests[i]
                    current_test_y = y_tests[i]

                    input_test_list = [current_test_x for j in range(0,len(base_model_list))]

                    ensemble_pred_y_prob = ensemble_model.predict(input_test_list, verbose=0)
                    ensemble_pred_y = np.around(ensemble_pred_y_prob, 0)

                    if (debug):
                         print(str(confusion_matrix(current_test_y, ensemble_pred_y)).replace("[", " ").replace("]", " "))
                    if (debug):
                         print(classification_report(current_test_y, ensemble_pred_y))

                    report_map = classification_report(current_test_y, ensemble_pred_y, output_dict=True)
                    auc_score = roc_auc_score(current_test_y, ensemble_pred_y_prob)
                    pr1, rec1, thr1 = precision_recall_curve(current_test_y, ensemble_pred_y_prob)
                    auc_score_pr = auc(rec1,pr1)
                    with open(file_output,'a') as f:
                        result_string=str(seed)+"\t"+dataset_names[i]+"\t"+str(parameters["ensemble_type"])+"_"+str(loss_type)+str(parfreeze)+"\t"+str(report_map['1.0']['precision'])+"\t"+str(report_map['1.0']['recall'])+"\t"+str(report_map['1.0']['f1-score'])+"\t"+str(auc_score)+"\t"+str(auc_score_pr)+"\t"+str(total_time)
                        f.write(result_string+'\n')
                    print(seed,"\t", dataset_names[i], "\t",parameters["ensemble_type"]+"_"+loss_type+parfreeze,"\t",report_map['1.0']['precision'],"\t",report_map['1.0']['recall'],"\t",report_map['1.0']['f1-score'],"\t",auc_score,"\t",auc_score_pr,"\t",total_time)
                
                #evaluate for all the test set
                current_test_x = overall_test_x
                current_test_y = overall_test_y

                input_test_list = [current_test_x for j in range(0,len(base_model_list))]

                ensemble_pred_y_prob = ensemble_model.predict(input_test_list, verbose=0)
                ensemble_pred_y = np.around(ensemble_pred_y_prob, 0)

                if (debug):
                    print(str(confusion_matrix(current_test_y, ensemble_pred_y)).replace("[", " ").replace("]", " "))
                if (debug):
                    print(classification_report(current_test_y, ensemble_pred_y))

                report_map = classification_report(current_test_y, ensemble_pred_y, output_dict=True)
                auc_score = roc_auc_score(current_test_y, ensemble_pred_y_prob)
                pr1, rec1, thr1 = precision_recall_curve(current_test_y, ensemble_pred_y_prob)
                auc_score_pr = auc(rec1,pr1)
                with open(file_output,'a') as f:
                    result_string=str(seed)+"\t"+"overall_data"+"\t"+str(parameters["ensemble_type"])+"_"+str(loss_type)+str(parfreeze)+"\t"+str(report_map['1.0']['precision'])+"\t"+str(report_map['1.0']['recall'])+"\t"+str(report_map['1.0']['f1-score'])+"\t"+str(auc_score)+"\t"+str(auc_score_pr)+"\t"+str(total_time)
                    f.write(result_string+'\n')
                print(seed,"\t", "overall_data", "\t",parameters["ensemble_type"]+"_"+loss_type+parfreeze,"\t",report_map['1.0']['precision'],"\t",report_map['1.0']['recall'],"\t",report_map['1.0']['f1-score'],"\t",auc_score,"\t",auc_score_pr,"\t",total_time)

                
                
                
   
    



