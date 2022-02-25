import os
import re
import json
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import utils
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import time


def json2csv_batch(input_file, feat_num=21):
    with open(input_file, 'r') as json_file:
        json_list = list(json_file)
    # fout = open(output_file,'w')
    j = 0
    n = len(json_list)
    features = np.zeros((n, 768))
    for json_str in json_list:
        tokens = json.loads(json_str)["features"]
        i = 0
        for token in tokens:
            if token['token'] in ['[CLS]', '[SEP]']:
                continue
            else:
                i = i + 1
                if i == feat_num:
                    last_layers = np.sum([
                        token['layers'][0]['values'],
                        token['layers'][1]['values'],
                        token['layers'][2]['values'],
                        token['layers'][3]['values'],
                        token['layers'][4]['values'],
                        token['layers'][5]['values'],
                        token['layers'][6]['values'],
                        token['layers'][7]['values'],
                        token['layers'][8]['values'],
                        token['layers'][9]['values'],
                        token['layers'][10]['values'],
                        token['layers'][11]['values'],
                    ], axis=0)
                    features[j, :] = last_layers
                    j = j + 1
    return features
    # fout.write(f'{",".join(["{:f}".format(i) for i in last_layers])}\n')


def read_seqs(file_path, seq_length=510):
    # pos_data
    data = re.split(
        r'(^>.*)', ''.join(open(file_path).readlines()), flags=re.M)
    n_seq = (len(data) - 1) // 2
    seq_list = []
    #1. extract pos seq
    print("================step1: extract pos seq============")
    for i in range(2, len(data), 2):
        fid = data[i - 1][1:].split('|')[0]
        # nseq = nseq + 1
        fasta = list(data[i].replace('\n', '').replace('\x1a', ''))
        seq = [' '.join(fasta[j:j + seq_length])
               for j in range(0, len(fasta) + 1, seq_length)]
        seq_list.append(seq[0] + '\n')
    return n_seq, seq_list


def extract_seq_batch(train_pos_file_path, train_neg_file_path, ind_pos_file_path, ind_neg_file_path, train_output_file_path, ind_output_file_path, seq_length=510):

    n_train_pos, train_pos_seq_list = read_seqs(train_pos_file_path)
    n_train_neg, train_neg_seq_list = read_seqs(train_neg_file_path)

    n_ind_pos, ind_pos_seq_list = read_seqs(ind_pos_file_path)
    n_ind_neg, ind_neg_seq_list = read_seqs(ind_neg_file_path)

    train_nseq = n_train_pos+ n_train_neg
    ind_nseq = n_ind_pos + n_ind_neg
    train_features = np.zeros((train_nseq, 769))
    ind_features = np.zeros((ind_nseq, 769))
    seq_list = train_pos_seq_list + train_neg_seq_list + ind_pos_seq_list + ind_neg_seq_list
    train_features[0:n_train_pos, 0] = 1
    train_features[n_train_pos:, 0] = 0

    ind_features[0:n_ind_pos, 0] = 1
    ind_features[n_ind_pos:, 0] = 0

    # n_seq = train_nseq + ind_nseq

    features = np.vstack((train_features,ind_features))

    with open("seq.txt", "w") as fw:
        fw.writelines(seq_list)
    #.bert2json
    print("=================step3: bert2json=================")
    cmd = "python ./bert/extract_features.py  --input_file=seq.txt --output_file=seq.jsonl --vocab_file=./vocab.txt --bert_config_file=./bert_config.json --init_checkpoint=./bert_model.ckpt.index --do_lower_case=False --layers=-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12 --max_seq_length=512 --batch_size=64"
    os.system(cmd)
    #json2csv
    print("=================step4: json2csv=================")
    features[:, 1:] = json2csv_batch("seq.jsonl", 21)

    train_features = features[0:train_nseq, :]
    ind_features = features[train_nseq:, :]
    train_df = pd.DataFrame(train_features)
    ind_df = pd.DataFrame(ind_features)
    train_df.to_csv(train_output_file_path, header=False, index=False, float_format='%.5f')
    ind_df.to_csv(ind_output_file_path, header=False, index=False, float_format='%.5f')

def d_cnn_model(input_length):
    nb_classes = 2
    model = Sequential()

    model.add(Dropout(0.2, input_shape=(input_length, 1)))
    model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Conv1D(32, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu'))
    # # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(128, 3, activation='relu'))
    # # model.add(Dropout(0.5))
    model.add(MaxPooling1D(2))

    #     model.add(Conv1D(256, 3, activation='relu'))
    #     # # model.add(Dropout(0.5))
    #     model.add(MaxPooling1D(2))

    #     model.add(Conv1D(512, 3, activation='relu'))
    #     # # model.add(Dropout(0.5))
    #     model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def calculate_CNN_binary_result(X_trn_bert, y_trn):
    num_features = 768
    num_epochs = 20
    nb_classes = 2
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train, test in kfold.split(X_trn_bert, y_trn):
        model = d_cnn_model(num_features)
        ## evaluate the model
        trn_new = np.asarray(X_trn_bert.iloc[train])
        tst_new = np.asarray(X_trn_bert.iloc[test])
        # trn_new = X_trn_bert.iloc[train]
        # tst_new = X_trn_bert.iloc[test]
        y_train = y_trn.iloc[train]
        y_test = y_trn.iloc[test]
        model.fit(trn_new.reshape(len(trn_new), num_features, 1),
                  utils.to_categorical(y_train, nb_classes),
                  epochs=num_epochs, batch_size=10, verbose=0, class_weight='auto')
        # evaluate the model
        true_labels = np.asarray(y_trn.iloc[test])
        predictions = model.predict(tst_new.reshape(len(tst_new), num_features, 1))
        pred_label = np.int64(predictions[:, 1] > 0.5)
        acc_cv_scores.append(accuracy_score(true_labels, pred_label))
        # print(confusion_matrix(true_labels, predictions))
        newTN, newFP, newFN, newTP = confusion_matrix(true_labels, pred_label).ravel()
        TP += newTP
        FN += newFN
        FP += newFP
        TN += newTN
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions[:, 1], pos_label=1)
        auc_cv_scores.append(metrics.auc(fpr, tpr))

    print('\nFeature: ', "bert")
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))


def calculate_CNN_binary_independent(X_train, y_trn, X_test, y_test):
    num_features = 768
    nb_classes = 2
    num_epochs = 20
    TP = FP = TN = FN = 0
    acc_cv_scores = []
    auc_cv_scores = []
    model = d_cnn_model(num_features)

    trn_new = np.asarray(X_train)
    tst_new = np.asarray(X_test)
    # trn_new = X_trn_bert.iloc[train]
    # tst_new = X_trn_bert.iloc[test]
    y_train = np.asarray(y_trn)
    y_test = np.asarray(y_test)
    model.fit(trn_new.reshape(len(trn_new), num_features, 1),
              utils.to_categorical(y_train, nb_classes),
              epochs=num_epochs, batch_size=10, verbose=0, class_weight='auto')
    # evaluate the model
    true_labels = np.asarray(y_test)
    predictions = model.predict(tst_new.reshape(len(tst_new), num_features, 1))
    pred_label = np.int64(predictions[:, 1] > 0.5)
    acc_cv_scores.append(accuracy_score(true_labels, pred_label))
    # print(confusion_matrix(true_labels, predictions))
    newTN, newFP, newFN, newTP = confusion_matrix(true_labels, pred_label).ravel()
    TP += newTP
    FN += newFN
    FP += newFP
    TN += newTN
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions[:, 1], pos_label=1)
    auc_cv_scores.append(metrics.auc(fpr, tpr))

    print('\nFeature: ', "bert")
    print('Accuracy = ', np.mean(acc_cv_scores))
    print('TP = %s, FP = %s, TN = %s, FN = %s' % (TP, FP, TN, FN))
    print('AUC = ', np.mean(auc_cv_scores))

if __name__ == "__main__":
    full_path = os.path.realpath(__file__)
    os.chdir(os.path.dirname(full_path))

    print(f"Change CWD to: {os.path.dirname(full_path)}")
    start_time = time.time()
    #1.train and test
    data_dir = './test_data/'
    tissue='h_b/'

    train_neg_file_path = data_dir+tissue+'train_negative.txt'
    train_pos_file_path = data_dir+tissue+'train_positive.txt'
    train_output_file_path = data_dir+tissue+"m6A.bert.cv.csv"

    ind_neg_file_path = data_dir+tissue+'independent_negative.txt'
    ind_pos_file_path = data_dir+tissue+'independent_positive.txt'
    ind_output_file_path = data_dir+tissue+"m6A.bert.ind.csv"



    print("=====================extract features================================")
    extract_seq_batch(ind_pos_file_path, ind_neg_file_path, train_pos_file_path, train_neg_file_path, ind_output_file_path, train_output_file_path)

    print("=====================train model================================")
    data_dir = './'
    df_train = pd.read_csv(os.path.join(data_dir, train_output_file_path), header=None)
    df_test = pd.read_csv(os.path.join(data_dir, ind_output_file_path), header=None)
    X_trn_bert = df_train.iloc[:, 1:]
    y_trn_bert = df_train[0]
    X_trn_bert2 = (X_trn_bert - X_trn_bert.mean()) / X_trn_bert.std()

    X_tst_bert = df_test.iloc[:, 1:]
    y_tst_bert = df_test[0]
    X_tst_bert2 = (X_tst_bert - X_tst_bert.mean()) / X_tst_bert.std()

    calculate_CNN_binary_result(X_trn_bert2, y_trn_bert)
    calculate_CNN_binary_independent(X_trn_bert2, y_trn_bert, X_tst_bert2, y_tst_bert)
    finish_time = time.time()
    print(f"the running time is:{finish_time-start_time} s")