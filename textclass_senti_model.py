#!/usr/bin/python
# coding=utf8

"""
# Created : 2018/12/28
# Version : python2.7
# Author  : yibo.li
# File    : bert_model.py
# Desc    :
"""

import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import codecs
from bert import modeling
from bert import modeling_anda
from bert import optimization
from bert.data_loader import *
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import Lambda,Bidirectional,LSTM,GRU,Dropout
from Transformer import MyTransformer


processors = {"psb":PsbProcessor,"semm":SemmProcessor}


tf.logging.set_verbosity(tf.logging.INFO)




class BertModel():
    def __init__(self, bert_config,senti_bert_config, num_labels, seq_length, init_checkpoint,senti_init_checkpoint):
        self.bert_config = bert_config
        self.senti_bert_config = senti_bert_config
        self.num_labels = num_labels
        self.seq_length = seq_length

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.senti_score=tf.placeholder(tf.float32,[None,self.seq_length], name='senti_score')

        self.senti_input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='senti_input_ids')
        self.senti_input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='senti_input_mask')
        self.senti_segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='senti_segment_ids')
        self.senti_labels = tf.placeholder(tf.int32, [None], name='senti_labels')

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        '''
        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids)
        self.senti_model = modeling.BertModel(
            config=self.senti_bert_config,
            is_training=self.is_training,
            input_ids=self.senti_input_ids,
            input_mask=self.senti_input_mask,
            token_type_ids=self.senti_segment_ids)
        '''
        self.model = modeling_anda.BertModel(
            config=self.bert_config,
            senti_score=self.senti_score,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            insert_figure=7)
        self.senti_model = modeling.BertModel(
            config=self.senti_bert_config,
            is_training=self.is_training,
            input_ids=self.senti_input_ids,
            input_mask=self.senti_input_mask,
            token_type_ids=self.senti_segment_ids)


        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)





        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        self.inference()


    def apply_dropout_last_layer(self,output_layer):
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.8)
        return output_layer

    def not_apply_dropout(self,output_layer):
        return output_layer

    def cnn_layer(self,inputs):
        with tf.name_scope('conv1'):
            conv2 = tf.layers.conv1d(inputs, 32, 2,
                                     padding='valid', activation=tf.nn.relu,
                                     kernel_regularizer=self.regularizer)
            conv3=tf.layers.conv1d(inputs, 32, 3,
                                    padding='valid', activation=tf.nn.relu,
                                    kernel_regularizer=self.regularizer)
            conv4 = tf.layers.conv1d(inputs, 32, 4,
                                     padding='valid', activation=tf.nn.relu,
                                     kernel_regularizer=self.regularizer)

            conv=tf.concat([conv2,conv3,conv4],axis=1)
        with tf.name_scope('drop'):
            drop_out = tf.cond(self.is_training, lambda: self.apply_dropout_last_layer(conv),
                               lambda: self.not_apply_dropout(conv))


        return drop_out
    def biRNN_layer(self,method,inputs):
        output_layer = inputs
        with tf.name_scope('biRNN'):
            if method == 'bilstm':
                output_layer = Bidirectional(LSTM(64, return_sequences=False))(inputs)
            elif method == 'gru':
                output_layer = Bidirectional(GRU(64, return_sequences=False))(inputs)
        with tf.name_scope('drop'):

            drop_out = tf.cond(self.is_training, lambda: self.apply_dropout_last_layer(output_layer),
                                   lambda: self.not_apply_dropout(output_layer))

        return drop_out


    def inference(self):

        #output_layer = self.model.get_pooled_output()

        with tf.device('/device:GPU:0'):
            output_layer=self.model.get_sequence_output()



            if use_senti_score:
                senti_score = tf.expand_dims(self.senti_score, 2)
                print(output_layer.get_shape())
                print(senti_score.get_shape())
                '''
                output_layer = tf.split(output_layer, output_layer.get_shape()[1], 1)[0]
                print(output_layer.get_shape())
                output_layer=tf.einsum('aij,ajk->aik',senti_score,output_layer)
                print(output_layer.get_shape())
                '''
                output_layer = output_layer * senti_score

            sentence_layer = tf.split(output_layer, output_layer.get_shape()[1], 1)
            sentence_layer1 = tf.split(output_layer, output_layer.get_shape()[1], 1)[1]
            # sentence_layer2 = tf.split(output_layer, output_layer.get_shape()[1], 1)[max_len + 2]
            for word in sentence_layer[2:output_layer.get_shape()[1]]:
                sentence_layer1 = tf.concat([sentence_layer1, word], 1)

        with tf.device('/device:GPU:1'):
            #loading seti bert model
            senti_tvars = tf.trainable_variables()
            if senti_init_checkpoint:
                (senti_assignment_map, senti_initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(senti_tvars, senti_init_checkpoint)
            #tf.train.init_from_checkpoint(senti_init_checkpoint, senti_assignment_map)
            senti_output_layer=self.senti_model.get_sequence_output()





        #output_layer=(output_layer+senti_output_layer)/2

        #output_layer = self.biRNN_layer('bilstm', output_layer)
        #senti_output_layer = self.biRNN_layer('bilstm', senti_output_layer)

        output_layer = tf.split(output_layer, output_layer.get_shape()[1], 1)[0]
        senti_output_layer = tf.split(senti_output_layer, senti_output_layer.get_shape()[1], 1)[0]
        output_layer=tf.concat([output_layer,senti_output_layer],1)


        #output_layer = self.biRNN_layer('bilstm',output_layer)#self.cnn_layer(output_layer)
        #output_layer1 = self.cnn_layer(sentence_layer1)

        output_layer = tf.reduce_mean(output_layer, axis=1, keep_dims=False)
        #output_layer1 = tf.reduce_mean(output_layer1, axis=1, keep_dims=False)
        #output_layer = tf.concat([output_layer, output_layer1], 1)

        print(output_layer.get_shape())
        with tf.variable_scope("loss"):

            output_layer = tf.cond(self.is_training, lambda: self.apply_dropout_last_layer(output_layer),
                               lambda: self.not_apply_dropout(output_layer))

            self.logits = tf.layers.dense(output_layer, self.num_labels, name='fc')



            one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)
            if use_balence_loss:
                const = tf.constant([[float(1/(1+balence_integer))], [float(1-float(1/(1+balence_integer)))]], dtype=tf.float32)
                cross_entropy=tf.transpose(tf.matmul(-tf.log(tf.nn.softmax(self.logits))*one_hot_labels,const))
                cross_entropy=tf.reshape(cross_entropy,[-1])
            else:
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_labels)

            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

            self.loss = tf.reduce_mean(cross_entropy, name="loss")


            self.optim = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")




def read_txt(file_path):
    fileData=codecs.open(file_path,"r",encoding='ascii',errors='ignore')
    readfile=fileData.readlines()
    return readfile
def write_txt(input_list,file_name):
    fdata = open(file_name, "a")
    fdata.write("\n".join(input_list)+'\n')
def get_senti_dict(senti_file):
    senti_lines=read_txt(senti_file)
    senti_dict={}
    for senti_line in senti_lines:
        senti_split=senti_line.split('\t')
        if len(senti_split)==6 and senti_split[0]=='a':
            senti_word_list=[senti_word.split('#')[0] for senti_word in senti_split[4].split(' ')]
            for word in senti_word_list:
                senti_dict[word]=[float(senti_split[2]),float(senti_split[3])]
    return senti_dict
def process_senti(senti_dict,text_list):
    senti_score_list=[]
    #pos_score_list=[]
    for word in text_list:
        word=word.lower()
        if word in senti_dict.keys() and float(senti_dict[word][1])>=float(senti_dict[word][0]):
            senti_score_list.append(float(1+senti_dict[word][1]))
            #pos_score_list.append(senti_dict[word][0])
            #pos_score_list.append(0)
        elif word in senti_dict.keys() and float(senti_dict[word][1])<float(senti_dict[word][0]):
            senti_score_list.append(float(1-senti_dict[word][0]))
        else:
            #senti_score_list.append(random.random()/100)
            senti_score_list.append(float(1))
            #pos_score_list.append(0)
    '''
    new_senti_score_list=[]
    for senti_score in senti_score_list:
        if float(senti_score)>0.0:
            new_senti_score_list.append(float(senti_score))
    new_senti_score_list.append(0)

    sentence_score=[float(np.mean(pos_score_list)),float(np.mean(new_senti_score_list))]
    '''
    return senti_score_list



def senti_make_tf_record(output_dir, data_dir, vocab_file):
    tf.gfile.MakeDirs(output_dir)
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    train_file = os.path.join(output_dir, "senti_train.tf_record")
    eval_file = os.path.join(output_dir, "senti_eval.tf_record")

    # save data to tf_record
    if use_co:
        train_examples = processor.get_co_train_examples(data_dir,co_data_dir,co_type)
    else:
        train_examples = processor.get_train_examples(data_dir)
    train_senti_score_list=[]
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, train_file,train_senti_score_list,use_mirror)
    # eval data
    if use_co:
        eval_examples = processor.get_co_dev_examples(data_dir,co_data_dir,co_type)
    else:
        eval_examples = processor.get_dev_examples(data_dir)
    eval_senti_score_list = []
    file_based_convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, eval_file,eval_senti_score_list,use_mirror)

    del train_examples, eval_examples




def make_tf_record(output_dir, data_dir, vocab_file,senti_file):
    tf.gfile.MakeDirs(output_dir)
    processor = processors[task_name]()
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")
    senti_dict = get_senti_dict(senti_file)
    # save data to tf_record
    if use_co:
        train_examples = processor.get_co_train_examples(data_dir,co_data_dir,co_type)
    else:
        train_examples = processor.get_train_examples(data_dir)
    train_senti_score_list=[]
    for t in range(len(train_examples)):
        text_list=train_examples[t].text_a.split(" ")
        train_senti_score_list.append(process_senti(senti_dict,text_list))
    file_based_convert_examples_to_features(
        train_examples, label_list, max_seq_length, tokenizer, train_file,train_senti_score_list,use_mirror)
    # eval data
    if use_co:
        eval_examples = processor.get_co_dev_examples(data_dir,co_data_dir,co_type)
    else:
        eval_examples = processor.get_dev_examples(data_dir)
    eval_senti_score_list = []
    for e in range(len(eval_examples)):
        eval_text_list=eval_examples[e].text_a.split(" ")
        eval_senti_score_list.append(process_senti(senti_dict,eval_text_list))
    file_based_convert_examples_to_features(
        eval_examples, label_list, max_seq_length, tokenizer, eval_file,eval_senti_score_list,use_mirror)

    del train_examples, eval_examples


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_data(data, batch_size, is_training, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "senti_score":tf.FixedLenFeature([max_seq_length], tf.float32)
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=50000)
        data = data.repeat(num_epochs)


    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data


def get_test_example(senti_file):
    processor = processors[task_name]()

    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    senti_tokenizer = tokenization.FullTokenizer(vocab_file=senti_vocab_file)

    # save data to tf_record
    if use_co:
        examples = processor.get_co_test_examples(data_dir,co_data_dir,co_type)
    else:
        examples = processor.get_test_examples(data_dir)
    senti_dict = get_senti_dict(senti_file)

    test_senti_score_list=[]
    for t in range(len(examples)):
        text_list=examples[t].text_a.split(" ")
        test_senti_score_list.append(process_senti(senti_dict,text_list))

    features = get_test_features(examples, label_list, max_seq_length, tokenizer,test_senti_score_list,use_mirror)
    senti_features = get_test_features(examples, label_list, max_seq_length, senti_tokenizer, test_senti_score_list,use_mirror)

    return features,senti_features



def evaluate(sess, model):
    """
    评估 val data 的准确率和损失
    """

    # dev data
    test_record = tf.data.TFRecordDataset(os.path.join(output_dir, "eval.tf_record"))
    senti_test_record = tf.data.TFRecordDataset(os.path.join(output_dir, "senti_eval.tf_record"))

    test_data = read_data(test_record, train_batch_size, False, num_train_epochs)
    senti_test_data = read_data(senti_test_record, train_batch_size, False, num_train_epochs)

    test_iterator = test_data.make_one_shot_iterator()
    senti_test_iterator = senti_test_data.make_one_shot_iterator()

    test_batch = test_iterator.get_next()
    senti_test_batch = senti_test_iterator.get_next()

    data_nums = 0
    total_loss = 0.0
    total_acc = 0.0

    while True:
        try:
            features = sess.run(test_batch)
            senti_features = sess.run(senti_test_batch)

            feed_dict = {model.input_ids: features["input_ids"],
                         model.input_mask: features["input_mask"],
                         model.segment_ids: features["segment_ids"],
                         model.labels: features["label_ids"],
                         model.senti_score: features["senti_score"],
                         model.senti_input_ids: senti_features["input_ids"],
                         model.senti_input_mask: senti_features["input_mask"],
                         model.senti_segment_ids: senti_features["segment_ids"],
                         model.senti_labels: senti_features["label_ids"],
                         model.is_training: False,
                         model.learning_rate: learning_rate}
            '''
            feed_dict = {model.input_x: features["senti_score"],
                         model.input_y: features["label_ids"],
                         model.drop_prob: 0,
                         model.learning_rate: learning_rate}
            '''
            batch_len = len(features["input_ids"])
            data_nums += batch_len
            # print(data_nums)

            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        except Exception as e:
            print(e)
            break

    return total_loss / data_nums, total_acc / data_nums


def main():
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    senti_bert_config= modeling.BertConfig.from_json_file(senti_bert_config_file)
    with tf.Graph().as_default():
        # train data

        train_record = tf.data.TFRecordDataset(os.path.join(output_dir, "train.tf_record"))
        senti_train_record = tf.data.TFRecordDataset(os.path.join(output_dir, "senti_train.tf_record"))

        train_data = read_data(train_record, train_batch_size, True, num_train_epochs)
        senti_train_data = read_data(senti_train_record, train_batch_size, True, num_train_epochs)

        train_iterator = train_data.make_one_shot_iterator()
        senti_train_iterator = senti_train_data.make_one_shot_iterator()

        model = BertModel(bert_config,senti_bert_config, num_labels, max_seq_length, init_checkpoint,senti_init_checkpoint)
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        best_acc_val = 0.0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch = train_iterator.get_next()
            senti_train_batch = senti_train_iterator.get_next()
            while True:
                try:
                    train_steps += 1
                    features = sess.run(train_batch)
                    senti_features = sess.run(senti_train_batch)
                    batch_len = len(features["input_ids"])
                    feed_dict = {model.input_ids: features["input_ids"],
                                 model.input_mask: features["input_mask"],
                                 model.segment_ids: features["segment_ids"],
                                 model.labels: features["label_ids"],
                                 model.senti_score:features["senti_score"],
                                 model.senti_input_ids: senti_features["input_ids"],
                                 model.senti_input_mask: senti_features["input_mask"],
                                 model.senti_segment_ids: senti_features["segment_ids"],
                                 model.senti_labels: senti_features["label_ids"],
                                 model.is_training: True,
                                 model.learning_rate: learning_rate}
                    _, train_loss, train_acc = sess.run([model.optim, model.loss, model.acc],
                                                        feed_dict=feed_dict)

                    if train_steps % 50 == 0 or batch_len < train_batch_size:
                        val_loss, val_acc = evaluate(sess, model)
                        saver.save(sess, output_dir+'/model', global_step=train_steps)
                        test(senti_dict_file)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))
                except Exception as e:
                    print(e)
                    break
def compute_F1(label_list,pred_list):
    TP_list=[]
    TN_list = []
    FP_list = []
    FN_list = []
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1 = 0
    for i in range(len(label_list)):
        if int(label_list[i])==int(pred_list[i])==1:
            TP_list.append((label_list[i],pred_list[i]))
        elif int(label_list[i])==int(pred_list[i])==0:
            TN_list.append((label_list[i],pred_list[i]))
        elif int(label_list[i])==0 and int(pred_list[i])==1:
            FP_list.append((label_list[i], pred_list[i]))
        elif int(label_list[i])==1 and int(pred_list[i])==0:
            FN_list.append((label_list[i], pred_list[i]))
    TP=len(TP_list)
    TN=len(TN_list)
    FP=len(FP_list)
    FN=len(FN_list)
    try:
        Accuracy=float((TP+TN)/(TP+TN+FP+FN))
        Precision=float(TP/(TP+FP))
        Recall=float(TP/(TP+FN))
        F1=float(2*(Precision*Recall)/(Precision+Recall))
    except ZeroDivisionError as e:
        print('division by zero')
    return Accuracy,F1,Precision,Recall
def str_int_list(str_list):
    int_list=[]
    for label in str_list:
        int_list.append(int(label))
    return int_list
def test_model(sess, graph, features,senti_features):
    """

    :param sess:
    :param graph:
    :param features:
    :return:
    """

    total_loss = 0.0
    total_acc = 0.0

    input_ids = graph.get_operation_by_name('input_ids').outputs[0]
    input_mask = graph.get_operation_by_name('input_mask').outputs[0]
    segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
    labels = graph.get_operation_by_name('labels').outputs[0]
    senti_input_ids = graph.get_operation_by_name('senti_input_ids').outputs[0]
    senti_input_mask = graph.get_operation_by_name('senti_input_mask').outputs[0]
    senti_segment_ids = graph.get_operation_by_name('senti_segment_ids').outputs[0]
    senti_labels = graph.get_operation_by_name('senti_labels').outputs[0]
    senti_score=graph.get_operation_by_name('senti_score').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]

    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]
    logits = graph.get_operation_by_name('loss/pred').outputs[0]
    data_len = len(features)

    batch_size = train_batch_size
    num_batch = int((len(features) - 1) / batch_size) + 1
    y_preds = []
    y_label=[]
    y_senti_label = []
    for i in range(num_batch):
        print(i)
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)

        batch_len = end_index-start_index

        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        _labels = np.array([data.label_id for data in features[start_index:end_index]])
        _senti_score=np.array([data.senti_score for data in features[start_index:end_index]])

        _senti_input_ids = np.array([data.input_ids for data in senti_features[start_index:end_index]])
        _senti_input_mask = np.array([data.input_mask for data in senti_features[start_index:end_index]])
        _senti_segment_ids = np.array([data.segment_ids for data in senti_features[start_index:end_index]])
        _senti_labels = np.array([data.label_id for data in senti_features[start_index:end_index]])
        y_label.extend(_labels.tolist())
        y_senti_label.extend(_labels.tolist())

        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     labels: _labels,
                     senti_input_ids: _senti_input_ids,
                     senti_input_mask: _senti_input_mask,
                     senti_segment_ids: _senti_segment_ids,
                     senti_labels: _senti_labels,
                     senti_score:_senti_score,
                     is_training: False}

        test_loss, test_acc,test_pred = sess.run([loss, acc,logits], feed_dict=feed_dict)
        total_loss += test_loss * batch_len
        total_acc+=test_acc*batch_len
        y_batch_pred=test_pred

        y_preds.extend(y_batch_pred)

    print('pred_list:',y_preds)
    print('label_list:',y_label)
    Accuracy,F1_score,precision,recall=compute_F1(y_label,y_preds)
    int_label_list=str_int_list(y_label)
    int_predict_list=str_int_list(y_preds)
    fpr, tpr, thresholds = roc_curve(int_label_list, int_predict_list, pos_label=1)
    auc_score = auc(fpr, tpr)
    print("Accuracy:",Accuracy)
    return total_loss / data_len, total_acc / data_len,F1_score,precision,recall,auc_score,y_preds,y_label,Accuracy


def get_int(s):
    l = len(s)
    numbers = []
    i = 0
    while i < l:
        num = ''
        symbol = s[i]
        while '0' <= symbol <= '9':  # symbol.isdigit()
            num += symbol
            i += 1
            if i < l:
                symbol = s[i]
            else:
                break
        i += 1
        if num != '' and num!= '000000' and num!= '000001':
            numbers.append(int(num))
    if len(numbers)>0:
        return max(numbers)
    else:
        return -1
def test(senti_file):
    features,senti_feature = get_test_example(senti_file)
    file_name_list = os.listdir(output_dir)
    flag_list = []
    for file_name in file_name_list:
        flag = get_int(file_name)
        flag_list.append(flag)
    maxflag = max(flag_list)
    # for file_name in file_name_list:
    outfile_name = "model-" + str(maxflag) + ".meta"

    graph_path = os.path.join(output_dir, outfile_name)
    model_path = output_dir
    graph = tf.Graph()
    saver = tf.train.import_meta_graph(graph_path, graph=graph)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph,config=config)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    test_loss, test_acc ,F1_score,precision,recall,auc_score,y_preds,y_label,Accuracy= test_model(sess, graph, features,senti_feature)
    print("Test loss: %f, Test acc: %f" % (test_loss, test_acc))
    print("Test Precision: %f, Test Recall: %f, Test F1: %f,AUC %f" % (precision,recall,F1_score,auc_score))
    write_list=[]
    pred_list=[]
    label_list=[]
    for pred in y_preds:
        pred_list.append(str(pred))
    for label in y_label:
        label_list.append(str(label))
    write_list.append("pred_list:"+','.join(pred_list))
    write_list.append("label_list:" + ','.join(label_list))
    write_list.append("Test loss:"+str(test_loss)+"Test acc:"+str(test_acc))
    write_list.append("Test Precision:"+str(precision)+',Test Recall:'+str(recall)+'Test F1:'+str(F1_score)+"AUC:"+str(auc_score))
    write_txt(write_list,result_file)



if __name__ == "__main__":
    task_name = "semm"
    use_co=True
    use_mirror=''#pair或add
    co_type='sentence'
    use_balence_loss=True
    use_senti_score=False
    balence_integer=4
    bert_type='uncased_L-12_H-768_A-12'
    #bert_type = 'wwm_uncased_L-24_H-1024_A-16'

    bi_bert_type='senti'
    data_dir = "data/"+task_name
    co_data_dir="data/co_"+task_name
    #data_dir = "data/" + task_name + '/5_cross/5'
    output_dir="model/doublebert"
    if use_co:
        output_dir = output_dir+'/'+bert_type + "_" +task_name+"_co"
    else:
        output_dir = output_dir+'/'+bert_type + "_" +task_name
    if use_balence_loss:
        output_dir=output_dir+"_bloss"
    if use_senti_score:
        output_dir = output_dir + "_sentiscore"
    #output_dir=output_dir+'_5*'
    #output_dir=output_dir+'_'+str(balence_integer)bert_model.ckpt
    result_file = output_dir+"/result.txt"
    vocab_file = "./bert/"+bert_type+"/vocab.txt"
    bert_config_file = "./bert/"+bert_type+"/bert_config.json"
    init_checkpoint = "./bert/"+bert_type+"/model.ckpt-100000"
    senti_dict_file="./SentiWordNet_3.0.0_20130122.txt"
    #co_data_dir='./data/co_psb_175'
    senti_vocab_file = "./bert/"+bi_bert_type+"/vocab.txt"
    senti_bert_config_file = "./bert/"+bi_bert_type+"/bert_config.json"
    senti_init_checkpoint = "./bert/"+bi_bert_type+"/model.ckpt-100000"
    max_seq_length = 84
    learning_rate = 1e-5
    train_batch_size = 16
    num_train_epochs = 5
    num_labels = 2
    senti_make_tf_record(output_dir,data_dir,senti_vocab_file)
    make_tf_record(output_dir, data_dir, vocab_file,senti_dict_file)
    main()
    test(senti_dict_file)

