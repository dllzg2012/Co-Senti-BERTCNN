# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/5/8 20:04
# @author   :Mo
# @function :extract feature of bert and keras

import codecs
import os

import keras.backend.tensorflow_backend as ktf_keras
import numpy as np
import tensorflow as tf
from keras.layers import Add
from keras.models import Model
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from FeatureProject.bert.layers_keras import NonMaskingLayer
from conf.feature_config import gpu_memory_fraction, config_name, ckpt_name, vocab_file, max_seq_len, layer_indexes
import random
from FeatureProject.distance_text_or_vec import jaccard_similarity_coefficient_distance,pearson_correlation_distance
# 全局使用，使其可以django、flask、tornado等调用
graph = None
model = None


# gpu配置与使用率设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
sess = tf.Session(config=config)
ktf_keras.set_session(sess)

class KerasBertVector():
    def __init__(self):
        self.config_path, self.checkpoint_path, self.dict_path, self.max_seq_len = config_name, ckpt_name, vocab_file, max_seq_len
        # 全局使用，使其可以django、flask、tornado等调用
        global graph
        graph = tf.get_default_graph()
        global model
        model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path,
                                                   seq_len=self.max_seq_len)
        print(model.output)
        print(len(model.layers))
        # lay = model.layers
        #一共104个layer，其中前八层包括token,pos,embed等，
        # 每4层（MultiHeadAttention,Dropout,Add,LayerNormalization）
        # 一共24层
        layer_dict = [7]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 4
            layer_dict.append(layer_0)
        # 输出它本身
        if len(layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight，取得不正确
        elif len(layer_indexes) == 1:
            if layer_indexes[0] in [i+1 for i in range(12)]:
                encoder_layer = model.get_layer(index=layer_dict[layer_indexes[0]]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-2]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            # layer_indexes must be [1,2,3,......12...24]
            # all_layers = [model.get_layer(index=lay).output if lay is not 1 else model.get_layer(index=lay).output[0] for lay in layer_indexes]
            all_layers = [model.get_layer(index=layer_dict[lay-1]).output if lay in [i+1 for i in range(12)]
                          else model.get_layer(index=layer_dict[-1]).output  #如果给出不正确，就默认输出最后一层
                          for lay in layer_indexes]
            print(layer_indexes)
            print(all_layers)
            # 其中layer==1的output是格式不对，第二层输入input是list
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = Add()(all_layers_select)
            print(encoder_layer.shape)
        print("KerasBertEmbedding:")
        print(encoder_layer.shape)
        output_layer = NonMaskingLayer()(encoder_layer)
        model = Model(model.inputs, output_layer)
        # model.summary(120)
        # reader tokenizer
        self.token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)

        self.tokenizer = Tokenizer(self.token_dict)


    def bert_encode(self, texts):
        # 文本预处理
        input_ids = []
        input_masks = []
        input_type_ids = []
        tokens_text_list=[]
        for text in texts:
            text=text.lower()
            text_split=text.split("\t")
            if len(text_split)==1:
                print(text)
                tokens_text = self.tokenizer.tokenize(text)
                tokens_text_list.append(tokens_text)
                input_id, input_type_id = self.tokenizer.encode(first=text, max_len=self.max_seq_len)
            elif len(text_split)==2:
                #print(text_split[0])
                #print(text_split[1])
                tokens_text = self.tokenizer.tokenize(text_split[0],text_split[1])
                tokens_text_list.append(tokens_text)
                input_id, input_type_id = self.tokenizer.encode(first=text_split[0],second=text_split[1], max_len=self.max_seq_len)
            input_mask = [0 if ids == 0 else 1 for ids in input_id]
            input_ids.append(input_id)
            input_type_ids.append(input_type_id)
            input_masks.append(input_mask)

        input_ids = np.array(input_ids)
        input_masks = np.array(input_masks)
        input_type_ids = np.array(input_type_ids)

        # 全局使用，使其可以django、flask、tornado等调用
        with graph.as_default():
            predicts = model.predict([input_ids, input_type_ids], batch_size=1)
        print(predicts.shape)
        for t in range(len(tokens_text_list)):
            sub_list=[]
            for i, token in enumerate(tokens_text_list[t]):
                sub_list.append(predicts[t][i].tolist())
            tokens_text_list[t]=sub_list
                #print(token, [len(predicts[t][i].tolist())], predicts[t][i].tolist())

        # 相当于pool，采用的是https://github.com/terrifyzhao/bert-utils/blob/master/graph.py
        mul_mask = lambda x, m: x * np.expand_dims(m, axis=-1)
        masked_reduce_mean = lambda x, m: np.sum(mul_mask(x, m), axis=1) / (np.sum(m, axis=1, keepdims=True) + 1e-9)

        pools = []
        for i in range(len(predicts)):
            pred = predicts[i]
            masks = input_masks.tolist()
            mask_np = np.array([masks[i]])
            pooled = masked_reduce_mean(pred, mask_np)
            pooled = pooled.tolist()
            pools.append(pooled[0])
        #print('bert:', pools)
        return pools,tokens_text_list

def read_txt(filename):
    fileData = codecs.open(filename, "r", encoding='ascii', errors='ignore')
    readfile = fileData.readlines()
    return readfile
def write_txt(file_name,input_list):
    fdata = open(file_name, "a")
    fdata.write("\n".join(input_list))

def write_single_txt(file_name, input_list):
    fdata = open(file_name, "a")
    fdata.write(input_list)
if __name__ == "__main__":
    cur = '/'.join(os.path.abspath(__file__).split('/')[:-3])
    train_data_path=os.path.join(cur, 'data/senti_pretrain/process_train.txt')
    test_data_path = os.path.join(cur, 'data/senti_pretrain/process_test.txt')
    dev_data_path = os.path.join(cur, 'data/senti_pretrain/process_dev.txt')
    train_embed_path= os.path.join(cur, 'data/senti_pretrain/embed_train.txt')
    test_embed_path = os.path.join(cur, 'data/senti_pretrain/embed_test.txt')
    dev_embed_path = os.path.join(cur, 'data/senti_pretrain/embed_dev.txt')
    bert_vector = KerasBertVector()
    '''
    pooled = bert_vector.bert_encode(['oxycodone [ROXICODONE] 5 mg tablet 0.5-1 tablets by mouth every 4 hours as needed.','Spent 25 minutes with the patient and greater than 50% of this time was spent counseling the patient regarding diagnosis and available treatment options.'])
    print(len(pooled))
    score=pearson_correlation_distance(pooled[0],pooled[1])
    print(score)

    while True:
        print("input:")
        ques = input()
        print(bert_vector.bert_encode([ques]))
    '''
    def write_clinical_data():
        input_data=read_txt(dev_data_path)
        for input in input_data:
            input_split = input.strip().split("\t")
            if len(input_split)==2:
                sentence = input_split[1]
                label = input_split[0]
                _,token_text_list=bert_vector.bert_encode([sentence])
                sentence_embed_list=[]
                for word1 in token_text_list[0]:
                    word1=[str(char).strip() for char in word1]
                    sentence_embed_list.append(" ".join(list(word1)))
                write_string='||'.join(sentence_embed_list)+"\t"+label+"\n"
                print('writing:',write_string)
                write_single_txt(dev_embed_path,write_string)
    write_clinical_data()
    '''
    sick_train_data_path = os.path.join(cur, 'sick_data/sick_train/SICK_train.txt')
    sick_test_data_path = os.path.join(cur, 'sick_data/sick_test_annotated/SICK_test_annotated.txt')
    sick_train_embed_path = os.path.join(cur, 'sick_data/sick_train/SICK_train_embedding.txt')
    sick_test_embed_path = os.path.join(cur, 'sick_data/sick_test_annotated/SICK_test_annotated_embedding_300.txt')

    def write_sick_data():
        input_data = read_txt(sick_test_data_path)
        input_data_train=read_txt(sick_train_data_path)
        input_data=input_data+input_data_train
        random.shuffle(input_data)
        for input in input_data:
            input_split=input.split('\t')
            if len(input_split)==5:
                sentence1 = input_split[1]
                sentence2 = input_split[2]
                score = input_split[3]
                _,token_text_list=bert_vector.bert_encode([sentence1,sentence2])
                sentence1_embed_list=[]
                for word1 in token_text_list[0]:
                    word1=[str(char).strip() for char in word1]
                    sentence1_embed_list.append(" ".join(list(word1)))
                sentence2_embed_list=[]
                for word2 in token_text_list[1]:
                    word2 = [str(char).strip() for char in word2]
                    sentence2_embed_list.append(" ".join(word2))
                write_string='||'.join(sentence1_embed_list)+"\t"+'||'.join(sentence2_embed_list)+"\t"+score+"\n"
                write_single_txt(sick_test_embed_path,write_string)
    #write_sick_data()
    '''

