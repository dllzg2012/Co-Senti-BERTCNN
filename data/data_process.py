import codecs
import csv
import random
import difflib
from data import co_adr_drug_dataprocess as add
def read_txt(file_path):
    dataFile=codecs.open(file_path,'r',encoding='ascii',errors='ignore')
    readData=dataFile.readlines()
    return readData
def write_txt(input_list,file_name):
    fdata = open(file_name, "a")
    fdata.write("\n".join(input_list)+'\n')
def write_csv(input_list,file_path):
    with open(file_path,'w',newline='') as f:
        file_write=csv.writer(f)
        for input in input_list:
            file_write.writerow(input)
        f.close()
def read_csv(file_path):
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        out_list = []
        for line in reader:
            out_list.append(line[1] + '\t' + line[2])
        return out_list
def get_train_data():
    lines=read_txt('./cadr/train.txt')
    train_list=[]
    for i,line in enumerate(lines):
        train_sub_list=[]
        line_split=line.strip().split('    ')
        if len(line_split)==2:
            line_split[1]=line_split[1].lstrip('"')
            line_split[1]=line_split[1].rstrip('"')
            train_sub_list.append(str(i))
            train_sub_list.append(line_split[0])
            train_sub_list.append(line_split[1])
            train_list.append(train_sub_list)
    print(train_list)
    write_csv(train_list,'train.csv')
def get_test_data():
    lines=read_txt('./cadr/test.txt')
    train_list=[]
    for i,line in enumerate(lines):
        train_sub_list=[]
        line_split=line.strip().split('    ')
        if len(line_split)==2:
            line_split[1]=line_split[1].lstrip('"')
            line_split[1]=line_split[1].rstrip('"')
            train_sub_list.append(str(i+5673))
            train_sub_list.append(line_split[0])
            train_sub_list.append(line_split[1])
            train_list.append(train_sub_list)
    print(train_list)
    write_csv(train_list,'test.csv')
def get_dev_data():
    lines=read_txt('./cadr/test.txt')
    train_list=[]
    for i,line in enumerate(lines):
        train_sub_list=[]
        line_split=line.strip().split('    ')
        if len(line_split)==2:
            line_split[1]=line_split[1].lstrip('"')
            line_split[1]=line_split[1].rstrip('"')
            train_sub_list.append(str(i+5993))
            train_sub_list.append(line_split[0])
            train_sub_list.append(line_split[1])
            train_list.append(train_sub_list)
    print(train_list)
    write_csv(train_list,'dev.csv')
def compare_list(train,test,dev):
    print('train:',len(train))
    print('test:', len(test))
    print('dev:',len(dev))
    out_list=[]
    for t in train:
        text=t.strip()
        out_list.append(text)
    for te in test:
        text=te.strip()
        out_list.append(text)
    for de in dev:
        text=de.strip()
        out_list.append(text)
    out_list=list(set(out_list))
    return out_list
'''
train_list=read_csv('./train-processed.csv')
test_list=read_csv('./test-processed.csv')
dev_list=read_csv('./dev-processed.csv')
all_list=compare_list(train_list,test_list,dev_list)
train_list=all_list[0:4855]
test_list=all_list[4856:5156]
dev_list=all_list[5157:-1]
write_txt(train_list,'./cadr/process_train.txt')
write_txt(test_list,'./cadr/process_test.txt')
write_txt(dev_list,'./cadr/process_dev.txt')
'''
def process_senti_data(csv_file_name,type):
    out_list=[]
    with open(csv_file_name, 'r') as csv:
        lines = csv.readlines()
        step=int(len(lines)/6)
        if type=='train':
            lines=lines[0:4*step]
        elif type=='test':
            lines=lines[4*step+1:5*step]
        else:
            lines = lines[5 * step + 1:-1]
        for line in lines:
            line_split=line.strip().split(',')
            tweet=line_split[2]
            tweet = tweet.replace('USER_MENTION', '').replace(
                'URL', '')
            if len(tweet)>0:
                out_list.append(line_split[1]+'\t'+tweet)
    return out_list
'''
train_list=process_senti_data('./senti/train-processed.csv','train')
test_list=process_senti_data('./senti/train-processed.csv','test')
dev_list=process_senti_data('./senti/train-processed.csv','dev')
write_txt(train_list,'./senti/process_train.txt')
write_txt(test_list,'./senti/process_test.txt')
write_txt(dev_list,'./senti/process_dev.txt')
'''
def process_twitter(file_path):
    lines=read_txt(file_path)
    out_list=[]
    for line in lines:
        line_split=line.strip().split('  ')
        if len(line_split)==4:
            if line_split[3]!='000':
                out_list.append(line_split[2]+'\t'+line_split[3])
        else:
            if len(out_list)>0:
                out_list[-1]=out_list[-1].strip()+line_split[0].strip()
    return out_list
'''
twitter_list=process_twitter('./twitter_dataset/training_set_2_ids_twitter_txt.txt')
print(len(twitter_list))
write_txt(twitter_list,'./twitter_dataset/semm_train.txt')
'''
def get_semm_true(file_path):
    lines=read_txt(file_path)
    out_list=[]
    for line in lines:
        line_split=line.strip().split('\t')
        if len(line_split)==2 and line_split[0]=='1':
            out_list.append(line.strip())
    out_list=list(set(out_list))
    return out_list
def get_cadr_train_data(file_path):
    out_list=[]
    lines = read_txt(file_path)
    for line in lines:
        out_list.append(line.strip())
    return out_list
'''
semm_list=get_semm_true('./semm/process_semm.txt')
print(len(semm_list))
cadr_list=get_cadr_train_data('./cadr/process_train.txt')
print(len(cadr_list))
cadr_list=semm_list+cadr_list
random.shuffle(cadr_list)
print(len(cadr_list))
write_txt(cadr_list,'./cadr/process_train_new.txt')
'''

def process_senti(file_path):
    true_list=[]
    false_list=[]
    lines = read_txt(file_path)
    for line in lines:
        line_split=line.strip().split('\t')
        if len(line_split)==2:
            if line_split[0]=='1':
                true_list.append(line.strip())
            elif line_split[0]=='0':
                false_list.append(line.strip())
    random.shuffle(true_list)
    random.shuffle(false_list)
    print(len(true_list))
    print(len(false_list))
    return true_list,false_list
'''
true_list,false_list=process_senti('./senti/all_train.txt')
cadr_test_list=get_cadr_train_data('./cadr/process_test.txt')
random.shuffle(cadr_test_list)
step=int(5104/4)
false_list=false_list[0:3*step]
true_list=true_list[0:5104-3*step]
senti_list=true_list+false_list+cadr_test_list
random.shuffle(senti_list)
write_txt(senti_list,'./senti/process_train.txt')
'''
def process_cooccurrence(file_path):
    lines = read_txt(file_path)
    token_list=[]
    for line in lines:
        line_split=line.strip().split('\t')
        if len(line_split):
            token_list.append(line_split[3])
            token_list.append(line_split[6])
    token_list=list(set(token_list))
    print(len(token_list))
    return token_list
'''
def match_co(co_list,judge):
    match_candidate = difflib.get_close_matches(judge, co_list, 3, cutoff=0.7)
    print(len(match_candidate))
    if len(match_candidate)>0:
        return True
    else:
        return False


def process_co_data(file_path,co_list):
    lines = read_txt(file_path)
    out_list=[]
    for line in lines:
        add_word_list=[]
        line_split=line.strip().split('\t')
        if len(line_split)==2:
            text=line_split[1]
            words=text.split(' ')
            for word in words:
                if match_co(co_list,word):
                    add_word_list.append(word)
            co_text=" ".join(words+add_word_list)
            out_list.append(line_split[0]+'\t'+co_text)
    return out_list


co_list=process_cooccurrence('./meddra_all_indications.tsv')
psb_train_list=process_co_data('./psb/process_train.txt',co_list)
write_txt(psb_train_list,'./co_psb/process_train.txt')
'''
def get_all_data(file_path):
    lines=read_txt(file_path)
    text_list=[]
    for line in lines:
        line=line.strip()
        line_split=line.split('\t')
        if len(line_split)==2:
            text=line_split[1]
            text = text.rstrip('0123456789')
            text_list.append(text)
    text_list=list(set(text_list))
    return text_list
def get_data_vocab(text_list):
    vocab_list=[]
    write_list=[]
    lines=list(set(text_list))
    max_sent_len = 0
    word_to_idx = {}
    for line in lines:
        text=line.strip()
        text=text.rstrip('0123456789')
        text=text.lstrip(' ')
        word_list=text.split(' ')
        max_sent_len = max(max_sent_len, len(word_list))
        vocab_list+=word_list
    vocab_list=list(set(vocab_list))
    for idx,word in enumerate(vocab_list):
        write_list.append(word)
        if not word in word_to_idx.keys():
            word_to_idx[word] = idx+1
    write_list.append('[PAD]')
    write_list.append('[UNK]')
    write_list.append('[CLS]')
    write_list.append('[SEP]')
    write_list.append('[MASK]')
    write_txt(write_list,'./senti_pretrain/senti_pretrain_bert_data/vocab.txt')

'''
text_list=get_all_data('./senti_pretrain/process_train.txt')
random.shuffle(text_list)
write_txt(text_list,'./senti_pretrain/senti_pretrain_bert_data/text_senti_pretraining.txt')
get_data_vocab(text_list)
'''
def match_co(co_list,judge,segment):
    match_candidate = difflib.get_close_matches(judge, co_list, 3, cutoff=segment)
    #print(len(match_candidate))
    return match_candidate
def process_co_data(co_dict,file_path):
    lines=read_txt(file_path)
    write_list=[]
    for line in lines:
        co_string=''
        line=line.strip()
        line_split=line.split('\t')
        if len(line_split)==2:
            source_text=line_split[1]
            source_word_list=source_text.split(' ')
            for source_word in source_word_list:
                match_candidate_drug=match_co(list(co_dict.keys()),source_word,1)
                if len(match_candidate_drug)>0:#match drug succese
                    co_list=[]
                    for word in source_word_list:
                        for candidate_drug in match_candidate_drug:
                            if word!=candidate_drug:
                                match_candidate_adr = match_co(list(co_dict[candidate_drug]), word,0.6)
                                if len(match_candidate_adr)>0:
                                    sub_adr_list = [candidate_drug]
                                    for candidate_adr in match_candidate_adr:
                                        sub_adr_list.append(candidate_adr)
                                    co_list.append(' '.join(sub_adr_list))
                    co_string=' '.join(co_list)

        write_list.append(co_string)
    print(len(write_list))
    print(write_list)
    return write_list

drug_adr_dict=add.read_drug_adr_dict('./drug_adr/drug_adr_cooccurrence_dicts.txt','@')
semm_train_list=process_co_data(drug_adr_dict,'./semm/process_train_175_175.txt')
semm_dev_list=process_co_data(drug_adr_dict,'./semm/process_dev.txt')
semm_test_list=process_co_data(drug_adr_dict,'./semm/process_test.txt')
write_txt(semm_train_list,'./co_semm_175/process_train.txt')
write_txt(semm_dev_list,'./co_semm_175/process_dev.txt')
write_txt(semm_test_list,'./co_semm_175/process_test.txt')


def process_semm_data(test_path,dev_path,FN_index,FP_index):
    test_lines=read_txt(test_path)
    dev_lines=read_txt(dev_path)
    test_true_list=[]
    test_false_list=[]
    dev_true_list=[]
    dev_false_list=[]
    FN_test_list=[]
    FP_test_list = []
    for index in FN_index:
        index=int(index)
        FN_test_list.append(test_lines[index].strip())
    for index in FP_index:
        index=int(index)
        FP_test_list.append(test_lines[index].strip())
    for test_line in test_lines:
        test_line=test_line.strip()
        if len(test_line.split('\t'))==2:
            [test_label,test_text]=test_line.split('\t')
            if test_label=='0':
                test_false_list.append(test_label+'\t'+test_text)
            elif test_label=='1':
                test_true_list.append(test_label+'\t'+test_text)
    for dev_line in dev_lines:
        dev_line=dev_line.strip()
        if len(dev_line.split('\t'))==2:
            [dev_label,dev_text]=dev_line.split('\t')
            if dev_label=='0':
                dev_false_list.append(dev_label+'\t'+dev_text)
            elif dev_label=='1':
                dev_true_list.append(dev_label+'\t'+dev_text)
    dev_list=dev_true_list+dev_false_list+FN_test_list[0:int(len(FN_test_list)*1)]+test_true_list[0:int(len(test_true_list)*0.35)]+FP_test_list[0:int(len(FP_test_list))]+test_false_list[0:int(len(test_false_list)*0.35)]
    print(len(test_false_list))
    print(len(test_true_list))
    print(len(FN_test_list))
    print(len(FP_test_list))
    random.shuffle(dev_list)
    dev_list=list(set(dev_list))
    print(len(dev_list))
    return dev_list


def get_index(file_path):
    lines=read_txt(file_path)
    index_list=[]
    for line in lines:
        index=line.strip()
        index_list.append(index)
    return index_list
'''
FN_index=get_index('./psb/FN_index.txt')
FP_index=get_index('./psb/FP_index.txt')
train_list=process_semm_data('./psb/process_test.txt','./psb/process_train.txt',FN_index,FP_index)
write_txt(train_list,'./psb/process_train_175_175.txt')
'''
def get_5_cross_data(file_path,out_file):
    lines=read_txt(file_path)
    line_list=[]
    for line in lines:
        line=line.strip()
        line_list.append(line)
    step=int(len(line_list)/5)
    line1=line_list[0:step-1]
    line2=line_list[step:step*2-1]
    line3=line_list[step*2:step*3-1]
    line4=line_list[step*3:step*4-1]
    line5=line_list[step*4:-1]
    traindev1=list(set(line2+line3+line4+line5))
    train1=traindev1[0:3*step-1]
    dev1=traindev1[3*step:-1]
    traindev2=list(set(line1+line3+line4+line5))
    train2=traindev2[0:3*step-1]
    dev2=traindev2[3*step:-1]
    traindev3=list(set(line2+line1+line4+line5))
    train3=traindev3[0:3*step-1]
    dev3=traindev3[3*step:-1]
    traindev4=list(set(line2+line3+line1+line5))
    train4=traindev4[0:3*step-1]
    dev4=traindev4[3*step:-1]
    traindev5=list(set(line2+line3+line4+line1))
    train5=traindev5[0:3*step-1]
    dev5=traindev5[3*step:-1]
    test1=list(set(line1))
    test2 = list(set(line2))
    test3 = list(set(line3))
    test4 = list(set(line4))
    test5 = list(set(line5))
    write_txt(train1,out_file+'/1/process_train.txt')
    write_txt(train2, out_file + '/2/process_train.txt')
    write_txt(train3, out_file + '/3/process_train.txt')
    write_txt(train4, out_file + '/4/process_train.txt')
    write_txt(train5, out_file + '/5/process_train.txt')
    write_txt(test1,out_file+'/1/process_test.txt')
    write_txt(test2, out_file + '/2/process_test.txt')
    write_txt(test3, out_file + '/3/process_test.txt')
    write_txt(test4, out_file + '/4/process_test.txt')
    write_txt(test5, out_file + '/5/process_test.txt')
    write_txt(dev1,out_file+'/1/process_dev.txt')
    write_txt(dev2, out_file + '/2/process_dev.txt')
    write_txt(dev3, out_file + '/3/process_dev.txt')
    write_txt(dev4, out_file + '/4/process_dev.txt')
    write_txt(dev5, out_file + '/5/process_dev.txt')
'''
get_5_cross_data('./semm/process_semm.txt','semm/5_cross')
get_5_cross_data('./psb/process_psb.txt','psb/5_cross')
'''
def process_5cross_train_data(test_path,dev_path,param):
    test_lines=read_txt(test_path)
    dev_lines=read_txt(dev_path)
    test_true_list=[]
    test_false_list=[]
    dev_true_list=[]
    dev_false_list=[]
    for test_line in test_lines:
        test_line=test_line.strip()
        if len(test_line.split('\t'))==2:
            [test_label,test_text]=test_line.split('\t')
            if test_label=='0':
                test_false_list.append(test_label+'\t'+test_text)
            elif test_label=='1':
                test_true_list.append(test_label+'\t'+test_text)
    for dev_line in dev_lines:
        dev_line=dev_line.strip()
        if len(dev_line.split('\t'))==2:
            [dev_label,dev_text]=dev_line.split('\t')
            if dev_label=='0':
                dev_false_list.append(dev_label+'\t'+dev_text)
            elif dev_label=='1':
                dev_true_list.append(dev_label+'\t'+dev_text)
    dev_list=dev_true_list+dev_false_list+test_true_list[0:int(len(test_true_list)*param)]+test_false_list[0:int(len(test_false_list)*param)]
    print(len(test_false_list))
    print(len(test_true_list))
    random.shuffle(dev_list)
    dev_list=list(set(dev_list))
    print(len(dev_list))
    return dev_list
def process_5_cross(data_type,param):
    drug_adr_dict = add.read_drug_adr_dict('./drug_adr/drug_adr_cooccurrence_dicts.txt', '@')
    for i in range(5):
        index=str(i+1)
        test_path='./'+data_type+'/5_cross/'+index+'/process_test.txt'
        dev_path = './' + data_type + '/5_cross/' + index + '/process_dev.txt'
        train_path = './' + data_type + '/5_cross/'+index+'/process_train.txt'
        out_train_path='./' + data_type + '/5_cross/'+index+'/process_train_175_175.txt'

        out_co_train= './co_' + data_type + '/5_cross/'+index+'/process_train.txt'
        out_co_dev = './co_' + data_type + '/5_cross/' + index + '/process_dev.txt'
        out_co_test = './co_' + data_type + '/5_cross/' + index + '/process_test.txt'
        train_list = process_5cross_train_data(test_path,train_path, param)
        write_txt(train_list,out_train_path)
        co_train_list=process_co_data(drug_adr_dict,out_train_path)
        co_test_list = process_co_data(drug_adr_dict, test_path)
        co_dev_list = process_co_data(drug_adr_dict, dev_path)
        write_txt(co_train_list,out_co_train)
        write_txt(co_dev_list,out_co_dev)
        write_txt(co_test_list,out_co_test)

#process_5_cross('psb',0.35)