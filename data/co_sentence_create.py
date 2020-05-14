import codecs
import difflib
def read_txt(file_path):
    fileData=codecs.open(file_path,"r",encoding='ascii',errors='ignore')
    readfile=fileData.readlines()
    return readfile
def write_txt(input_list,file_name):
    fdata = open(file_name, "a")
    fdata.write("\n".join(input_list)+'\n')

#senti word part
def get_senti_dict(senti_file,threshold_value):#./SentiWordNet_3.0.0_20130122.txt
    senti_lines=read_txt(senti_file)
    senti_dict={}
    for senti_line in senti_lines:
        senti_split=senti_line.split('\t')
        if len(senti_split)==6 and senti_split[0]=='a':
            senti_word_list=[senti_word.split('#')[0] for senti_word in senti_split[4].split(' ')]
            for word in senti_word_list:
                if float(senti_split[3])>float(senti_split[2]) and float(senti_split[3])>=threshold_value:
                    senti_dict[word]=float(senti_split[3])
    return senti_dict

#drug and adr part
def read_drug_adr_dict(file_path,sep):
    lines = read_txt(file_path)
    drug_list=[]
    all_adr_list = []
    drug_adr_dict={}
    for line in lines[1:-1]:
        line=line.strip()
        [drug_name,adr_string]=line.split('#')
        adr_list=adr_string.split(sep)
        adr_list=list(set(adr_list))
        drug_adr_dict[drug_name]=adr_list
        drug_list.append(drug_name)
        all_adr_list+=adr_list
    drug_list=list(set(drug_list))
    all_adr_list = list(set(all_adr_list))
    return drug_adr_dict,drug_list,all_adr_list


#match part

def match_co(co_list,judge,segment):
    match_candidate = difflib.get_close_matches(judge, co_list, 3, cutoff=segment)
    return match_candidate
def replace_word(source_word_list,candidate_list,flag,type):
    out_word_list=[]
    target_word_list=[]
    for source_word in source_word_list:
        match_candidate = match_co(candidate_list, source_word, flag)
        if len(match_candidate)>0 and source_word!='[NEW]' and source_word!='[DRUG]' and source_word!='[ADR]':
            if type=='new':
                out_word_list.append('[NEW]')
                target_word_list.append(source_word)
            elif type=='drug':
                out_word_list.append('[DRUG]')
                target_word_list.append(source_word)
            elif type == 'adr':
                out_word_list.append('[ADR]')
                target_word_list.append(source_word)
        else:
            out_word_list.append(source_word)
    target_word_list=list(set(target_word_list))
    return out_word_list,target_word_list

def judge_adr_strength(drug_word_list,adr_word_list,drug_adr_dict):
    strong_adr_list = []
    weak_adr_list = []
    for drug in drug_word_list:
        for adr in adr_word_list:
            if adr in drug_adr_dict[drug]:
                strong_adr_list.append(adr)
            else:
                weak_adr_list.append(adr)
    return strong_adr_list,weak_adr_list

def create_sentence(senti_dict,drug_list,all_adr_list,drug_adr_dict,file_path):
    lines = read_txt(file_path)
    senti_list=list(senti_dict.keys())
    out_list=[]
    for line in lines:
        line=line.strip()
        line_split=line.split('\t')
        if len(line_split)==2:
            source_text=line_split[1]
            source_word_list=source_text.split(' ')
            out_newsentence_list,new_word_list=replace_word(source_word_list,senti_list,1,'new')
            out_drugsentence_list,drug_word_list = replace_word(out_newsentence_list, drug_list, 1, 'drug')
            out_adrsentence_list,adr_word_list = replace_word(out_drugsentence_list, all_adr_list, 1, 'adr')
            if len(new_word_list)>0 and len(drug_word_list)>0 and len(adr_word_list)>0:#NEW,DRUG,ADR
                strong_adr_list,weak_adr_list=judge_adr_strength(drug_word_list, adr_word_list, drug_adr_dict)
                if len(strong_adr_list)==0:
                    supporting_sentence='I feel '+','.join(new_word_list)+' because '+','.join(drug_word_list)+' may cause '+','.join(weak_adr_list)
                else:
                    supporting_sentence = 'I feel ' + ','.join(new_word_list) + ' because ' + ','.join(drug_word_list) + ' cause ' + ','.join(strong_adr_list)
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)==0 and len(drug_word_list)>0 and len(adr_word_list)>0:#DRUG,ADR
                strong_adr_list, weak_adr_list = judge_adr_strength(drug_word_list, adr_word_list, drug_adr_dict)
                if len(strong_adr_list) == 0:
                    supporting_sentence='I feel uncomfortable possibly owing to '+','.join(weak_adr_list)+' of '+','.join(drug_word_list)
                else:
                    supporting_sentence = 'I feel ' + ','.join(strong_adr_list) + ' as ' + ','.join(drug_word_list)
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)==0 and len(drug_word_list)>0 and len(adr_word_list)==0:#DRUG
                supporting_sentence='I need '+','.join(drug_word_list)+',non-ADR '
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)==0 and len(drug_word_list)==0 and len(adr_word_list)>0:#ADR
                supporting_sentence='The annoying '+','.join(adr_word_list)+' makes me uncomfortable '
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)>0 and len(drug_word_list)==0 and len(adr_word_list)==0:#NEW
                supporting_sentence='I dont know why I feel '+','.join(new_word_list)
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)>0 and len(drug_word_list)>0 and len(adr_word_list)==0:#NEW,DRUG
                supporting_sentence='I feel '+','.join(new_word_list)+' for '+','.join(drug_word_list)
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)
            elif len(new_word_list)==0 and len(drug_word_list)==0 and len(adr_word_list)==0:#nothing
                supporting_sentence='Non-drug include non-ADR'
                out_string=line_split[0]+'\t'+line_split[1]+'\t'+supporting_sentence
                print(out_string)
                out_list.append(out_string)

    return out_list


if __name__ == "__main__":
    drug_adr_dict,drug_list,all_adr_list=read_drug_adr_dict('./drug_adr/drug_adr_cooccurrence_dicts.txt','@')
    senti_dict=get_senti_dict('../SentiWordNet_3.0.0_20130122.txt',0.5)
    co_sentence_list=create_sentence(senti_dict, drug_list, all_adr_list,drug_adr_dict, './semm/testDataTrue/process_dev.txt')
    write_txt(co_sentence_list,'./semm/testDataTrue/co_semm/process_dev.txt')