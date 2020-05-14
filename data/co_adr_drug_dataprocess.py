import codecs
import argparse

def read_txt(file_path):
    dataFile=codecs.open(file_path,'r',encoding='ascii',errors='ignore')
    readData=dataFile.readlines()
    return readData
def write_txt(input_list,file_name):
    fdata = open(file_name, "a")
    fdata.write("\n".join(input_list))


def get_diff_drug(corpus_adr_path,media_adr_path,corpus_drug_path,medra_drug_path):
    ca_lines=read_txt(corpus_adr_path)
    ma_lines = read_txt(media_adr_path)
    cd_lines = read_txt(corpus_drug_path)
    md_lines = read_txt(medra_drug_path)

    ca_id_adr={}
    ca_adr_id={}
    ma_id_adr={}
    ma_mad_dict={}
    for ca_line in ca_lines:
        ca_line=ca_line.strip()
        ca_split=ca_line.split('\t')
        if len(ca_split)==3:
            if ca_split[0] not in ca_id_adr.keys():
                ca_id_adr[ca_split[0]]=ca_split[1]
            if ca_split[1] not in ca_adr_id.keys():
                ca_adr_id[ca_split[1]]=ca_split[0]
    for ma_line in ma_lines:
        ma_line=ma_line.strip()
        ma_split=ma_line.split('\t')
        ma_id = ma_split[1].lower()
        ma_name=ma_split[3].lower()
        mad_id = ma_split[0].lower()
        ma2_id = ma_split[5].lower()
        ma2_name = ma_split[6].lower()

        if len(ma_split)==7:
            if ma_id==ma2_id:
                if ma_id not in ma_id_adr.keys():
                    ma_id_adr[ma_id.lower()]=ma_name
                if ma_id not in ma_mad_dict.keys():
                    ma_mad_dict[ma_id] = mad_id
            else:
                if ma_id not in ma_id_adr.keys():
                    ma_id_adr[ma_id.lower()]=ma_name
                if ma_id not in ma_mad_dict.keys():
                    ma_mad_dict[ma_id] = mad_id
                if ma2_id not in ma_id_adr.keys():
                    ma_id_adr[ma2_id.lower()] = ma2_name
                if ma2_id not in ma_mad_dict.keys():
                    ma_mad_dict[ma2_id] = mad_id

    ca_ma_colist=list(set(list(ca_id_adr.keys())).intersection(set(list(ma_id_adr.keys()))))
    co_ma_drug_list=[]

    for adr_id in ca_ma_colist:
        co_ma_drug_list.append(ma_mad_dict[adr_id])

    md_id_drug={}
    md_drug_id = {}
    for md_line in md_lines:
        md_line=md_line.strip().lower()
        md_split=md_line.split('\t')
        if md_split[0] not in md_id_drug.keys():
            md_id_drug[md_split[0]]=md_split[1]
        if md_split[1] not in md_drug_id.keys():
            md_drug_id[md_split[1]]=md_split[0]
    drug_id_list=list(set(md_id_drug.keys()).intersection(set(co_ma_drug_list)))
    drug_name_list=[]

    for drug_id in drug_id_list:
        drug_name_list.append(md_id_drug[drug_id])
    cd_drug_name_list=[]
    for cd_line in cd_lines:
        cd_drug_name_list.append(cd_line.strip())
    co_drug_list=list(set(cd_drug_name_list).intersection(set(drug_name_list)))
    diff_drug_list=list(set(cd_drug_name_list).difference(set(co_drug_list)))


    return ca_adr_id,ca_id_adr,md_id_drug,md_drug_id

def get_drug_adr_dict(file_path,md_id_drug):
    lines = read_txt(file_path)
    drug_list=[]
    drug_adr_dict={}
    for line in lines:
        line=line.strip()
        line_split=line.split('\t')
        drug_id = line_split[0].lower()
        drug_list.append(drug_id)
    drug_list=list(set(drug_list))
    for drug in drug_list:
        if drug in md_id_drug.keys():
            sub_adr_list=[]
            for line in lines:
                line = line.strip()
                line_split = line.split('\t')
                adr_id = line_split[1].lower()
                adr_name = line_split[3].lower()
                drug_id = line_split[0].lower()
                adr2_id = line_split[5].lower()
                adr2_name = line_split[6].lower()
                if drug==drug_id:
                    if adr_id==adr2_id:
                        sub_adr_list.append(adr_name)
                    else:
                        sub_adr_list.append(adr_name)
                        sub_adr_list.append(adr2_name)
            sub_adr_list=list(set(sub_adr_list))
            drug_adr_dict[md_id_drug[drug]]=sub_adr_list
    return drug_adr_dict

def read_drug_adr_dict(file_path,sep):
    lines = read_txt(file_path)
    drug_adr_dict={}
    for line in lines[1:-1]:
        line=line.strip()
        [drug_name,adr_string]=line.split('#')
        adr_list=adr_string.split(sep)
        adr_list=list(set(adr_list))
        drug_adr_dict[drug_name]=adr_list
    return drug_adr_dict
def write_drug_adr_dict(drug_adr_dict,write_file_path):
    write_list=[]
    for drug in drug_adr_dict.keys():
        write_line=drug+'#'+','.join(drug_adr_dict[drug])
        write_list.append(write_line)
    write_txt(write_list,write_file_path)

def merge_drug_adr_dict(dict,add_dict):
    drug_adr_dict={}
    drug_list=dict.keys()
    add_drug_list=add_dict.keys()
    inter_list=list(set(drug_list).intersection(set(add_drug_list)))
    add_list=add_drug_list-inter_list
    for drug in drug_list:
        drug_adr_dict[drug]=dict[drug]
    for add_drug in add_list:
        drug_adr_dict[add_drug] = add_dict[add_drug]
    return drug_adr_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_adr_path', type=str, default='./drug_adr/corpus/ADR_lexicon.tsv',help='file path of cadr_dataset')
    parser.add_argument('--media_adr_path', type=str, default='./drug_adr/medra/meddra_all_indications.tsv', help='file path of cadr_dataset')
    parser.add_argument('--corpus_drug_path', type=str, default='./drug_adr/corpus/drug_names.txt', help='file path of senti_dataset')
    parser.add_argument('--media_drug_path', type=str, default="./drug_adr/medra/drug_names.tsv", help='file path of senti_dict')
    parser.add_argument('--add_drug_adr_file', type=str, default="./drug_adr/drug-adr-co-occurrence_dicts.txt",help='file path of senti_dict')
    parser.add_argument('--drug_adr_file', type=str, default="./drug_adr/drug-adr-dict.txt",help='file path of senti_dict')

    args = parser.parse_args()
    params = vars(args)

    ca_adr_id,ca_id_adr,md_id_drug,md_drug_id=get_diff_drug(args.corpus_adr_path,args.media_adr_path,args.corpus_drug_path,args.media_drug_path)

    drug_adr_dict=get_drug_adr_dict(args.media_adr_path,md_id_drug)
    add_drug_adr_dict=read_drug_adr_dict(args.add_drug_adr_file,'@')
    all_drug_adr_dict=merge_drug_adr_dict(drug_adr_dict,add_drug_adr_dict)
    print(len(all_drug_adr_dict))
    print(all_drug_adr_dict)
    write_drug_adr_dict(all_drug_adr_dict,args.drug_adr_file)