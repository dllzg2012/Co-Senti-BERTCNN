import logging as logger
import re




'''读取txt文件'''
def txtRead(filePath, encodeType = 'utf-8'):
    listLine = []
    try:
        file = open(filePath, 'r', encoding= encodeType)

        while True:
            line = file.readline()
            if not line:
                break

            listLine.append(line)

        file.close()

    except Exception as e:
        logger.info(str(e))

    finally:
        return listLine

'''读取txt文件'''
def txtWrite(listLine, filePath, type = 'w',encodeType='utf-8'):

    try:
        file = open(filePath, type, encoding=encodeType)
        file.writelines(listLine)
        file.close()

    except Exception as e:
        logger.info(str(e))



def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248

        rstring += chr(inside_code)
    return rstring

'''去除标点符号'''

def get_syboml(ques):
    # ques = '•“鑫菁英”教育分期手续费怎么收取？可以'
    ques = strQ2B(ques)
    # answer = re.sub(u'([。.,，、\；;：:？?！!“”"‘’'''（）()…——-《》<>{}_~【】\\[])', ' ', ques).replace('  ', ' ').replace('   ', ' ')
    answer = re.sub("[\.\!\/_,?;:$%^*<>()+\"\']+|[！，；：。？、“”’‘《》[\]（|）{}【】~@#￥%…&*\/\-—_]+", " ", ques).strip()
    return answer