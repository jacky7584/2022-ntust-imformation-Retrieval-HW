import pandas as pd
import math
from sklearn.preprocessing import normalize
from collections import defaultdict
import operator
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import gc

def tftoidf(word_dict,cut_code,word_idf):
    ans={}
   
    for i in word_dict:
        ans[i]=(cut_code[word_dict[i]])*word_idf[i]
    return ans

def compare2(s1_cut,s2_cut,DF,co):
   
   
    word_set = set(s1_cut).union(set(s2_cut))
    word_dict = dict()
    i = 0
   
    for word in word_set:#比較標號
        word_dict[word] = i
        i += 1
    
    #--------TF
    s1_cut_code = [word_dict[word] for word in s1_cut]#TF
    s1_cut_code = [0]*len(word_dict)

    for word in s1_cut:#S1詞頻
        s1_cut_code[word_dict[word]]+=1
    
    s2_cut_code = [word_dict[word] for word in s2_cut]

    s2_cut_code = [0]*len(word_dict)
    for word in s2_cut:#S2詞頻
        s2_cut_code[word_dict[word]]+=1
    for i in range(len(word_dict)):
        s1_cut_code[i]=s1_cut_code[i]/len(word_dict)
        s2_cut_code[i]=s2_cut_code[i]/len(word_dict)
    #--------IDF1

     
    doc_num=1000
    word_idf={}
    word_doc=DF
    s1_idf_code = s1_cut #S1單詞

    for i in word_dict:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
    #print(word_idf)

    #--------IDF2.

    doc_num=1000
    word_idf2={}
    word_doc=DF
    s2_idf_code = s2_cut

    for i in word_dict:
        word_idf2[i]=math.log(doc_num/(word_doc[i]+1))
    #--------TF*IDF.  

    word_tf_idf=tftoidf(word_dict,s1_cut_code,word_idf)
   
    word_tf_idf2=tftoidf(word_dict,s2_cut_code,word_idf2)

    # 计算余弦相似度

    sum = 0
    sq1 = 0
    sq2 = 0
    for i in word_dict:
        sum += word_tf_idf[i] * word_tf_idf2[i]
        #print(i)
        sq1 += pow(word_tf_idf[i], 2)
        sq2 += pow(word_tf_idf2[i], 2)

    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 5)
    except ZeroDivisionError:
        result = 0.0
    if co%10000==0:
        del word_tf_idf
        del word_tf_idf2
        del word_set
        del word_dict
        gc.collect()
    return result

documentPath = 'documents/'
documentList = os.listdir(documentPath)
documentnum=[]
dolist=[]
for file in documentList:
    f = open(documentPath+file, 'r')
    dolist.append(f.read())
    file=file.replace('.txt','')
    documentnum.append(file)
setdolist=[]   
col=[]
col.append('name')
col.append('cut')
documents_test = pd.DataFrame(columns=col,dtype=float)

documents_test['name']=documentnum
for i in range(len(documents_test)):
    documents_test['cut'][i]=dolist[i].split(' ')
DF = {} 
for i in range(len(documents_test)): 
    tokens = documents_test['cut'][i] 
    for w in tokens: 
        try: 
            DF[w].add(i) 
        except: 
            DF[w] = {i}         
for i in DF:
    DF[i]=len(DF[i])


queriesPath = 'queries/'
queriesList = os.listdir(queriesPath)
qlist=[]
namelist=[]
af = open('queries_id_list.txt', 'r')
for file in af:
    file=file.replace('\n','')
    namelist.append(file)
    f = open(queriesPath+file+'.txt', 'r')
    qlist.append(f.read())   
    
col=[]
col.append('name')
col.append('cut')
f_test = pd.DataFrame(columns=col,dtype=float)
f_test['name']=namelist
for i in range(len(f_test)):
    f_test['cut'][i]=qlist[i].split(' ')

final=[]
score=[]
co=0
for i in range(len(f_test)):
    key_value ={}     
    for j in range(len(documents_test)):
        ans=compare2(f_test['cut'][i],documents_test['cut'][j],DF,co)
        key_value[documents_test['name'][j]]=ans
        co+=1
    score.append(sorted(key_value.items(), key = lambda kv:(kv[1], kv[0]),reverse=True))
col=[]
col.append('Query')
col.append('RetrievedDocuments')
answer = pd.DataFrame(columns=col,dtype=float)
answer['Query']=namelist
answer['RetrievedDocuments']=score

answer1=answer.copy()
for i in range(len(answer1)):
    st=''
    for j in range(len(answer1['RetrievedDocuments'][i])):
        if(j==999):
            st=st+answer1['RetrievedDocuments'][i][j][0]
        else:
            st=st+answer1['RetrievedDocuments'][i][j][0]+' '
    answer1['RetrievedDocuments'][i]=st

answer1.to_csv('submission.csv',index=0)