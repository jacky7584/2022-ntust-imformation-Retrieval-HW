import pandas as pd
import math
from sklearn.preprocessing import normalize
from collections import defaultdict
import operator
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import gc

def bm25(s1_cut,s2_cut,DF,co,k1,k3,b,dk,avg_doclen):
    wSum=0
    for i in s1_cut:
        try:
            tda=k3
            tfverse=s2_cut[i]/(1+(b * (dk/avg_doclen)))
            upper1 = (k1 + 1) * s2_cut[i]
            down1 = (s2_cut[i] + k1 * ((1 - b)+ b * dk / avg_doclen))
            idf = math.log((5000+1) / (DF[i] + 0.5)) 
            wSum += (((upper1 / down1))  * idf)
        except:
            wSum +=0
            
    if co%1000==0:
        gc.collect()
    return wSum


def tf(dict1):
    td={}
    for word in dict1:
        td[word]=dict1.count(word)/len(dict1)
    return td


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
    documents_test['cut'][i]=tf(documents_test['cut'][i])

DF = {} 
avg_doclen = 0
for i in range(len(documents_test)): 
    tokens = documents_test['cut'][i] 
    avg_doclen+=len(tokens)
    for w in tokens: 
        try: 
            DF[w].add(i) 
        except: 
            DF[w] = {i}         
for i in DF:
    DF[i]=len(DF[i])
    
avg_doclen/=5000

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
    f_test['cut'][i]=tf(f_test['cut'][i])

final=[]
score=[]
co=0
k1=0.0001
k3=1
b=0
for i in range(len(f_test)):
    key_value ={}     
    for j in range(len(documents_test)):
        ans=bm25(f_test['cut'][i],documents_test['cut'][j],DF,co,k1,k3,b,len(documents_test['cut'][j]),avg_doclen)
        key_value[documents_test['name'][j]]=ans
        co+=1
        print(co)
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