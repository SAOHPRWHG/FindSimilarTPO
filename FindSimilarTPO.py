# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:36:44 2020

@author: August Yu
"""
from gensim import corpora,models,similarities
import jieba
from collections import defaultdict
import heapq
import argparse


#寻找最相似的X篇TPO文章
'''
find file name: 
    Reading materials : 1-1.txt 
    Listening materials : C1-1.txt(Conversation) or L1-1.txt(Lecture)
Each TPO have 3 reading materials, 2 conversations and 4 lectures.
'''
parser = argparse.ArgumentParser(description='Find Similar TPO')
parser.add_argument(
    "-f",
    '--filename',
    metavar="FILE",
    help="filename: \nReading materials : 1-1.txt \n Listening materials : C1-1.txt(Conversation) or L1-1.txt(Lecture)",
    type=str,
)
parser.add_argument(
    "-n",
    '--num',
    default = 10,
    help="Show how many records",
    type=int,
)
parser.add_argument(
    "-t",
    '--tpo',
    default = 54,
    help="Find in how many TPO, default is 54, you may change to 64 if you got them",
    type=int,
)

#argv = parser.parse_args('-f C2-1.txt -n 10 -t 54'.split())
#print(argv)

find_file = argv.filename
sim_num = argv.num #show the similar text number要显示的最相似的数目
TPO_num = argv.tpo #目前可访问的TPO数目，如增加到64则改为64



lib = []
for i in range(1,TPO_num+1):
    for j in range(1,4): 
        file_name = str(i)+'-'+str(j)+'.txt'
        doc = './data/TPO/'
        doc += file_name
        #print(doc)
        d = open(doc, encoding='utf-8').read()
        data = jieba.cut(d)
        data_ = ''
        for item in data:
            data_+=item+' '
        lib.append(data_)
    for j in range(1,5):
        file_name = 'L' + str(i)+'-'+str(j)+'.txt'
        doc = './data/TPO/'
        doc += file_name
        d = open(doc, encoding='utf-8').read()
        data = jieba.cut(d)
        data_ = ''
        for item in data:
            data_+=item+' '
        lib.append(data_)
    for j in range(1,3):
        file_name = 'C' + str(i)+'-'+str(j)+'.txt'
        doc = './data/TPO/'
        doc += file_name
        d = open(doc, encoding='utf-8').read()
        data = jieba.cut(d)
        data_ = ''
        for item in data:
            data_+=item+' '
        lib.append(data_)



#存储文档到列表
#documents=[data11,data21]
texts=[[word for word in document.split()]
       for document in lib]
 
#计算词语的频率
frequency=defaultdict(int)#构建频率对象
for text in texts:
    for token in text:
        frequency[token]+=1
'''
#如果词汇量过多，去掉低频词
texts=[[word for word in text if frequency[token]>3]
 for text in texts]
'''
#通过语料库建立词典
dictionary=corpora.Dictionary(texts)
dictionary.save('./dictionary.txt')
#加载要对比文档
doc3='./data/TPO/'

#find_file = '1-1.txt'
doc3+=find_file

d3=open(doc3,encoding='utf-8').read()
data3=jieba.cut(d3)
data31=''
for item in data3:
    data31+=item+' '
new_doc=data31
new_vec=dictionary.doc2bow(new_doc.split())#转换为稀疏矩阵
#得到新的语料库
corpus=[dictionary.doc2bow(text) for text in texts]
 
ifidf=models.TfidfModel(corpus)
featureNUm=len(dictionary.token2id.keys())#得到特征数
index=similarities.SparseMatrixSimilarity(ifidf[corpus],num_features=featureNUm)
sim=index[ifidf[new_vec]]

r = {}
l = {}
for i in range(1,TPO_num+1):
    for j in range(1,4): 
        file_name = str(i)+'-'+str(j)+'.txt'
        ind = (i-1)*9 + j
        r[file_name] = sim[ind-1]*100
    for j in range(1,5): 
        file_name = 'L' + str(i)+'-'+str(j)+'.txt'
        ind = (i-1)*9 + 3 + j
        l[file_name] = sim[ind-1]*100 
    for j in range(1,3): 
        file_name = 'C' + str(i)+'-'+str(j)+'.txt'
        ind = (i-1)*9 + 7 + j
        l[file_name] = sim[ind-1]*100              
        
#sim_num = 10
outputR = heapq.nlargest(sim_num, r, key = r.get)
outputL = heapq.nlargest(sim_num, l, key = l.get)
#print(output)
print('Reading:')
print('Rank\tName\t\tSimilarity')
for i in range(len(outputR)):
    print(i,'\t', outputR[i],':\t',r[outputR[i]])
    
print('\nListening:')
print('Rank\tName\t\tSimilarity')
for i in range(len(outputL)):
    print(i,'\t', outputL[i],':\t',l[outputL[i]])

