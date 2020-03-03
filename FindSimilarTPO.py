# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:36:44 2020

@author: August Yu
"""
#寻找最相似的X篇TPO文章
#输入参数
find_file = '1-3.txt' #要找的文档内容
sim_num = 10 #要显示的最相似的数目

from gensim import corpora,models,similarities
import jieba
from collections import defaultdict
import heapq

TPO_num = 54 #目前可访问的TPO数目，如增加到64则改为64
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

d = {}
for i in range(1,TPO_num+1):
    for j in range(1,4): 
        file_name = str(i)+'-'+str(j)+'.txt'
        ind = (i-1)*3 + j
        d[file_name] = sim[ind-1]*100
                
#sim_num = 10
output = heapq.nlargest(sim_num, d, key = d.get)
#print(output)
for i in range(len(output)):
    print(i,'\t', output[i],':\t',d[output[i]])

