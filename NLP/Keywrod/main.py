from textrank4zh import TextRank4Keyword
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from os.path import dirname
import numpy as np
import pandas as pd
import os
import time

BASEPATH = dirname(os.path.abspath(__file__)).replace('\\', '/')
stop_words=BASEPATH+'/english.txt'
stop_list=[]
f=open(stop_words,mode='r')
for row in f.readlines():
   stop_list.append(row.strip())
   
	  
######
# author: leicheng，此处作者请更换为自己的名字
# create_time 20180327
####


def textrank(text):
   # text 传入为字符串形式
   # textrank提取摘要 
  
   word=TextRank4Keyword(stop_words_file=stop_words)
   word.analyze(text,window=5,lower=True)
   wor_list=word.get_keywords(num=5,word_min_len=1)
   return wor_list 

def tf_idf(text): 
    # text 传入为数组形式 ['this is content']
    vectorizer = CountVectorizer(stop_words=stop_list)
    X = vectorizer.fit_transform(text)
    #获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()  
    transformer = TfidfTransformer()
    #将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    key={}
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
         for j in range(len(word)):
            key[word[j]]=weight[i][j]
    
    list=sorted(key.items(),key = lambda x:x[1],reverse = True)
    return list[0:5]


def read_row_data(read_path):
   
    return pd.read_excel(read_path,header=None)


def wirte_data(file_out,aline): 
    file_out.write(aline)
    file_out.flush()

def mian():
   #原始数据地址一定要与代码存放地址保持一致
   read_path=BASEPATH+'/abstract.xlsx'
   df_data=read_row_data(read_path)
   #print(df_data.head(5))
   save_path=BASEPATH+'/abstract_out.xlsx'
   file_out=open(save_path,mode='w')
   list=[]
   for row in df_data.iterrows():
    
       contens=row[1][0]
       contents1=contens.split('AB  -')
       content=[]
       content.append(contents1[1])
       key=textrank(contents1[1])
       key1=tf_idf(content)
       words=[]
       weigths=[]
       for row1 in range(len(key)):
           word=key[row1]['word']
           weigth=key[row1]['weight']
         
           words.append(word) 
           weigths.append(str(weigth))
           
       alone=','.join(words)    
       weigth1=','.join(weigths)
       
       
       
       words=[]
       weigths=[]
       for row1 in range(len(key1)):
           word=key1[row1][0]
           weigth=key1[row1][1]
           
           words.append(word)
           weigths.append(str(weigth))
           
       alone1=','.join(words)
       weigth2=','.join(weigths)
      
       
       #aline=format('"%s"|"%s"|"%s" \n' %(contens,alone,alone1))
       #print(weigth1,weigth2)
       list.append([contens,alone,weigth1,alone1,weigth2])
       #wirte_data(file_out,aline)
   df=pd.DataFrame(list,columns=['stract','textrank_keyword','textrank_weight','tf_idf_keyword','tf_idf_weight'])
   df.to_excel(save_path)
       
       

if __name__ == '__main__':
    mian()
    
    read_path=BASEPATH+'/abstract.xlsx'
    df_data=read_row_data(read_path)
    save_path=BASEPATH+'/abstract_out1.xlsx'
    crops=[]
    for row in df_data.iterrows():
        contens=row[1][0]
        crops.append(contens)  
    vectorizer=CountVectorizer(stop_words=stop_list)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值  
    tfidf=transformer.fit_transform(vectorizer.fit_transform(crops))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重  
    df_list=[]
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
        print ("-------这里输出第",i,u"类文本的词语tf-idf权重------")  
        key={}
        for j in range(len(word)):  
            key[word[j]]=weight[i][j]
        list=sorted(key.items(),key = lambda x:x[1],reverse = True)
        print(list[0:5][0][1])
        
        words=[]
        weigths=[]
        for row in range(len(list[0:5])):
            print(list[0:5][row][0])
            words.append(list[0:5][row][0])
            weigths.append(str(list[0:5][row][1]))
        alone1=','.join(words)
        weigth2=','.join(weigths)    
        df_list.append([i,alone1,weigth2])
       
    df=pd.DataFrame(df_list,columns=['id','tf_idf_keyword','tf_idf_weight'])
    df.to_excel(save_path)

   
   
	
	