# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 06:48:20 2018

@author: 85242
"""

#导入数据分析包
import numpy as np
import pandas as pd
import sklearn as sk
from os.path import dirname
import os
import sys


# 基础路径:即代码所在路径
BASEPATH=str(os.getcwd()).replace('\\','/')


f=open(BASEPATH+'/entry09.csv')
df=pd.read_csv(f,low_memory=False)

#查看标签数据有哪些类别，且做数值转换
labe=df['266']
category = pd.Categorical(labe)
df['266']=category.labels

df1=df.select_dtypes(include=['object'])


#此部分为数据预处理部分
df1=df.select_dtypes(include=['object'])
print(df1.head(5))
colums=df1.columns
print(colums)
for colum in colums:
    
    if int(colum)<=86:
        labe=df1[colum]
        category = pd.Categorical(labe)
        df1[colum]=category.labels
    else:
        df1[colum]=df1[colum].replace('?',0)
        df1[colum]=df1[colum].astype(np.float64)

#删除object数据类型        
for colum in colums:
    df.drop(colum,axis=1, inplace=True)
        
df=pd.concat([df1,df],axis=1)
print(df.head(5))
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score  
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#保存结果
file_out=open(BASEPATH+'/result2.csv',mode='w')
def save_result(rfr,x,y,type,cnt):
    
    if type==1:
        a=accuracy_score(y, rfr.predict(x) )
        b=recall_score(y, rfr.predict(x),average='micro')
        c=f1_score(y, rfr.predict(x),average='micro')
        d=classification_report(y, rfr.predict(x))
        aline=format('%s,%s,%s,%s,%s'%('信息熵',cnt,a,b,c))
        print(aline)
        file_out.write(aline+'\n')
        file_out.flush()
        
    else:    
        a=accuracy_score(y, rfr.predict(x) )
        b=recall_score(y, rfr.predict(x),average='micro')
        c=f1_score(y, rfr.predict(x),average='micro')
        d=classification_report(y, rfr.predict(x))
        aline=format('%s,%s,%s,%s,%s'%('fiter',cnt,a,b,c))
        print(aline)
        file_out.write(aline+'\n')
        file_out.flush() 
    
        
#选择RF（随机森林）建立模型，并且对特征进行打分。随机森林打分的最终落脚点为CART树的基尼指数(基尼指数越高代表信息越确定)，跟熵的性质差不多，熵是衡量信息的不确定性
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#打乱数据集
df=shuffle(df)
#获取x,y
x= df.iloc[:,:248] 
y=df.iloc[:,248:249]

rfr = RandomForestClassifier(random_state=0, n_estimators=2000, n_jobs=-1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0) 

names=list(x.columns)

clf = ExtraTreesClassifier(criterion='entropy')
X_new = clf.fit(x, y)
names=sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names),reverse=True)

list_names=[]
for row in range(len(names)):
        list_names.append(names[row][1])
        if ((row%20==0) or (row==247)) and (row>166):
           #信息熵训练模型
              row=row+1
              x_train1=x_train[list_names]
              x_test1=x_test[list_names]
              print(x_test1.shape)   
              rfr.fit(x_train1,y_train)   
              save_result(rfr,x_test1,y_test,1,row)
            #卡方验证输出top前N特征
             
              X_new = SelectKBest(chi2, k=row).fit_transform(x_train, y_train)  
              X_new1 = SelectKBest(chi2, k=row).fit_transform(x_test, y_test)
              print(X_new1.shape)
              rfr.fit(X_new,y_train)
              save_result(rfr,X_new1,y_test,2,row)

               