# coding: utf-8
"""
Create on 2018/3/27
@author:chenglei

"""
import os
import re
import os, re, time, random, socket, urllib.request,datetime
from lxml import etree
from os.path import dirname
from itertools import product

BASEPATH = dirname(os.path.abspath(__file__)).replace('\\', '/')

def visit_url(url):
    user_agents = [
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.31 (KHTML, like Gecko) Chrome/26.0.1410.43 BIDUBrowser/6.x Safari/537.31',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.44 Safari/537.36 OPR/24.0.1558.25 (Edition Next)',
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1985.125 Safari/537.36 OPR/23.0.1522.60 (Edition Campaign 54)'
        'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36',
        'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19',
        'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0',
        'Mozilla/5.0 (Android; Mobile; rv:14.0) Gecko/14.0 Firefox/14.0',
        'Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Safari/535.19',
        'Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30'
    ]
    user_agent = random.choice(user_agents)
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
               'Connection': 'keep-alive',
               'User-Agent': user_agent
               }
    req = urllib.request.Request(url, headers=headers, method='GET')
    response = urllib.request.urlopen(req)
    html = response.read()
    response.close()    
    return html.decode('utf-8')#, 'ignore')


def url_math(url):
    #网页上获取到的url不完成，需要对其进行拼
    if len(url)>0:
        full_url='https://www.chunyuyisheng.com'+url[0]
    else:
        full_url='https://www.chunyuyisheng.com'
    
    return full_url
   
 

def url_math1(url):
    #网页上获取到的url不完成，需要对其进行拼
    full_url='https://www.chunyuyisheng.com'+url
    return full_url
       

    
def get_next_url(html):
    # 获取下一页面url
    e_HTML=etree.HTML(html)
    page_next_url=e_HTML.xpath('//a[@class="next"]/@href')
    return page_next_url
        



def get_all_url(html):
     #获取当前页面所有Url
     e_HTML=etree.HTML(html)
     e_HTML1=e_HTML.xpath('//div[@class="doctor-list"]')
     for html in e_HTML1:
         list_url=html.xpath('div/div[1]/a/@href')
     return list_url

def get_path(content):
     content=content.strip()
     if re.findall('湿疹',content):
         save_path=BASEPATH+'/湿疹'
         mk_dir(save_path)
         
     elif re.findall('痤疮',content):
          save_path=BASEPATH+'/痤疮'
          mk_dir(save_path)
         
     elif re.findall('胎记',content):
          save_path=BASEPATH+'/胎记'
          mk_dir(save_path)
          
     elif re.findall('蒙古斑',content):
          save_path=BASEPATH+'/蒙古斑'
          mk_dir(save_path)
     else:
          save_path=BASEPATH+'/其他'
          mk_dir(save_path)
     return save_path


def mk_dir(SAVEPATH):
   if not os.path.exists(SAVEPATH):
        os.mkdir(SAVEPATH)


def get_huati_more_url(url):
      #返回医生话题更多url
      html=visit_url(url)
      time.sleep(2)
      html=etree.HTML(html)
      url=html.xpath('//a[@class="more"]/@href')
      full_url=url_math(url)
      return  full_url 
      
  
def get_huti_urllist(html):
      #xpath获取更多链接      
      html=etree.HTML(html)
      for html1 in html.xpath('/html/body/div[4]/div[2]'): 
          list_url=html1.xpath('div/div[1]/a/@href')
          list_content=html1.xpath('div/div[1]/a/text()')     
      return dict(zip(list_url,list_content))


def get_huati_main(url):
        #循环获取话题url链接，且进行访问
        #正则表达式获取图片url数据
        page_next=url
        flag=True
        while flag==True:
           html1=visit_url(page_next)
           print(page_next)
           huti_all=get_huti_urllist(html1)
           for huati_url in huti_all:  
               path=get_path(huti_all[huati_url])
               huati_url=url_math1(huati_url)
               print('话题url:'+huati_url)
               html=visit_url(huati_url)
               time.sleep(2)
               try:
                   get_image(html,path) 
               except:
                   pass
           page_next= get_next_url(html1)
           if len(page_next)==0:
               flag=False
           page_next=url_math(page_next) 
           time.sleep(5) 
            
        
        
        
        
    
def get_image(html,path):
    #获取图片url并且进行保存
    p=r'<img src="(https.*?)"'
    html_list=re.findall(p,html)
    if len(html_list)>0:
        print('已获取到该网页所有图片')
        for im_url in html_list:
            save_image(im_url,path)
    else:
        print('该网页没有图片数据')
  
      
        
def save_image(url,path):        
   #保存图片到指定文件夹下面
   #https://r.sinaimg.cn/large/article/b4166b1244113c5afb9cca3853693f58.png
    imgurl=url
    if len(re.findall('\.png',imgurl))==0:
        if len(re.findall('\.jpg',imgurl))==0:
            p = r'[A-Za-z0-9]+'
            a = re.findall(p, imgurl)
            urllib.request.urlretrieve(imgurl, path + '/' + a[-1]+'.jpg')
        else:
            p = r'[A-Za-z0-9]+\.jpg'
            a = re.findall(p, imgurl)
            print(a,imgurl)
            urllib.request.urlretrieve(imgurl, path + '/' + a[0])

    
def main(url):
    #解析当前页面Url
        page_next=url
        flag=True
        while flag==True:
           html=visit_url(page_next)
           doctor_list=get_all_url(html) 
           for dortor_url in doctor_list:
               #获取话题链接 
               try:
                   
                  dortor_url=url_math1(dortor_url)
                  print('医生url:'+dortor_url)
                  talk_url=get_huati_more_url(dortor_url) 
                  get_huati_main(talk_url)
                  
               except:
                   print('抛出异常情况')
           page_next= get_next_url(html)
           if len(page_next)==0:
               break;
           page_next=url_math(page_next) 
 
if __name__ == '__main__':
    
	#获取所有医生下的图片	
     #main('https://www.chunyuyisheng.com/pc/search/doctors/?query=%E7%9A%AE%E8%82%A4%E7%97%85')
	#获取一个医生下的图片，答辩做演示用。
	 get_huati_main('https://www.chunyuyisheng.com/pc/topic/list/?doctor_id=clinic_web_176d72ee15105284')
	 
     