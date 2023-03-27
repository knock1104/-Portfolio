#!/usr/bin/env python
# coding: utf-8

# ### 쥬피터 노트북 창크기 조정

# In[34]:


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# ### 테스트 및 코드 확인시 autosave 0으로 설정 후 이용
# * default 0

# In[143]:


get_ipython().run_line_magic('autosave', '3600')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
get_ipython().run_line_magic('matplotlib', 'inline')
#pd.set_option('max_columns', 20, 'max_rows', 20)


# In[3]:


# 차트에서 한글 출력을 위한 설정
import platform
your_os = platform.system()
if your_os == 'Linux':
    rc('font', family='NanumGothic')
elif your_os == 'Windows':
    ttf = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=ttf).get_name()
    rc('font', family=font_name)
elif your_os == 'Darwin':
    rc('font', family='AppleGothic')
rc('axes', unicode_minus=False)


# In[10]:


features = pd.DataFrame({'cust_id': tr.cust_id.unique()})


# In[12]:


# 데이터 확인을 위해 주석 
# tr = pd.read_csv('transactions.csv', encoding='cp949')
# tr


# ### Numeric Features [파생변수 1~33]

# #### [파생변수 1~5] 총구매액, 구매건수, 평균구매액, 최소구매액,최대구매액

# In[53]:


p=tr.groupby('cust_id')['amount'].agg([
    ('총구매액',np.sum),
    ('구매건수',np.size),
    ('평균구매액',lambda x:np.round(np.mean(x))),
    ('최소구매액',np.min),
    ('최대구매액',np.max),
]).reset_index()
features=features.merge(p,how='left');features


# In[17]:


p.hist(bins=20, figsize=(15,10))
plt.show()


# #### [파생변수 6~7] 환불금액, 환불건수

# In[35]:


p=tr[tr.amount<0].groupby('cust_id')['amount'].agg([
    ('환불금액',lambda x :x.sum()*-1),
    ('환불건수',np.size)
]).reset_index()
features=features.merge(p,how='left');features


# #### [파생변수 8~10] 구매상품종류(goos_id, gds_grp_nm,gds_mclas_nm)
# * 상품 소분류, 중분류, 대분류로 상위분류에 속한 상품을 여러개 구입할 경우 구입 size가 다를 수 있음

# In[41]:


p=tr.groupby('cust_id').agg({
    'goods_id':[('상품소분류',lambda x :x.nunique())],
    'gds_grp_nm':[('상품중분류',lambda x:x.nunique())],
    'gds_grp_mclas_nm':[('상품대분류',lambda x:x.nunique())]    
})
p.columns=p.columns.droplevel()
p=p.reset_index()
features=features.merge(p,how='left');features


# In[42]:


p.hist(bins=20,figsize=(15,10))
plt.show()


# #### [파생변수11~12]최근 n 개월 내 구매 상품, 구매 건수
# * 최근 n개월 내 구매 상품, 구매 건수를 비교하여 남 여를 구분하는데 도움이 될 것이라 생각
# * 여성일수록 최근 구매 개월이 짧고 남성의 경우 길것으로 판단

# In[20]:


# [3,6,9,12] 원하는 개월을 대입하여 추출가능
for m in [3]:
    start = str(pd.to_datetime(tr.tran_date.max()) - pd.offsets.MonthBegin(m))
    f = tr.query('tran_date >= @start').groupby('cust_id')['gds_grp_nm'].agg([
        (f'최근{m}개월_구매상품', np.sum), 
        (f'최근{m}개월_구매건수', np.size)
    ]).reset_index()
    display(p)
    features = features.merge(p, how='left'); features


# In[ ]:


#개선된 코드
# # pd.to_datetime(), pd.offsets.MonthBegin(3)
# for m in [3,6,12]:
#     start = str(pd.to_datetime(tr.tran_date.max()) - pd.offsets.MonthBegin(m))
#     f = tr.query('tran_date >= @start').groupby('cust_id')['amount'].agg([
#         (f'최근{m}개월_구매금액', np.sum), 
#         (f'최근{m}개월_구매건수', np.size)
#     ]).reset_index()
#     display(f)
#     features = features.merge(f, how='left'); features


# #### [파생변수13~14] 세일선호도
# * 백화점 세일기간은 보통 1,4,7,11 (신세계, 현대, 롯데백화점 시즌 검색) 달로 1년에 다섯번 정도 시행한다. 이 시즌에 여성들이 남성들보다 더 민감하게 반응할 것으로 판단

# In[43]:


p = tr.groupby('cust_id')['tran_date'].agg([
    ('세일기간 내 구매비율', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([1,4,7,11]))),
    ('비세일기간 내 구매비율', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([2,3,5,6,8,9,12]))),    
]).reset_index()
display(p)
features = features.merge(p, how='left'); features


# In[44]:


p.hist(bins=20, figsize=(15,10))
plt.show


# #### [파생변수15~20] 시즌선호도 2
# * 남성들은 세일 기간과 시즌오프에 여성들보다 덜 민감하게 반응할 것이다. 따라서 시즌 오프 전 구매 비율이 높을 것으로 판단된다.

# In[45]:


p = tr.groupby('cust_id')['tran_date'].agg([
    ('시즌오프 후 구매비율', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([1,2]))),
    (' 시즌오프 전 구매비율(3-4월)', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([3,4]))),
    (' 시즌오프 전 구매비율(5-6월)', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([5,6]))),
    (' 시즌오프 전 구매비율(7-8월)', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([7,8]))),
    (' 시즌오프 전 구매비율(9-10월)', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([9,10]))),
    (' 시즌오프 전 구매비율(11-12월)', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([11,12])))
]).reset_index()
display(p)
features = features.merge(p, how='left'); features


# In[46]:


p.hist(bins=20,figsize=(15,10))
plt.show()


# #### [파생변수 20 ~ 31] 구매추세 패턴
# * 남성들은 백화점에서 필요물품만을 구입하는 편이고 여성들은 필요물품 외 세일이나 부과적인 물품을 같이 구입하는 성향이 있다고 생각. 주부들의 경우 식료품, 생활용품의 구매로 인해 구매금액은 적도 구매건수는 많을 것이다.
# * 월별로 데이터를 나눠 월별 구매추세 패턴을 확인

# In[49]:


p = tr.groupby('cust_id')['tran_date'].agg([
    ('1월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([1]))),
    ('2월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([2]))),
    ('3월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([3]))),
    ('4월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([4]))),
    ('5월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([5]))),
    ('6월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([6]))),
    ('7월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([7]))),
    ('8월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([8]))),
    ('9월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([9]))),
    ('10월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([10]))),
    ('11월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([11]))),
    ('12월 구매량', lambda x: np.sum(pd.to_datetime(x).dt.month.isin([12]))),
]).reset_index()
features=features.merge(p,how='left');features
#display issue로 인해 해당 시트는 features를 merge하지 않음


# In[50]:


p.hist(bins=20,figsize=(15,10))
plt.show()


# #### [파생변수 32] 주중방문비율
# * 남 녀 모두 주말 방문이 높을 것으로 예상되나 여성의 경우 주부의 주중방문이 높을 것으로 예상된다

# In[51]:


p = tr.groupby('cust_id')['tran_date'].agg([
    (('주중방문비율', lambda x: np.mean(pd.to_datetime(x).dt.dayofweek<4))),
    ]).reset_index()
features = features.merge(p, how='left'); features


# #### [파생변수 33] 가격 선호도
# * 물품별 가격 선호도가 성별 유추에 도움이 될 것 같아 추출

# In[25]:


p = tr.groupby('cust_id')['amount'].agg([
    ('총구매액',np.sum), 
    ('구매건수', np.size), 
    ('평균구매액', lambda x: np.round(np.mean(x))),
    ('최대구매액', np.max),
]).reset_index()


# In[26]:


p['가격선호도'] = p['총구매액'] / p['구매건수']
p.head()
features = features.merge(p, how='left'); features


# ### Categorical Features [파생변수 34 ~]
# * 상품카테고리와 고객이 구매한 상품카테고리는 다름 해석에 주의
# * 상품카테고리 tr['gds_grp_mclas_nm'].value_counts()
# * 고객구매상품 tr.groupby('cust_id)['gds_grp_mclas_nm]...

# In[88]:


# Data check
# print('상품대분류',tr['gds_grp_mclas_nm'].unique())
# print('상품소분류',tr['gds_grp_nm'].unique())
# print('상품ID',tr['goods_id'].unique())
# tr['gds_grp_mclas_nm'].value_counts()


# In[118]:


#상품대분류를 통한 주구매상품 추출
p=tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([
    ('주구매상품',lambda x: x.value_counts().index[0])
]).reset_index()
features=features.merge(p,how='left');features['주구매상품'].nunique()


# In[120]:


fig,ax=plt.subplots(figsize=(15,10))
sns.countplot(y='주구매상품',data=p,alpha=0.5)
#https://hwi-doc.tistory.com/entry/matplotlib-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC 참고
#https://datascienceschool.net/01%20python/05.04%20%EC%8B%9C%EB%B3%B8%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%ED%8F%AC%20%EC%8B%9C%EA%B0%81%ED%99%94.html


# #### 주구매지점
# * 온라인 쇼핑몰 더미변수를 만들고 주거래 사이트, 주방문 사이트 변수도 만들 수 있을것

# In[92]:


p=tr.groupby('cust_id')['store_nm'].agg([
    ('주구매지점',lambda x:x.value_counts().index[0])
]).reset_index()
features=features.merge(p,how='left');features


# ### 최소제곱법을 이용한 구매주기 기울기
# 
# 출처 : 교수님 제공

# In[104]:


#구매달만 추출하기 위해 자르기
tr['tran_month']=tr['tran_date'].apply(lambda x:x[5:7])

#월별 구매size를 알기 위해 cust_id와 tran_month를 각각 groupby
month=tr.groupby([tr['cust_id'],tr['tran_month']])['amount'].count().reset_index()
month.tran_month=month.tran_month.apply(lambda x:int(x))
df=pd.DataFrame(columns=['cust_id','tran_month'])

#cust_id와 구매 달을 추출해 빈 데이터프레임에 저장
for cust_id in range(0,5982):
    for tran_month in range(1,13):
        df=df.append(pd.DataFrame([[cust_id,tran_month]],columns=['cust_id','tran_month']))
df=df.merge(month,how='left')
df=df.fillna(0)

#최소제곱법을 이용하여 기울기 계산 
df1=pd.DataFrame(columns=['cust_id','기울기'])
for i in range(0,df.cust_id.nunique()):
    a=df.query('cust_id==@i').tran_month #구매 달
    b=df.query('cust_id==@i').amount #총 구매량
    c=np.polyfit(a,b,1)[0].round(2)
    df1=df1.append(pd.DataFrame([[i,c]],columns=['cust_id','구매 추세 기울기']))
features=features.merge(df1,how='left')


# #### 화장품 구매주기

# In[112]:


p=tr.query('gds_grp_mclas_nm=="화장품"').groupby('cust_id')['tran_date'].agg([
    ('화장품구매주기',lambda x:int((x.astype('datetime64').max()-x.astype('datetime64').min()).days/x.str[0:10].nunique()))
]).reset_index()


# In[115]:


p.hist(bins=20,figsize=(15,10))
plt.show()


# ### 식료품 구매관련 데이터 확인

# In[124]:


#식료품 대분류 확인
#농산물, 가공식품, 축산가공, 수산품, 육류, 차/커피, 젓갈/반찬
tr['gds_grp_mclas_nm'].value_counts()


# In[140]:


#식료품 소분류 확인
#농산가공, 청과, 야채, 유기농야채, 건과, 산지통합, 곡물, 선식(가루류), 농산단기행사, 특수야채, 식자재
#test=tr.loc[(tr.gds_grp_mclas_nm=='농산물')&(tr.gds_grp_nm),] >> ex)농산물의 하위 분류 모두 추출
test['gds_grp_nm'].value_counts()


# ### 식료품 구매비율

# In[182]:


food=['농산물', '가공식품', '축산가공', '수산품', '육류', '차/커피', '젓갈/반찬'] 

#식료품 구매비율 = 식품구매액/총구매액
p=tr.groupby('cust_id')['amount'].agg([('총구매액',np.sum)]).reset_index()
food_amount=tr.query('gds_grp_mclas_nm==@food').groupby('cust_id')['amount'].agg([
    ('식료품구매액',np.sum)
]).reset_index()
p=pd.merge(p,food_amount,on='cust_id',how='left').fillna(0)

p['식료품구매액비율']=p['식료품구매액']/p['총구매액']
p['식료품구매액비율']=np.round(p['식료품구매액비율'],2)
features=features.merge(p[['cust_id','식료품구매액비율']],how='left')


# In[ ]:


#총구매액
p=tr.groupby('cust_id')['amount'].agg([
    ('총구매액',np.sum)
]).reset_index()

#식료품구매액
food = 
#식료품구매액비율


# In[183]:


p


# In[186]:


food=['농산물', '가공식품', '축산가공', '수산품', '육류', '차/커피', '젓갈/반찬']
pk=tr.query('gds_grp_mclas_nm==@food').groupby('cust_id')['amount'].agg([
    ('식료품구매액',lambda x:round(np.sum(x)))
]).reset_index()

pk2=tr.query('gds_grp_mclas_nm==@food').groupby('cust_id')['amount'].agg([
    ('식료품구매건수',np.size)
]).reset_index()
p=pd.merge(pk,pk2,on='cust_id',how='left').fillna(0)
p['식료품평균구매액']=p['식료품구매액']/p['식료품구매건수']
p['식료품평균구매액']=np.round(p['식료품평균구매액'],)


p['식료품평균구매액분위']=pd.qcut(p['식료품평균구매액'], q=5, labels=range(1,6)).astype(int)
features = features.merge(p[['cust_id', '식료품평균구매액분위']], how = 'left')


# In[187]:


p


# In[ ]:





# ## 기타 파생변수
# * 유의하지 않은 변수로 판단하여 학습에서 제외된 또는 해석에 어려움이 있어 보류해놓은 변수들

# ### 고객들의 소비성향을 추출

# In[28]:


p['전체고객 평균 구매액']=0
p.head()
p['소비성향']=0
p.head()
p['전체고객 평균 구매액']= p['총구매액'].sum()/35000
p['소비성향'] = p['총구매액']-p['전체고객 평균 구매액']
p.head()


# ### 충동소비 경향

# In[57]:


p['충동 소비 성향'] = p['최대구매액'] - p['평균구매액'].round()
p.head()


# ### 휴면고객
# * n개월 이상 구매가 없는 고객을 휴면고객으로 정의
# * 같은 원리로 n개월 이상 구매가 없는 고객을 이탈 고객으로 정의할 수 있을것.

# In[31]:


p = tr.groupby('cust_id')['tran_date'].agg([
    ('휴면고객 여부', lambda x: np.mean(pd.to_datetime(x).dt.month.isin([9,10,11]))),
    ]).reset_index()
display(p)


# In[45]:


#f = A.groupby('cust_id')['tran_date'].agg([
#('휴면고객', lambda x:x=0, range(1))으로 3개월 간 방문기록이 0인 고객을
# 휴면고객으로 정의하려 했습니다.


# In[32]:


plt.figure(figsize=(10,8))
sns.heatmap(features.iloc[:,1:].corr(), annot=True, cmap="YlGnBu")
plt.show()


# In[33]:


display(features)
features.to_csv('features.csv', index=False, encoding='cp949')
features.info()


# In[ ]:




