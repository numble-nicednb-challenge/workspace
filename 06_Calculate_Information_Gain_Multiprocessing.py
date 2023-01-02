import multiprocessing
from multiprocessing import Pool
import pandas as pd
from functools import reduce
import time
from datetime import timedelta
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

df1 = pd.read_csv('./custom_data/basic_finance_ratio_data.csv', encoding='cp949')
target = pd.read_csv('./custom_data/basic_finance_data.csv', encoding='cp949')

df1.rename(columns={'사업자번호':'사업자등록번호'}, inplace=True)
df1['결산년월'] = df1['결산년월'].astype(str)
idx = df1.loc[df1['결산년월'].str.contains('2018')].index
df1.drop(idx, inplace=True)

target = target[['사업자등록번호', '휴폐업구분']]

df2_no_dup = target.drop_duplicates(subset='사업자등록번호')

for num, ind in zip(df2_no_dup['사업자등록번호'], df2_no_dup['휴폐업구분']):
    df1.loc[(df1['사업자등록번호']==num), "휴폐업구분"] = ind

def change_target(value):
    if value in ['폐업', '휴업']:
        return 1
    return 0

df1['휴폐업구분'] = df1['휴폐업구분'].apply(change_target)

df_train = df1.iloc[:, 2:]

def calculate_ig(col_list):
    result = []
    df_new = df_train[col_list]
    result.append(getGainA(df_new, "휴폐업구분"))
    return result

def getEntropy(df, feature) :
    D_len = df[feature].count() # 데이터 전체 길이
    # reduce함수를 이용하여 초기값 0에 
    # 각 feature별 count을 엔트로피 식에 대입한 값을 순차적으로 더함
    return reduce(lambda x, y: x+(-(y[1]/D_len) * np.log2(y[1]/D_len)), \
                df[feature].value_counts().items(), 0)
                
def get_target_true_count(col, name, target, true_val, df):
    """
    df[col]==name인 조건에서 Target이 참인 경우의 갯수를 반환
    """
    if df.groupby([col,target]).size()[name].index.tolist() == [0]:
        return 0
    return df.groupby([col,target]).size()[name][true_val]

def NoNan(x):
    """
    Nan의 경우 0을 반환
    """
    return np.nan_to_num(x)

def getGainA(df, feature) :
    info_D = getEntropy(df, feature) # 목표변수 Feature에 대한 Info(Entropy)를 구한다.
    columns = list(df.loc[:, df.columns != feature]) # 목표변수를 제외한 나머지 설명변수들을 리스트 형태로 저장한다.
    gains = []
    D_len = df.shape[0] # 전체 길이
    for col in columns:
        info_A = 0
        # Col내 개별 Class 이름(c_name)과 Class별 갯수(c_len)
        for c_name, c_len in df[col].value_counts().items():
            target_true = get_target_true_count(col, c_name, feature, 1, df)
            prob_t = target_true / c_len
            # Info_A <- |Dj|/|D| *  Entropy(label) | NoNan을 이용해 prob_t가 0인 경우 nan이 나와 생기는 오류 방지
            info_A += (c_len/D_len) * -(NoNan(prob_t*np.log2(prob_t)) + NoNan((1 - prob_t)*np.log2(1 - prob_t)))
        gains.append(info_D - info_A)
    result = dict(zip(columns,gains)) # 각 변수에 대한 Information Gain 을 Dictionary 형태로 저장한다.

    filename = df.columns[0]
    with open(f'./json_file_new/{filename}.json','w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    start = time.process_time()

    col_list = []
    for i in range(0, 18, 3):
        start = i
        end = i+3
        tmp = df_train.columns.tolist()[start:end]
        tmp.append('휴폐업구분')
        col_list.append(tmp)

    for i in range(18, 30, 2):
        start = i
        end = i+2
        tmp = df_train.columns.tolist()[start:end]
        tmp.append('휴폐업구분')
        col_list.append(tmp)

    cpu_count = multiprocessing.cpu_count()

    p = Pool(processes=cpu_count)
    p.map(calculate_ig, col_list)
    
    end = time.process_time()
    print("Time elapsed: ", timedelta(seconds=end-start))