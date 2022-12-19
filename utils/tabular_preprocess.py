import pandas as pd
import datetime as dt

# 한글 컬럼명 영문으로 변경
def en_ko(train_df, test_df):
    train_df = train_df.rename(columns={'나이':'age','수술연월일':'date', '진단명':'Diagnosis', '암의 위치':'location', '암의 개수':'count', '암의 장경':'size', 'DCIS_or_LCIS_여부':'DCIS_or_LCIS_existence'}) 
    test_df = test_df.rename(columns={'나이':'age','수술연월일':'date', '진단명':'Diagnosis', '암의 위치':'location', '암의 개수':'count', '암의 장경':'size', 'DCIS_or_LCIS_여부':'DCIS_or_LCIS_existence'}) 
    return train_df, test_df

# date 변환
def daysconvert(train_df, test_df):
    someday = pd.to_datetime('2000-01-01')
    train_df['date'] = pd.to_datetime(train_df['date']) - someday
    train_df['date'] = train_df['date'].dt.days / 1000

    test_df['date'] = pd.to_datetime(test_df['date']) - someday
    test_df['date'] = test_df['date'].dt.days / 1000
    return train_df, test_df

# 0이 의미가 있는 컬럼들은 숫자를 1씩 올려줘서 nan값이 채워질 0과 구분할수 있게 해준다.
def upgrade(train_df, test_df):
    train_df['DCIS_or_LCIS_existence'] += 1
    train_df['ER'] += 1
    train_df['HER2'] += 1
    train_df['HER2_IHC'] += 1
    train_df['PR'] += 1
    train_df['T_category'] += 1
    train_df['BRCA_mutation'] += 1
    train_df['HER2_SISH'] += 1

    test_df['DCIS_or_LCIS_existence'] += 1
    test_df['ER'] += 1
    test_df['HER2'] += 1
    test_df['HER2_IHC'] += 1
    test_df['PR'] += 1
    test_df['T_category'] += 1
    test_df['BRCA_mutation'] += 1
    test_df['HER2_SISH'] += 1
    return train_df, test_df

# 이상치 휴먼에러 제거
def change_outlier(train_df):
    train_df['PR_Allred_score'].loc[train_df['PR_Allred_score']==23] = 3
    train_df['PR_Allred_score'].loc[train_df['PR_Allred_score']==54] = 4
    return train_df

# one hot encoding
def cate_onehot(train_df, test_df):
    train_df['location_left'] = train_df['location'].apply(lambda x :1 if x==1 else 0)
    train_df['location_right'] = train_df['location'].apply(lambda x :1 if x==2 else 0)
    train_df['location_both'] = train_df['location'].apply(lambda x :1 if x==3 else 0)

    train_df['Diagnosis_ductal'] = train_df['Diagnosis'].apply(lambda x :1 if x==1 else 0)
    train_df['Diagnosis_lobular'] = train_df['Diagnosis'].apply(lambda x :1 if x==2 else 0)
    train_df['Diagnosis_mucinous'] = train_df['Diagnosis'].apply(lambda x :1 if x==3 else 0)
    train_df['Diagnosis_other'] = train_df['Diagnosis'].apply(lambda x :1 if x==4 else 0)

    test_df['location_left'] = test_df['location'].apply(lambda x :1 if x==1 else 0)
    test_df['location_right'] = test_df['location'].apply(lambda x :1 if x==2 else 0)
    test_df['location_both'] = test_df['location'].apply(lambda x :1 if x==3 else 0)

    test_df['Diagnosis_ductal'] = test_df['Diagnosis'].apply(lambda x :1 if x==1 else 0)
    test_df['Diagnosis_lobular'] = test_df['Diagnosis'].apply(lambda x :1 if x==2 else 0)
    test_df['Diagnosis_mucinous'] = test_df['Diagnosis'].apply(lambda x :1 if x==3 else 0)
    test_df['Diagnosis_other'] = test_df['Diagnosis'].apply(lambda x :1 if x==4 else 0)

    train_df = train_df.drop(columns = ['location','Diagnosis',])
    test_df = test_df.drop(columns = ['location','Diagnosis',])
    return train_df, test_df

# 결측치 채우기
def fill_missing(train_df, test_df):
    train_df['KI-67_LI_percent'].loc[(train_df['KI-67_LI_percent'].isnull() == True) & (train_df['T_category']==1)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==1].median()
    train_df['KI-67_LI_percent'].loc[(train_df['KI-67_LI_percent'].isnull() == True) & (train_df['T_category']==2)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==2].median()
    train_df['KI-67_LI_percent'].loc[(train_df['KI-67_LI_percent'].isnull() == True) & (train_df['T_category']==3)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==3].median()
    train_df['KI-67_LI_percent'].loc[(train_df['KI-67_LI_percent'].isnull() == True) & (train_df['T_category']==4)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==4].median()

    test_df['KI-67_LI_percent'].loc[(test_df['KI-67_LI_percent'].isnull() == True) & (test_df['T_category']==1)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==1].median()
    test_df['KI-67_LI_percent'].loc[(test_df['KI-67_LI_percent'].isnull() == True) & (test_df['T_category']==2)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==2].median()
    test_df['KI-67_LI_percent'].loc[(test_df['KI-67_LI_percent'].isnull() == True) & (test_df['T_category']==3)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==3].median()
    test_df['KI-67_LI_percent'].loc[(test_df['KI-67_LI_percent'].isnull() == True) & (test_df['T_category']==4)] = train_df['KI-67_LI_percent'].loc[train_df['T_category']==4].median()

    train_df['size'].loc[(train_df['size'].isnull() == True) & (train_df['T_category']==1)] = train_df['size'].loc[train_df['T_category']==1].median()
    train_df['size'].loc[(train_df['size'].isnull() == True) & (train_df['T_category']==2)] = train_df['size'].loc[train_df['T_category']==2].median()
    train_df['size'].loc[(train_df['size'].isnull() == True) & (train_df['T_category']==3)] = train_df['size'].loc[train_df['T_category']==3].median()
    train_df['size'].loc[(train_df['size'].isnull() == True) & (train_df['T_category']==4)] = train_df['size'].loc[train_df['T_category']==4].median()

    test_df['size'].loc[(test_df['size'].isnull() == True) & (test_df['T_category']==1)] = train_df['size'].loc[train_df['T_category']==1].median()
    test_df['size'].loc[(test_df['size'].isnull() == True) & (test_df['T_category']==2)] = train_df['size'].loc[train_df['T_category']==2].median()
    test_df['size'].loc[(test_df['size'].isnull() == True) & (test_df['T_category']==3)] = train_df['size'].loc[train_df['T_category']==3].median()
    test_df['size'].loc[(test_df['size'].isnull() == True) & (test_df['T_category']==4)] = train_df['size'].loc[train_df['T_category']==4].median()

    train_df['PR_Allred_score'].loc[(train_df['PR_Allred_score'].isnull() == True) & (train_df['NG']==1)] = train_df['PR_Allred_score'].loc[train_df['NG']==1].mean().astype(int)
    train_df['PR_Allred_score'].loc[(train_df['PR_Allred_score'].isnull() == True) & (train_df['NG']==2)] = train_df['PR_Allred_score'].loc[train_df['NG']==2].mean().astype(int)
    train_df['PR_Allred_score'].loc[(train_df['PR_Allred_score'].isnull() == True) & (train_df['NG']==3)] = train_df['PR_Allred_score'].loc[train_df['NG']==3].mean().astype(int)

    test_df['PR_Allred_score'].loc[(test_df['PR_Allred_score'].isnull() == True) & (test_df['NG']==1)] = train_df['PR_Allred_score'].loc[train_df['NG']==1].mean().astype(int)
    test_df['PR_Allred_score'].loc[(test_df['PR_Allred_score'].isnull() == True) & (test_df['NG']==2)] = train_df['PR_Allred_score'].loc[train_df['NG']==2].mean().astype(int)
    test_df['PR_Allred_score'].loc[(test_df['PR_Allred_score'].isnull() == True) & (test_df['NG']==3)] = train_df['PR_Allred_score'].loc[train_df['NG']==3].mean().astype(int)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    return train_df, test_df

# 필수삭제 columns
def del_columns(train_df, test_df):
    train_df = train_df.drop(columns = ['ID','img_path','mask_path'])
    test_df = test_df.drop(columns = ['ID','img_path'])
    return train_df, test_df

def preprocess_all(train_df, test_df):
    train_df, test_df = en_ko(train_df), en_ko(test_df)
    train_df, test_df = daysconvert(train_df), daysconvert(test_df)
    train_df, test_df = upgrade(train_df), upgrade(test_df)
    train_df = change_outlier(train_df)
    train_df, test_df = cate_onehot(train_df), cate_onehot(test_df)
    train_df, test_df = fill_missing(train_df), fill_missing(test_df)
    train_df, test_df = del_columns(train_df), del_columns(test_df)
    return train_df, test_df