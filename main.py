import numpy as np
import pandas as pd
import os
from utils.tabular_preprocess import preprocess_all
from utils.feature_extraction import feature_extraction
from model.Stacking_model import lgb_model_train, xgb_model_train, importance_slice, stack_model_train
from sklearn.model_selection import TimeSeriesSplit,KFold
import argparse

# 0 load data
train_df = pd.read_csv('./data/train.csv') # 경로 수정 필요
test_df = pd.read_csv('./data/test.csv')

# 1 ratio file이 존재하면 train_df와 concat
if os.path.exist('./data/ratio.csv'):
    ratio_df = pd.read_csv('./data/ratio.csv')
else:
    pass # ratio.csv 만드는 함수 넣기 


# 2 crop된 image 폴더가 없으면 image crop 진행
if os.path.exist('./data/cropped_image'):
    image_path = './data/cropped_image'
else:
    pass # image crop 진행

# 3 feature extraction
for t in ['new_train', 'new_test']:
    image_folder = os.path.join(image_path, t)

    if t == 'new_train':
        train_df = feature_extraction(train_df, image_folder)
    else:
        test_df = feature_extraction(test_df, image_folder)

# 4 tabular preprocessing 
# concat -> pp 
train_df, test_df = preprocess_all(train_df, test_df)
    

# 5 divide train& test data
n_fold = 5
folds = KFold(n_splits=n_fold,shuffle=True)

Y = train_df['N_category']
X = train_df.drop(columns = ['N_category'])
xtest = test_df.iloc[:,:]
sample_submission = pd.read_csv('./data/sample_submission.csv')


# 6 ML model 
lgb_submission = lgb_model_train(X, Y, xtest, sample_submission, folds, n_fold)
xgb_submission = xgb_model_train(X, Y, xtest, sample_submission, folds, n_fold)

X, xtest = importance_slice(X, Y, xtest)
stack_submission = stack_model_train(X, Y, xtest, sample_submission)


# 7 submission file
# voting
ensemble=sample_submission.copy()
ensemble.N_category=lgb_submission.N_category*0.4+xgb_submission.N_category*0.4+stack_submission.N_category*0.2
ensemble.N_category = np.where(ensemble.N_category >= 0.5 , 1, 0)
ensemble.to_csv('/submits/xgb_lgb_stacking305.csv', index=False)
