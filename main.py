import numpy as np
import pandas as pd
import xgboost as xgb
import os

import argparse

# 1 load data
train_df = pd.read_csv('./data/train.csv') # 경로 수정 필요
test_df = pd.read_csv('./data/test.csv')

# ratio file이 존재하면 train_df와 concat
if os.path.exist('./data/ratio.csv'):
    ratio_df = pd.read_csv('./data/ratio.csv')
else:
    pass # ratio.csv 만드는 함수 넣기 

# 2 tabular preprocessing 
# concat -> pp 

# 3 divide train& test data

# 4 ML model 

# 5 submission file 