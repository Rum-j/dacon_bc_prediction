# Stacking model kfold
import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit,KFold
from sklearn.metrics import roc_auc_score
# https://github.com/h2oai/pystacknet
from pystacknet.pystacknet import StackNetClassifier
import gc

# LGB model kfold
def lgb_model_train(X, Y, xtest, sample_submission, folds, n_fold):
    lgb_submission=sample_submission.copy()
    lgb_submission['N_category'] = 0
    print('\n\n...lightgbm training...\n')
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(fold_n)
        
        lgbclf = lgb.LGBMClassifier(
            num_leaves= 512,
            n_estimators=512,
            max_depth=9,
            learning_rate=0.064,
            subsample=0.85,
            colsample_bytree=0.85,
            boosting_type= "gbdt",
            reg_alpha=0.3,
            reg_lamdba=0.243
        )
        
        X_, X_valid = X.iloc[train_index], X.iloc[valid_index]
        Y_, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
        lgbclf.fit(X_,Y_)
        
        print('finish train')
        pred=lgbclf.predict_proba(xtest)[:,1]
        val=lgbclf.predict_proba(X_valid)[:,1]
        print('finish pred')
        print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
        lgb_submission['N_category'] = lgb_submission['N_category']+pred/n_fold
        
        gc.collect()
        return lgb_submission


# XGB model kfold
def xgb_model_train(X, Y, xtest, sample_submission, folds, n_fold):
    xgb_submission=sample_submission.copy()
    xgb_submission['N_category'] = 0
    print('\n\n...lightgbm training...\n')
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(fold_n)
        xgbclf = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.9,
                # colsample_bytree=0.9,
                
                random_state=2019,
                tree_method='gpu_hist',
        )
        
        X_, X_valid = X.iloc[train_index], X.iloc[valid_index]
        Y_, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
        xgbclf.fit(X_,Y_)
        pred=xgbclf.predict_proba(xtest)[:,1]
        val=xgbclf.predict_proba(X_valid)[:,1]
        print('ROC accuracy: {}'.format(roc_auc_score(y_valid, val)))
        xgb_submission['N_category'] = xgb_submission['N_category']+pred/n_fold
        
        gc.collect()
        return xgb_submission


def importance_slice(X, Y, xtest):
    X=X.replace(np.inf,-999)
    X=X.replace(-np.inf,-999)
    xtest=xtest.replace(np.inf,-999)
    xtest=xtest.replace(-np.inf,-999)

    rf = RandomForestRegressor()
    rf.fit(X, Y)
    feature_cols=X.columns.values.tolist()

    feature_imp=sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), feature_cols), 
                reverse=True)
    print(feature_imp)
    
    feature_imp=[x[1] for x in feature_imp]
    X=X[feature_imp[:27]].values
    xtest=xtest[feature_imp[:27]].values

    print('reduce dimention finished')
    return X, xtest



def stack_model_train(X, Y, xtest, sample_submission):
    # model define
    lgbclf = lgb.LGBMRegressor(
            num_leaves= 512,
            n_estimators=512,
            max_depth=9,
            learning_rate=0.064,
            subsample=0.85,
            colsample_bytree=0.85,
            boosting_type= "gbdt",
            reg_alpha=0.3,
            reg_lamdba=0.243,
            metric="AUC"
        )
    xgbclf = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            # colsample_bytree=0.9,
            
            random_state=2019,
            tree_method='gpu_hist',
        )
    rfclf = RandomForestRegressor(n_estimators=512,
                                max_depth=5, 
                                    max_features='sqrt', 
                                    random_state=0)

    models = [[lgbclf,xgbclf], # Level 1
            [rfclf]] # Level 2

    # Specify parameters for stacked model and begin training
    model = StackNetClassifier(models, 
                            metric="auc", 
                            folds=5,
                            restacking=False,
                            use_retraining=True,
                            use_proba=True, # To use predict_proba after training
                            random_state=0,
                            n_jobs=1, 
                            verbose=1)

    # Fit the entire model tree
    model.fit(X, Y)
    stack_submission = sample_submission.copy()
    preds = model.predict_proba(xtest)[:, 1]
    stack_submission['N_category'] = preds

    return stack_submission