import numpy as np
import optuna
from optuna.integration import CatBoostPruningCallback
import pandas as pd
import catboost as cb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import random

def custom_train_test_split(df, ratio=0.7, split=True):
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def objective(trial: optuna.Trial) -> float:
    FEATS = [
       'KnowledgeTag', 'month', 'hour', 'week', 'elapsed', 'elapsed_cate',
       'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2',
       'testId0', 'testId1', 'test0_mean', 'test0_std', 'test1_mean',
       'test1_std', 'tag_mean', 'tag_std', 'ass0_mean', 'ass0_std',
       'ass1_mean', 'ass1_std', 'ass2_mean', 'ass2_std',
       'user_total_answer',]

    cate = ['KnowledgeTag', 'month', 'hour', 'week', 'elapsed_cate', 'testId0', 'testId1',
           'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2']

    df = pd.read_csv('../../data/pkj.csv')
        
    train, test = custom_train_test_split(df)
    
    y_train_cat = train['answerCode']
    X_train_cat = train.drop(['answerCode'], axis=1)[FEATS]

    y_test_cat = test['answerCode']
    X_test_cat = test.drop(['answerCode'], axis=1)[FEATS]

    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "eval_metric": "AUC",
    }

    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

    gbm = cb.CatBoostClassifier(**param)

    pruning_callback = CatBoostPruningCallback(trial, "AUC")
    gbm.fit(
        X_train_cat,
        y_train_cat,
        eval_set=[(X_test_cat, y_test_cat)],
        verbose=0,
        early_stopping_rounds=100,
        callbacks=[pruning_callback],
        cat_features = cate
    )

    # evoke pruning manually.
    pruning_callback.check_pruned()
    
    preds = gbm.predict_proba(X_test_cat)[:,1]
    #acc = accuracy_score(y_test_cat, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test_cat, preds)
    # preds = gbm.predict(valid_x)
    # pred_labels = np.rint(preds)
    # accuracy = accuracy_score(valid_y, pred_labels)
    
    return auc


if __name__ == "__main__":
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))