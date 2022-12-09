import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


class lightGBM:
    def __init__(self, args):
        self.args = args
        self.cate = self.args.cate_feats
        self.conti = self.args.conti_feats
        self.feats = self.cate + self.conti
        self.iter = self.args.iter
        csv_file_path = os.path.join(self.args.data_dir, self.args.file_name)
        self.df = pd.read_csv(csv_file_path)[self.feats + ['answerCode']]

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(['answerCode'], axis=1), self.df['answerCode'], test_size=0.2, shuffle=True, random_state=42)
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test)
        
        model = lgb.train(
            {'objective': 'binary'}, 
            lgb_train,
            valid_sets=[lgb_test],
            verbose_eval=200,
            num_boost_round=self.iter,
            early_stopping_rounds=500,
        )

        preds = model.predict(test[FEATS])
        acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_test, preds)

        print(f'VALID AUC : {auc} ACC : {acc}\n')
        
    def k_fold(self):
        n_splits = 5
        str_kf = StratifiedKFold(n_splits = n_splits, shuffle=True)

        y = self.df['answerCode']
        X = self.df.drop(['answerCode'], axis=1)

        for i, (train_index, test_index) in tqdm(enumerate(str_kf.split(X, y))):
            X_train, X_valid = X.loc[train_index], X.loc[test_index]
            y_train, y_valid = y.loc[train_index], y.loc[test_index]

            lgb_train = lgb.Dataset(X_train[FEATS], y_train)
            lgb_test = lgb.Dataset(X_valid[FEATS], y_valid)

            model = lgb.train(
                {'objective': 'binary'}, 
                lgb_train,
                valid_sets=[lgb_train, lgb_test],
                verbose_eval=200,
                num_boost_round=self.iter,
                early_stopping_rounds=500,
            )

            preds = model.predict(X_valid[FEATS])
            acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
            auc = roc_auc_score(y_valid, preds)

            print(f'VALID AUC : {auc} ACC : {acc}\n')