import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


class CatBoost:
    def __init__(self, args):
        self.args = args
        self.cate = self.args.cate_feats
        self.conti = self.args.conti_feats
        self.feats = self.cate + self.conti
        self.iter = self.args.iter
        csv_file_path = os.path.join(self.args.data_dir, self.args.file_name)
        self.df = pd.read_csv(csv_file_path)[self.feats + ["answerCode"]]
        self.df = self.df[self.cate].astype("category")

    def train(self):
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            self.df.drop(["answerCode"], axis=1),
            self.df["answerCode"],
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )

        catboost_cl = CatBoostClassifier(cat_features=self.cate, n_estimators=self.iter)
        # catboost_cl = CatBoostRegressor(cat_features=self.cate, n_estimators=2500)
        catboost_cl.fit(
            X_train_cat, y_train_cat, verbose=100, eval_set=(X_test_cat, y_test_cat)
        )
        preds = catboost_cl.predict_proba(X_test_cat)[:, 1]
        acc = accuracy_score(y_test_cat, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_test_cat, preds)

        print(f"VALID AUC : {auc} ACC : {acc}\n")

    def k_fold(self):
        n_splits = 5
        str_kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        y = self.df["answerCode"]
        X = self.df.drop(["answerCode"], axis=1)

        for i, (train_index, test_index) in tqdm(enumerate(str_kf.split(X, y))):
            X_train_cat, X_test_cat = X.loc[train_index], X.loc[test_index]
            y_train_cat, y_test_cat = y.loc[train_index], y.loc[test_index]

            catboost_cl = CatBoostClassifier(
                cat_features=self.cate, n_estimators=self.iter
            )
            catboost_cl.fit(
                X_train_cat, y_train_cat, verbose=100, eval_set=(X_test_cat, y_test_cat)
            )
            preds = catboost_cl.predict_proba(X_test_cat)[:, 1]
            acc = accuracy_score(y_test_cat, np.where(preds >= 0.5, 1, 0))
            auc = roc_auc_score(y_test_cat, preds)

            print(f"VALID AUC : {auc} ACC : {acc}\n")

            # catboost_cl.save_model(f'catboost_info/model/catboost_{i}')


# import matplotlib.pyplot as plt
# import seaborn as sns
# def plot_feature_importance(importance,names,model_type):

#     #Create arrays from feature importance and feature names
#     feature_importance = np.array(importance)
#     feature_names = np.array(names)

#     #Create a DataFrame using a Dictionary
#     data={'feature_names':feature_names,'feature_importance':feature_importance}
#     fi_df = pd.DataFrame(data)

#     #Sort the DataFrame in order decreasing feature importance
#     fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

#     #Define size of bar plot
#     plt.figure(figsize=(10,8))
#     #Plot Searborn bar chart
#     sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
#     #Add chart labels
#     plt.title(model_type + 'FEATURE IMPORTANCE')
#     plt.xlabel('FEATURE IMPORTANCE')
#     plt.ylabel('FEATURE NAMES')

# #plot the catboost result
# plot_feature_importance(catboost_cl.get_feature_importance(),[i for i in df.columns if i != 'rating'],'CATBOOST')
