import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.valid_data = None

    def custom_train_test_split(self, df, ratio=0.7, split=True):

        users = list(
            zip(df["userID"].value_counts().index, df["userID"].value_counts())
        )
        random.shuffle(users)

        max_train_data_len = ratio * len(df)
        sum_of_train_data = 0
        user_ids = []

        for user_id, count in users:
            sum_of_train_data += count
            if max_train_data_len < sum_of_train_data:
                break
            user_ids.append(user_id)

        train = df[df["userID"].isin(user_ids)]
        test = df[df["userID"].isin(user_ids) == False]

        # test데이터셋은 각 유저의 마지막 interaction만 추출
        test = test[test["userID"] != test["userID"].shift(-1)]
        return train, test

    def make_data(path):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = df[["userID", "answerCode"] + self.args.cate_feats + self.args.conti_feats]

        for i in self.args.cate_feats:
            n_embdings[i] = df[i].nunique()

        self.train_data, self.valid_data = custom_train_test_split(df)
