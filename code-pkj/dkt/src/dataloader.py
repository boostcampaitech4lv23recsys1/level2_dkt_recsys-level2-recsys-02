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

    def get_train_data(self):
        return self.train_data
    
    def get_valid_data(self):
        return self.valid_data
    
    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.8, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = list(df.columns)
        cate_cols.remove('Timestamp')
        cate_cols.remove('answerCode')
        cate_cols.remove('userID')
        

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

#         def convert_time(s):
#             timestamp = time.mktime(
#                 datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
#             )
#             return int(timestamp)

#         df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __augmentation(self, df):
        cnt = pd.DataFrame(df.groupby('userID')['answerCode'].count())
        cnt.answerCode = cnt.answerCode//self.args.max_seq_len
        i_cnt = cnt[cnt['answerCode'] >= 1]

        n_index = -1
        for i,j in enumerate(i_cnt.index):
            c = i_cnt.iloc[i].answerCode
            index = df[df['userID'] == j].index
            for k in range(2, c+2):
                ii = index[-k*100:-100*(k-1)]
                if len(ii) > 20:
                    df.loc[ii, ['userID']] = n_index
                    n_index -= 1
        return df
    
    def __feature_engineering(self, df):
        # TODO
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        if self.args.aug:
            df = self.__augmentation(df)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        # self.args.n_questions = len(
        #     np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        # )
        # self.args.n_test = len(
        #     np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        # )
        # self.args.n_tag = len(
        #     np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        # )
            
        #df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = list(df)
        
        group = (
            df[columns]
            .groupby("userID")
            .apply(lambda r: tuple(r[col].values for col in columns))
        )
        if is_train:
            self.args.n_embdings = dict()

            for col_name in self.args.cate_feats:
                self.args.n_embdings[col_name] = len(
                    np.load(os.path.join(self.args.asset_dir, col_name + "_classes.npy"))
                )
            self.args.columns = {col_name: idx for idx, col_name in enumerate(columns)}

            # category feature location

            self.args.cate_loc = {
                col: i for i, col in enumerate(columns) if col in self.args.cate_feats
            }

            # category feature location
            self.args.conti_loc = {
                col: i for i, col in enumerate(columns) if col in self.args.conti_feats
            }

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)
        
    def load_valid_data(self, file_name):
        self.valid_data = self.load_data_from_file(file_name)
        
    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        #test, question, tag, correct = row[0], row[1], row[2], row[3]

        fea_cols = list(row)

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(fea_cols):
                fea_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        fea_cols.append(mask)
        continuos = [1]
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(fea_cols):
            if i in continuos:
                fea_cols[i] = torch.FloatTensor(col)
            else:
                fea_cols[i] = torch.tensor(col)

        return fea_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
        
    return train_loader, valid_loader
