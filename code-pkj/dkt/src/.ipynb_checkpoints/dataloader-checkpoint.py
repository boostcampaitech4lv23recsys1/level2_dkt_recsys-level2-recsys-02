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
    def split_data2(self, args, ratio=0.8, shuffle=True, seed=0):
        
        csv_file_path = os.path.join(args.data_dir, args.file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = df[['answerCode'] + args.cate_feats + args.conti_feats]
        df = df.sample(frac=1).reset_index(drop=True)
        # if shuffle:
        #     random.seed(seed)  # fix to default seed 0
        #     random.shuffle(data)

        size = int(len(df) * ratio)
        data_1 = df[:size]
        data_2 = df[size:]

        return data_1, data_2
    
    def custom_train_test_split(self, df, ratio=0.7, split=True):
        
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

        #test??????????????? ??? ????????? ????????? interaction??? ??????
        #test = test[test['userID'] != test['userID'].shift(-1)]
        return train, test

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = list(df.columns)
        #cate_cols.remove('Timestamp')
        cate_cols.remove('answerCode')
        cate_cols.remove('userID')
        #print(cate_cols)

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

#         for col in cate_cols:

#             le = LabelEncoder()
#             if is_train:
#                 # For UNKNOWN class
#                 try:
#                     label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
#                     le.classes_ = np.load(label_path)
                    
#                     df[col] = df[col].apply(
#                         lambda x: x if str(x) in le.classes_ else "unknown"
#                     )
#                 except:
#                     a = df[col].unique().tolist() + ["unknown"]
#                     le.fit(a)
#                     self.__save_labels(le, col)
#             else:
#                 label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
#                 le.classes_ = np.load(label_path)

#                 df[col] = df[col].apply(
#                     lambda x: x if str(x) in le.classes_ else "unknown"
#                 )

#             # ?????? ????????? ?????????????????? ??????
#             df[col] = df[col].astype(str)
#             test = le.transform(df[col])
#             df[col] = test

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
        # cate = ['recCount']
        # for i in cate:
        #     cate2label = {j:i for i,j in enumerate(df[i].unique())}
        #     df[i] = df[i].map(cate2label)
        # group = df.groupby(['userID'])
        # df = group.filter(lambda x:len(x) >= self.args.max_seq_len)
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = df[['userID', 'answerCode'] + self.args.cate_feats + self.args.conti_feats]
        # if is_train:
        #     df, valid = self.custom_train_test_split(df)

        if self.args.aug:
            df = self.__augmentation(df)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # ?????? feature??? embedding??? ?????? embedding_layer??? input ????????? ???????????? ??????

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
                self.args.n_embdings[col_name] = df[col_name].nunique()

            self.args.columns = {col_name: idx for idx, col_name in enumerate(columns)}

            # category feature location

            self.args.cate_loc = {
                col: i for i, col in enumerate(columns) if col in self.args.cate_feats
            }

            # category feature location
            self.args.conti_loc = {
                col: i for i, col in enumerate(columns) if col in self.args.conti_feats
            }
            
#             columns = list(valid)
        
#             group_v = (
#                 valid[columns]
#                 .groupby("userID")
#                 .apply(lambda r: tuple(r[col].values for col in columns))
#             )

        return group.values#, group_v.values

    def load_train_data(self, file_name):
        #self.train_data, self.valid_data = self.load_data_from_file(file_name)
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
        
        # ??? data??? sequence length
        seq_len = len(row[0])

        #test, question, tag, correct = row[0], row[1], row[2], row[3]

        fea_cols = list(row)
        
        index = np.arange(self.args.max_seq_len)
        eids = np.random.permutation(index)
        
        # max seq len??? ??????????????? ????????? ?????? ????????? ?????? ?????? ????????? ?????????
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(fea_cols):
                fea_cols[i] = col[-self.args.max_seq_len :][eids]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)[eids]
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)[eids]
            mask[-seq_len:] = 1

        # mask??? columns ????????? ????????????
        fea_cols.append(mask)

        # np.array -> torch.tensor ?????????
        for i, col in enumerate(fea_cols):
            if i in self.args.conti_loc.values():
                fea_cols[i] = torch.FloatTensor(col)
            else:
                fea_cols[i] = torch.tensor(col)
        #print(fea_cols[0].shape, len(fea_cols))
        return fea_cols

    def __len__(self):
        return len(self.data)

class DKTDataset2(torch.utils.data.Dataset):
    def __init__(self, df, args, embed):
        self.df = df
        self.args = args
        self.seq_len = args.max_seq_len
        self.cate = args.cate_feats
        self.conti = args.conti_feats
        self.embed = embed
    def __getitem__(self, index):
        data = self.df[index*self.seq_len:(index+1)*self.seq_len]
        cate = torch.tensor(data[self.cate].values)
        conti = torch.FloatTensor(data[self.conti].values)
        
        for i, j in enumerate(self.embed):
            cate[:, i] += j
        
        answer = data['answerCode'].values
        
        return cate, conti, answer

    def __len__(self):
        return int(len(self.df) // self.seq_len) -1
    
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])
    #print(np.array(batch).shape)
    # batch??? ????????? ??? column?????? ?????????
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            # if torch.Size([100, 4]) == col.shape:
            #     print(col)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)

def collate2(batch):
#     col_n = len(batch[0])
#     col_list = [[] for _ in range(col_n)]
#     max_seq_len = len(batch[0][-1])
#     #print(np.array(batch).shape)
#     # batch??? ????????? ??? column?????? ?????????
#     for row in batch:
#         for i, col in enumerate(row):
#             pre_padded = torch.zeros(max_seq_len)
#             # if torch.Size([100, 4]) == col.shape:
#             #     print(col)
#             pre_padded[-len(col) :] = col
#             col_list[i].append(pre_padded)

#     for i, _ in enumerate(col_list):
#         col_list[i] = torch.stack(col_list[i])

    return tuple(batch[:5])

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

def get_loaders2(args, train, valid, embed):
    
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset2(train, args, embed)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
      #      collate_fn=collate2,
        )
    if valid is not None:
        valset = DKTDataset2(valid, args, embed)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
    #        collate_fn=collate2,
        )
        
    return train_loader, valid_loader

def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = window_size + 1

    augmented_datas = []
    if args.shuffle:
        print('-'*20, 'shuffle', '-'*20)
    for row in data:
        seq_len = len(row[0])

        # ?????? window ???????????? seq len??? ????????? ????????? augmentation??? ?????? ?????????
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1

            # ??????????????? slidding window ??????
            for window_i in range(total_window):
                # window??? ?????? ???????????? ????????? ?????????
                window_data = []
                for col in row:
                    window_data.append(
                        col[window_i * stride : window_i * stride + window_size]
                    )

                # Shuffle
                # ????????? ???????????? ?????? shuffle??? ?????? ?????????
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window?????? ???????????? ????????? ?????? ??????
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))

    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle ???????????? window??? ???????????? ?????? ????????? ???????????? ??????
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.window == True:
        print("\n[4-1] Do Sliding Window Augmentation")
        data = slidding_window(data, args)

    return data