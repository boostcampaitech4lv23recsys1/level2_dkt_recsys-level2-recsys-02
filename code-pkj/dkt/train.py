import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds
import pdb
#from src.boost import CatBoost, lightGBM
import random

def main(args):
    #wandb.login()
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Preprocess(args)
    #preprocess.load_train_data(args.file_name)
    #preprocess.load_valid_data(args.valid_name)
    #preprocess.load_test_data(args.test_file_name)
    
    #train_data = preprocess.get_train_data()
    #valid_data = preprocess.get_valid_data()
    #test_data = preprocess.get_test_data()
    rd = random.randint(0, 10000)
    #train_data, valid_data = preprocess.split_data(train_data, seed=rd)
    train_data, valid_data = preprocess.split_data2(args, seed=rd)
    print('random data seed :', rd)
    # print('start')
    # print(args.cate_loc)
    # print(args.conti_loc)
    
    wandb.init(project="dkt", config=vars(args))
    wandb.run.name = f'{args.model}-org-pkj'
    model = trainer.get_model(args)
    trainer.run2(args, train_data, valid_data, model)


if __name__ == "__main__":
    args = parse_args()
    print()
    print('-'*20, 'conti_feats', '-'*19)
    print(args.conti_feats)
    print('-'*20, 'cate_feats', '-'*20)
    print(args.cate_feats)
    print('-'*52)
    if args.model == 'cat':
        cat = CatBoost()
        if args.fold:
            cat.k_fold()
        else:
            cat.train()
    elif args.model == 'lgbm':
        lgbm = lightGBM()
        if args.fold:
            lgbm.k_fold()
        else:
            lgbm.train()
    else:
        if args.pos:
            print('--position encoding model--')
        if args.aug:
            print('augmentation on')
        os.makedirs(args.model_dir, exist_ok=True)
        main(args)
    
