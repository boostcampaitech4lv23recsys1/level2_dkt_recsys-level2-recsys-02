import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds
import pdb

def main(args):
    #wandb.login()
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    #preprocess.load_valid_data(args.valid_name)
    preprocess.load_test_data(args.test_file_name)
    
    train_data = preprocess.get_train_data()
    #valid_data = preprocess.get_valid_data()
    test_data = preprocess.get_test_data()
    train_data, valid_data = preprocess.split_data(train_data)
    
    # print('start')
    # print(args.cate_loc)
    # print(args.conti_loc)
    
    wandb.init(project="dkt", config=vars(args))
    wandb.run.name = f'{args.model}-org-pkj'
    model = trainer.get_model(args).to(args.device)
    trainer.run(args, train_data, valid_data, test_data, model)


if __name__ == "__main__":
    args = parse_args()
    if args.pos:
        print('--position encoding model--')
    if args.aug:
        print('augmentation on')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
    
