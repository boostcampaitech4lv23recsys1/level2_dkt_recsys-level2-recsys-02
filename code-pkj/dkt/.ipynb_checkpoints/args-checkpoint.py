import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--data_seed", default=42, type=int, help="seed")
    
    parser.add_argument("--device", default="cuda", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )
    
    parser.add_argument(
        "--aug", default=False, action='store_true', help="augmentation"
    )
    
    parser.add_argument(
        "--file_name", default="pkj.csv", type=str, help="train file name"
    )
    
    parser.add_argument(
        "--valid_name", default="validation.csv", type=str, help="train file name"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test.csv", type=str, help="test file name"
    )
    parser.add_argument(
        "--n_embdings", default={}, type=dict, help="Feature number"
    )
    parser.add_argument(
        "--allF", default=0, type=int, help="Feature nunique"
    )
    # parser.add_argument('--cate_feats', type=list,
    #                     default= ['assessmentItemID', 'testId',
    #    'KnowledgeTag', 'problem_number', 'last_problem', 'hour', 'dow',
    #    'elapsed', 'grade', 'mid', 'assessmentItemID0', 'assessmentItemID1',
    #     'solve_order', 'solved_disorder'], help='category feature')
    
    # parser.add_argument('--cate_feats', type=list,
    #                     default=['KnowledgeTag', 'month', 'hour', 'week', 'elapsed_cate',
    #    'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2',
    #    ], help='category feature')

    # parser.add_argument('--cate_feats', type=list, default=['KnowledgeTag', 'month', 'hour', 'week', 'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2', 'elapsed_cate', 'week_hour', 'day',
    #                                                        'solve_order'])
    # parser.add_argument('--cate_feats', type=list, default=['KnowledgeTag', 'month', 'hour', 'week', 'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2', 'elapsed_cate', 'week_hour'])
    parser.add_argument('--cate_feats', type=list, default=['KnowledgeTag', 'month', 'hour', 'week', 'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2', 'testId0', 'testId1', 'elapsed_cate', 'as0_as1', 'as0_as2', 'as1_as2', 'assessmentItemID', 'week_hour'])
    
    # parser.add_argument('--cate_feats', type=list, default=['KnowledgeTag',  'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2', 'testId0', 'testId1', 'elapsed_cate',  'assessmentItemID',])
    ### continous featurs

    parser.add_argument('--org_feats', type=list, default=['KnowledgeTag', 'assessmentItemID0', 'assessmentItemID1', 'assessmentItemID2', 'week', 'hour', 'week_hour'])
    
    parser.add_argument('--fe_feats', type=list, default=['elapsed_cate', 'week_hour', 'as0_as1', 'as0_as2', 'as1_as2', 'month', 'hour', 'week',])

    # parser.add_argument('--conti_feats', type=list, default=['tag_mean', 'tag_std', 'ass0_mean', 'ass0_std', 'ass1_mean', 'ass1_std', 'ass2_mean', 'ass2_std', 'Time', 'elapsed_log'])
    # parser.add_argument('--conti_feats', type=list, default=['tag_mean', 'tag_std', 'ass0_mean', 'ass0_std', 'ass1_mean', 'ass1_std', 'ass2_mean', 'ass2_std', 'elapsed', 'user_acc', 'user_correct_answer' ])
    
    parser.add_argument('--conti_feats', type=list, default=['tag_mean', 'tag_std', 'ass0_mean', 'ass0_std', 'ass1_mean', 'ass1_std', 'ass2_mean', 'ass2_std', 'test0_mean', 'test0_std',
    'test1_mean', 'test1_std', 'elapsed',])
    
    parser.add_argument(
        "--max_seq_len", default=100, type=int, help="max sequence length"
    )
    
    parser.add_argument(
        "--iter", default=1000, type=int, help="n_estimators"
    )
    
    parser.add_argument(
        "--bert_layers", default=1, type=int, help="number of bert layers"
    )
    
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    
    parser.add_argument("--isinfer", default=False, type=bool, help="train or inference")
    
    # 모델
    parser.add_argument(
        "--hidden_dim", default=100, type=int, help="hidden dimension size"
    )
    
    parser.add_argument(
        "--hidden_size", default=496, type=int, help="hidden dimension size"
    )
    
    parser.add_argument("--n_layers", default=1, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=16, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")

    # 훈련
    parser.add_argument("--n_epochs", default=6, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=0.0003, type=float, help="learning rate") # 0.003788
    parser.add_argument("--clip_grad", default=20, type=int, help="clip grad")
    parser.add_argument("--patience", default=100, type=int, help="for early stopping")

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstmattn", type=str, help="model type")
    parser.add_argument("--optimizer", default="adamW", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )
    
    parser.add_argument("--pos", default=False, action='store_true', help="position encoding")

    parser.add_argument("--window", default=True, action='store_true', help="sliding window")
    parser.add_argument("--stride", default=101, type=int, help="window stride")
    parser.add_argument("--shuffle", default=False, action='store_true', help="data shuffle option")
    parser.add_argument("--shuffle_n", default= 3, type=int)
    parser.add_argument("--fold", default=False, action='store_true', help="k-fold (5)")
    
    args = parser.parse_args()
    return args
