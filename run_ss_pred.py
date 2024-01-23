
import argparse

import torch
from RNABERT.rnabert import BertModel
from RNAFM import fm
from RNAMSM.model import MSATransformer
from utils import get_config
from torch.optim import Adam
from utils import str2bool, str2list
import os.path as osp

from tokenizer import RNATokenizer
import param_turner2004
from ss_pred import (
    RNABertForSsp,
    MixedFold,
    RNAFmForSsp,
    RNAMsmForSsp,
)
from metrics import SspMetrics
from collators import SspCollator
from losses import StructuredLoss
from datasets import BPseqDataset
from trainers import SspTrainer

# ========== Define constants
MODELS = ["RNABERT", "RNAMSM", "RNAFM"]
TASKS = ["RNAStrAlign", "bpRNA1m"]
MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512}
EMBED_DIMS = {"RNABERT": 120,
              "RNAMSM": 768,
              "RNAFM": 640}
DATASETS = {
    "RNAStrAlign": ("RNAStrAlign600.lst", "archiveII600.lst"),
    "bpRNA1m": ("TR0.lst", "TS0.lst"),
}
# ========== Configuration
parser = argparse.ArgumentParser(
    description='RNA secondary structure prediction using deep learning with thermodynamic integrations', add_help=True)

# model args
parser.add_argument('--model_name', type=str, default="RNABERT", choices=MODELS)
parser.add_argument('--vocab_path', type=str, default="./vocabs/")
parser.add_argument('--pretrained_model', type=str,
                    default="./checkpoints/")
parser.add_argument('--config_path', type=str,
                    default="./configs/")

parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--max_seq_len', type=int, default=0)
# data args
parser.add_argument('--task_name', type=str, default="RNAStrAlign",
                    choices=TASKS, help='Task name of training data.')
parser.add_argument('--dataloader_num_workers', type=int,
                    default=8, help='The number of threads used by dataloader.')
parser.add_argument('--dataloader_drop_last', type=str2bool,
                    default=True, help='Whether drop the last batch sample.')
parser.add_argument('--dataset_dir', type=str,
                    default="./data/ssp", help='Local path for dataset.')
parser.add_argument('--replace_T', type=bool, default=True)
parser.add_argument('--replace_U', type=bool, default=False)
# training args
parser.add_argument('--disable_tqdm', type=str2bool,
                    default=False, help='Disable tqdm display if true.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--device', type=str, default='cpu')

parser.add_argument('--num_train_epochs', type=int, default=50,
                    help='The number of epoch for training.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='The number of samples used per step, must be 1.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='the learning rate for optimizer (default: 0.001)')
parser.add_argument('--metrics',
                    type=str2list,
                    default="F1s,Accuracy,Precision,Recall",
                    help='Use which metrics to evaluate model, could be concatenate by ",".')

# logging args
parser.add_argument('--output', type=str,
                    default="./output_ft/ssp", help='Logging directory.')
parser.add_argument('--visualdl_dir', type=str,
                    default="visualdl", help='Visualdl logging directory.')
parser.add_argument('--logging_steps', type=int, default=100,
                    help='Update visualdl logs every logging_steps.')
parser.add_argument('--save_max', type=str2bool, default=True,
                    help='Save model with max metric.')

args = parser.parse_args()


if __name__ == "__main__":
    # ========== post process
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
    args.dataset_train = osp.join(args.dataset_dir, DATASETS[args.task_name][0])
    args.dataset_test = osp.join(args.dataset_dir, DATASETS[args.task_name][1])

    # ========== args check
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # ========== set device
    torch.cuda.set_device(args.device)

    # ========== Build tokenizer, model, criterion
    tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))

    if args.model_name == "RNABERT":
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        pretrained_model = BertModel(model_config)
        pretrained_model = RNABertForSsp(pretrained_model)
        pretrained_model._load_pretrained_bert(
            args.pretrained_model+"{}.pth".format(args.model_name))
    elif args.model_name == "RNAMSM":
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        pretrained_model = MSATransformer(**model_config)
        pretrained_model = RNAMsmForSsp(pretrained_model)
        pretrained_model._load_pretrained_bert(
            args.pretrained_model+"{}.pth".format(args.model_name))
    elif args.model_name == "RNAFM":
        pretrained_model, alphabet = fm.pretrained.rna_fm_t12()
        pretrained_model = RNAFmForSsp(pretrained_model)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))

    pretrained_model = pretrained_model.to(args.device)

    # load model
    config = {
        'max_helix_length': 30,
        'embed_size': 64,
        'num_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'pool_size': (1, ),
        'dilation': 0,
        'num_lstm_layers': 2,
        'num_lstm_units': 32,
        'num_transformer_layers': 0,
        'num_hidden_units': (32, ),
        'num_paired_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'paired_filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'dropout_rate': 0.5,
        'fc_dropout_rate': 0.5,
        'num_att': 8,
        'pair_join': 'cat',
        'no_split_lr': False,
        'n_out_paired_layers': 3,
        'n_out_unpaired_layers': 0,
        'exclude_diag': True,
        'embed_dim': EMBED_DIMS[args.model_name]
    }
    model = MixedFold(init_param=param_turner2004, **config).to(args.device)
    # load loss function
    _loss_fn = StructuredLoss(loss_pos_paired=0.5, loss_neg_paired=0.005,
                              loss_pos_unpaired=0., loss_neg_unpaired=0., l1_weight=0., l2_weight=0.)
    _loss_fn = _loss_fn.to(args.device)

    # ========== Prepare data
    train_dataset = BPseqDataset(args.dataset_dir, args.dataset_train)
    test_dataset = BPseqDataset(args.dataset_dir, args.dataset_test)

    # ========== Create the data collator
    _collate_fn = SspCollator(max_seq_len=args.max_seq_len,
                              tokenizer=tokenizer, replace_T=args.replace_T, replace_U=args.replace_U)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    _optimizer = Adam(params=model.parameters())

    # ========== Create the metrics
    _metric = SspMetrics(metrics=args.metrics)

    # ========== Training
    ssp_trainer = SspTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        pretrained_model=pretrained_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=_optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            if not ssp_trainer.get_status():
                print("Epoch: {}".format(i_epoch))
                ssp_trainer.train(i_epoch)
                ssp_trainer.eval(i_epoch)
