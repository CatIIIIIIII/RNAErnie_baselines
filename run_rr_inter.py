import argparse

from RNABERT.rnabert import BertModel
from RNABERT.utils.bert import get_config
from torch.optim import AdamW
from RNAMSM.model import MSATransformer
from metrics import RRInterMetrics
from trainers import RRInterTrainer

from utils import str2bool, str2list
from losses import RRInterLoss
import torch
from datasets import GenerateRRInterTrainTest
from rr_inter import RNABertForRRInter, RNAMsmForRRInter
from tokenizer import RNATokenizer
from collators import RRDataCollator

# ========== Define constants
MODELS = ["RNABERT", "RNAMSM"]

# ========== Configuration
parser = argparse.ArgumentParser(
    'Implementation of RNA-RNA Interaction prediction.')
# model args
parser.add_argument('--model_name', type=str, default="RNAMSM", choices=MODELS)
parser.add_argument('--vocab_path', type=str, default="./vocabs/")
parser.add_argument('--pretrained_model', type=str,
                    default="./checkpoints/")
parser.add_argument('--config_path', type=str,
                    default="./configs/")

parser.add_argument('--dataset', type=str, default="MirTarRAW",)
parser.add_argument('--dataset_dir', type=str, default="./data/rr")
parser.add_argument('--replace_T', type=bool, default=True)
parser.add_argument('--replace_U', type=bool, default=False)

parser.add_argument('--dataloader_num_workers', type=int, default=0)
# training args
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_seq_lens', type=list, default=[26, 40])
parser.add_argument('--hidden_size', type=int, default=120)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--disable_tqdm', type=str2bool,
                    default=False, help='Disable tqdm display if true.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=50,
                    help='The number of epoch for training.')
parser.add_argument('--metrics', type=str2list,
                    default="F1s,Precision,Recall,Accuracy",)

# logging args
parser.add_argument('--logging_steps', type=int, default=1000,
                    help='Update visualdl logs every logging_steps.')
args = parser.parse_args()

if __name__ == "__main__":
    # ========== post process

    # ========== args check
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # ========== set device
    torch.set_default_device(args.device)

    # ========== Build tokenizer, model, criterion
    tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))

    model_config = get_config(
        args.config_path + "{}.json".format(args.model_name))
    if args.model_name == "RNABERT":
        model = BertModel(model_config)
        model = RNABertForRRInter(model)
    elif args.model_name == "RNAMSM":
        model = MSATransformer(model_config)
        model = RNAMsmForRRInter(model)
    model._load_pretrained_msa(
        args.pretrained_model+"{}.pth".format(args.model_name))

    _loss_fn = RRInterLoss()

    # ========== Prepare data
    # load datasets
    # train & test datasets
    datasets_generator = GenerateRRInterTrainTest(rr_dir=args.dataset_dir,
                                                  dataset=args.dataset,
                                                  split=0.8,)
    dataset_train, dataset_eval = datasets_generator.get()

    # ========== Create the data collator
    _collate_fn = RRDataCollator(
        max_seq_lens=args.max_seq_lens,
        tokenizer=tokenizer,
        replace_T=args.replace_T,
        replace_U=args.replace_U)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    # optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # ========== Create the metrics
    _metric = RRInterMetrics(metrics=args.metrics)

    # ========== Training
    # train model
    rr_inter_trainer = RRInterTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            rr_inter_trainer.eval(i_epoch)
            rr_inter_trainer.train(i_epoch)
