import argparse

from RNABERT.rnabert import BertModel
from RNABERT.utils.bert import get_config
from datasets import SeqClsCollator, SeqClsDataset
from losses import SeqClsLoss
import torch
from torch.optim import AdamW
from metrics import SeqClsMetrics
from trainers import SeqClsTrainer
from utils import str2bool, str2list
from tokenizer import RNATokenizer
from seq_cls import RNABertForSeqCls

# ========== Define constants
MODELS = ["RNABERT"]
MAX_SEQ_LEN = {"RNABERT": 440}

TASKS = ["nRC", "lncRNA_H", "lncRNA_M"]
NUM_CLASSES = {
    "nRC": 13,
    "lncRNA_H": 2,
    "lncRNA_M": 2,
}
LABEL2ID = {
    "nRC": {
        "5S_rRNA": 0,
        "5_8S_rRNA": 1,
        "tRNA": 2,
        "ribozyme": 3,
        "CD-box": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "riboswitch": 7,
        "IRES": 8,
        "HACA-box": 9,
        "scaRNA": 10,
        "leader": 11,
        "miRNA": 12
    },
    "lncRNA_H": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_M": {
        "lnc": 0,
        "pc": 1
    },
}

# ========== Configuration
parser = argparse.ArgumentParser(
    'Implementation of RNA sequence classification.')
# model args
parser.add_argument('--model_name', type=str, default="RNABERT", choices=MODELS)
parser.add_argument('--vocab_path', type=str, default="./vocabs/")
parser.add_argument('--pretrained_model', type=str,
                    default="./checkpoints/bert/")
parser.add_argument('--config_path', type=str,
                    default="./RNABERT/RNA_bert_config.json")

parser.add_argument('--dataset_dir', type=str, default="./data/seq_cls")
parser.add_argument('--dataset', type=str, default="nRC", choices=TASKS)
parser.add_argument('--num_classes', type=int, default=0)
parser.add_argument('--replace_T', type=bool, default=True)
parser.add_argument('--replace_U', type=bool, default=False)

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_seq_len', type=int, default=0)
parser.add_argument('--hidden_size', type=int, default=120)
parser.add_argument('--dataloader_num_workers', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--disable_tqdm', type=str2bool,
                    default=False, help='Disable tqdm display if true.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=50,
                    help='The number of epoch for training.')
parser.add_argument('--metrics', type=str2list,
                    default="F1s,Precision,Recall,Accuracy,Mcc",)

parser.add_argument('--logging_steps', type=int, default=1000,
                    help='Update visualdl logs every logging_steps.')
args = parser.parse_args()

if __name__ == "__main__":
    # ========== post process
    args.num_classes = NUM_CLASSES[args.dataset]
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]

    # ========== args check
    assert args.num_classes > 0, "num_classes must be greater than 0."
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # ========== set device
    torch.set_default_device(args.device)

    # ========== Build tokenizer, model, criterion
    tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))

    if args.model_name == "RNABERT":
        model_config = get_config("./configs/RNABERT.json")
        model = BertModel(model_config)
        model = RNABertForSeqCls(model, args.hidden_size, args.num_classes)
    model._load_pretrained_bert(
        args.pretrained_model+"{}.pth".format(args.model_name))

    _loss_fn = SeqClsLoss()

    # ========== Prepare data
    dataset_train = SeqClsDataset(
        fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer)
    dataset_eval = SeqClsDataset(
        fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer, train=False)

    # ========== Create the data collator
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len, tokenizer=tokenizer,
        label2id=LABEL2ID[args.dataset], replace_T=args.replace_T, replace_U=args.replace_U)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # ========== Create the metrics
    _metric = SeqClsMetrics(metrics=args.metrics)

    # ========== Create the trainer
    seq_cls_trainer = SeqClsTrainer(
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
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
