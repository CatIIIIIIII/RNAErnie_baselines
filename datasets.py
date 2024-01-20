import os.path as osp
from Bio import SeqIO
import torch
from torch.utils.data import Dataset
import numpy as np
from base_classes import BaseCollator
from utils import seq2kmer


class SeqClsDataset(Dataset):
    """.fasta Dataset for sequence classification.
    """

    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True):
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer

        file_name = "train.fa" if train else "test.fa"
        fasta = osp.join(osp.join(fasta_dir, prefix), file_name)
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), x.description.split(" ")[1]) for x in records]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        instance = self.data[idx]
        seq = instance[0]
        label = instance[1]
        return {"seq": seq, "label": label}

    def __len__(self):
        return len(self.data)


class SeqClsCollator(BaseCollator):
    """Data collator for sequence classification.
    """

    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False):

        super(SeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U

    def __call__(self, raw_data_b):
        input_ids_b = []
        label_b = []
        for raw_data in raw_data_b:
            seq = raw_data["seq"]
            seq = seq.upper()
            seq = seq.replace(
                "T", "U") if self.replace_T else seq.replace("U", "T")
            kmer_text = seq2kmer(seq, self.tokenizer)
            # input_text = "[CLS] " + kmer_text + " [SEP]"
            input_text = "[CLS] " + kmer_text
            input_ids = self.tokenizer(input_text)["input_ids"]
            if None in input_ids:
                # replace all None with 0
                input_ids = [0 if x is None else x for x in input_ids]
            input_ids_b.append(input_ids)

            label = raw_data["label"]
            label_b.append(self.label2id[label])

        if self.max_seq_len == 0:
            self.max_seq_len = max([len(x) for x in input_ids_b])

        input_ids_stack = []
        labels_stack = []

        for i_batch in range(len(input_ids_b)):
            input_ids = input_ids_b[i_batch]
            label = label_b[i_batch]

            if len(input_ids) > self.max_seq_len:
                # move [SEP] to end
                # input_ids[self.max_seq_len-1] = input_ids[-1]
                input_ids = input_ids[:self.max_seq_len]

            input_ids += [0] * (self.max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack)),
        }
