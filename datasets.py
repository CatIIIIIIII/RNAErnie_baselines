import os.path as osp

import numpy as np
import pandas as pd
from Bio import SeqIO

from torch.utils.data import Dataset


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


class GenerateRRInterTrainTest:
    """generate train and test dataset for rna rna interaction prediction.
    """

    def __init__(self,
                 rr_dir,
                 dataset,
                 split=0.8,
                 seed=0):
        """init function

        Args:
            rr_dir (str): data root dir
            dataset (str): dataset name
            split (float, optional): split ratio. Defaults to 0.8.
            seed (int, optional): random seed. Defaults to 0.
        """

        csv_path = osp.join(rr_dir, dataset) + ".csv"
        self.data = pd.read_csv(csv_path, sep=",").values.tolist()

        self.split_index = int(len(self.data) * split)

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def get(self):
        """get train and test dataset

        Returns:
            tuple: RRInterDataset, RRInterDataset
        """
        return RRInterDataset(self.data[:self.split_index]), RRInterDataset(self.data[self.split_index:])


class RRInterDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):

        instance = self.data[idx]
        return {
            "a_name": instance[0],
            "a_seq": instance[1],
            "b_name": instance[2],
            "b_seq": instance[3],
            "label": instance[4],
        }

    def __len__(self):
        return len(self.data)
