import torch

from utils import seq2kmer
from base_classes import BaseCollator


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
            kmer_text = seq2kmer(seq)
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
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class RRDataCollator(BaseCollator):
    def __init__(self, max_seq_lens, tokenizer,
                 replace_T=True, replace_U=False):
        super(RRDataCollator, self).__init__()
        self.max_seq_lens = max_seq_lens
        self.tokenizer = tokenizer
        self.replace_T = replace_T
        self.replace_U = replace_U

    def __call__(self, raw_data_b):
        (max_seq_length_a, max_seq_length_b) = self.max_seq_lens
        max_seq_len = max_seq_length_a + max_seq_length_b

        names = []
        tokens_stack = []
        input_ids_stack = []
        labels_stack = []
        for raw_data in raw_data_b:

            label = raw_data["label"]

            # combine names
            a_name = raw_data["a_name"]
            b_name = raw_data["b_name"]
            name = a_name + "+" + b_name

            a_seq = raw_data["a_seq"].upper()
            a_seq = a_seq.replace(
                "T", "U") if self.replace_T else a_seq.replace("U", "T")
            b_seq = raw_data["b_seq"].upper()
            b_seq = b_seq.replace(
                "T", "U") if self.replace_T else b_seq.replace("U", "T")
            # encoder maps N,A,T,C,G to 0,1,2,3,4
            encoder = dict(zip('NATCG', range(5))) if self.replace_U else dict(
                zip('NAUCG', range(5)))
            tokens_a = [encoder[x] for x in a_seq]
            tokens_b = [encoder[x] for x in b_seq]
            if len(tokens_b) > max_seq_length_b:
                tokens_b = tokens_b[:max_seq_length_b]
            elif len(tokens_b) < max_seq_length_b:
                tokens_b = tokens_b + [0] * (max_seq_length_b - len(tokens_b))
            tokens = tokens_a + tokens_b
            # pad whole tokens
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            tokens += [0] * (max_seq_len - len(tokens))

            # tokenizer
            kmer_text_a = seq2kmer(a_seq)
            input_ids_a = self.tokenizer(kmer_text_a)["input_ids"]
            if None in input_ids_a:
                # replace all None with 0
                input_ids_a = [0 if x is None else x for x in input_ids_a]
            kmer_text_b = seq2kmer(b_seq)
            input_ids_b = self.tokenizer(kmer_text_b)["input_ids"]
            if None in input_ids_b:
                # replace all None with 0
                input_ids_b = [0 if x is None else x for x in input_ids_b]
            if len(input_ids_b) > max_seq_length_b:
                input_ids_b = input_ids_b[:max_seq_length_b]
            elif len(input_ids_b) < max_seq_length_b:
                input_ids_b = input_ids_b + [0] * \
                    (max_seq_length_b - len(input_ids_b))
            input_ids = input_ids_a + input_ids_b
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
            input_ids += [0] * (max_seq_len - len(input_ids))

            names.append(name)
            tokens_stack.append(tokens)
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "names": names,
            "tokens": torch.from_numpy(self.stack_fn(tokens_stack)),
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack)),
        }
