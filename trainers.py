import time
from tqdm import tqdm

import torch

from base_classes import BaseTrainer
from collections import defaultdict

import numpy as np
from metrics import compare_bpseq


class SeqClsTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                logits = self.model(input_ids)
                loss = self.loss_fn(logits, labels)

                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()

                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(
                        loss_total / num_total))
                    pbar.update(self.args.logging_steps)
                    num_total, loss_total = 0, 0

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset), disable=self.args.disable_tqdm) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    logits = self.model(input_ids)

                num_total += self.args.batch_size
                outputs_dataset.append(logits)
                labels_dataset.append(labels)

                if num_total >= self.args.logging_steps:
                    pbar.update(self.args.logging_steps)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)
        # save best model
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        # log results to visualdl
        tag_value = defaultdict(float)
        # extract results
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


class RRInterTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=False) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # names = data["names"]
                tokens = data["tokens"].to(self.args.device)
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                preds = self.model(tokens, input_ids)
                loss = self.loss_fn(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()

                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(
                        loss_total / num_total))
                    pbar.update(self.args.logging_steps)
                    num_total, loss_total = 0, 0

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()

        with tqdm(total=len(self.eval_dataset), disable=True) as pbar:
            names_dataset, outputs_dataset, labels_dataset = [], [], []
            for i, data in enumerate(self.eval_dataloader):
                names = data["names"]
                tokens = data["tokens"].to(self.args.device)
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    output = self.model(tokens, input_ids)

                names_dataset += names
                outputs_dataset.append(output)
                labels_dataset.append(labels)
                pbar.update(self.args.batch_size)

            outputs_dataset = torch.concat(outputs_dataset, axis=0)
            labels_dataset = torch.concat(labels_dataset, axis=0)
            # save best model
            metrics_dataset = self.compute_metrics(
                outputs_dataset, labels_dataset)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        # log results to visualdl
        tag_value = defaultdict(float)
        # extract results
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


class SspTrainer(BaseTrainer):
    """Secondary structure prediction trainer.
    """

    def __init__(self, args, tokenizer, model, pretrained_model=None,
                 indicator=None, train_dataset=None, eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):

        super(SspTrainer, self).__init__(
            args=args,
            tokenizer=tokenizer,
            model=model,
            pretrained_model=pretrained_model,
            indicator=indicator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            visual_writer=visual_writer
        )
        self.is_break = False

    def get_status(self):
        return self.is_break

    def train(self, epoch):

        self.model.train()
        time_st = time.time()
        loss_total, num_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, instance in enumerate(self.train_dataloader):

                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )

                input_tensor = torch.tensor(
                    input_ids[0]).unsqueeze(0).to(self.args.device)

                if instance["input_ids"].shape[0] < 600:
                    with torch.no_grad():
                        embeddings = self.pretrained_model(input_tensor)
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(
                        self.model, seqs, refs, embeddings, fname=headers)
                    if loss.item() > 0.:
                        loss.backward()
                        self.optimizer.step()
                    if torch.isnan(loss):
                        self.is_break = True
                else:
                    print(headers)
                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.3e}'.format(
                    loss_total / num_total))
                pbar.update(self.args.batch_size)
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    loss_total, num_total = 0, 0
        elapsed_time = time.time() - time_st
        print('Train Epoch: {}\tTime: {:.3f}s'.format(epoch, elapsed_time))

    def eval(self, epoch):

        self.model.eval()
        n_dataset = len(self.eval_dataloader.dataset)
        with tqdm(total=n_dataset) as pbar:
            # init with default values
            res = defaultdict(list)
            for instance in self.eval_dataloader:
                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )
                seqs = (seqs[0], )
                refs = (refs[0], )
                input_tensor = torch.from_numpy(
                    input_ids[0]).unsqueeze(0).to(self.args.device)
                if instance["input_ids"].shape[0] < 600:

                    with torch.no_grad():
                        embeddings = self.pretrained_model(input_tensor)
                    scs, preds, bps = self.model(seqs, embeddings)
                    for header, seq, ref, sc, pred, bp in zip(headers, seqs, refs, scs, preds, bps):
                        x = compare_bpseq(ref, bp)
                        ret = self.compute_metrics(*x)
                        for k, v in ret.items():
                            res[k].append(v)

                pbar.set_postfix(eval_fls='{:.3e}'.format(
                    np.mean(res[self.name_pbar])))
                pbar.update(self.args.batch_size)
            metrics_dataset = {}
            # log results to screen/bash
            results = {}
            log = 'Test\t' + self.args.task_name + "\t"
            # log results to visualdl
            tag_value = defaultdict(float)
            # extract results
            for k, v in metrics_dataset.items():
                log += k + ": {" + k + ":.4f}\t"
                results[k] = v
                tag = "eval/" + k
                tag_value[tag] = v
            print(log.format(**results))
