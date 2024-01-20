import time
from tqdm import tqdm

import torch

from base_classes import BaseTrainer
from collections import defaultdict


class SeqClsTrainer(BaseTrainer):
    """Trainer for sequence classification.
    """

    def train(self, epoch):
        """train function

        Returns:
            None
        """
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"]
                labels = data["labels"]

                logits = self.model(input_ids)
                loss = self.loss_fn(logits, labels)

                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.4f}'.format(
                    loss_total / num_total))
                pbar.update(self.args.batch_size)
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    num_total, loss_total = 0, 0
                # log to visualdl
                if (i + 1) % self.args.logging_steps == 0:
                    # log to directory
                    tag_value = {"train/loss": loss.item()}
                    self.visual_writer.update_scalars(
                        tag_value=tag_value, step=self.args.logging_steps)

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        """eval function

        Returns:
            None
        """
        self.model.eval()
        time_st = time.time()
        with tqdm(total=len(self.eval_dataset), disable=self.args.disable_tqdm) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"]
                labels = data["labels"]

                with torch.no_grad():
                    logits = self.model(input_ids)

                metrics = self.compute_metrics(logits, labels)

                pbar.set_postfix(accuracy='{:.4f}'.format(
                    metrics[self.name_pbar]))
                pbar.update(self.args.batch_size)

                outputs_dataset.append(logits)
                labels_dataset.append(labels)

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)
        # save best model
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
        if self.args.save_max and self.args.train:
            self.save_model(metrics_dataset, epoch)

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
        if self.args.train:
            self.visual_writer.update_scalars(tag_value=tag_value, step=1)

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


class RRInterTrainer(BaseTrainer):
    """rna rna interaction trainer
    """

    def train(self, epoch):
        """train function

        Args:
            epoch (int): current epoch
        """
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=False) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # names = data["names"]
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["labels"]

                preds, _ = self.model(tokens, input_ids)
                loss = self.loss_fn(preds, labels)

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.4f}'.format(
                    loss_total / num_total))
                pbar.update(self.args.batch_size)
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    num_total, loss_total = 0, 0
                # log to visualdl
                if (i + 1) % self.args.logging_steps == 0:
                    # log to directory
                    tag_value = {"train/loss": loss.item()}
                    self.visual_writer.update_scalars(
                        tag_value=tag_value, step=self.args.logging_steps)

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        """eval function

        Args:
            epoch (int): current epoch
        """
        self.model.eval()
        time_st = time.time()

        with tqdm(total=len(self.eval_dataset), disable=True) as pbar:
            names_dataset, outputs_dataset, labels_dataset, attn_dataset = [], [], [], []
            for i, data in enumerate(self.eval_dataloader):
                names = data["names"]
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["labels"]

                with torch.no_grad():
                    output, attn = self.model(tokens, input_ids)

                names_dataset += names
                outputs_dataset.append(output)
                labels_dataset.append(labels)
                attn_dataset.append(attn)
                pbar.update(self.args.batch_size)

            outputs_dataset = torch.concat(outputs_dataset, axis=0)
            labels_dataset = torch.concat(labels_dataset, axis=0)
            # save best model
            metrics_dataset = self.compute_metrics(
                outputs_dataset, labels_dataset)
            if self.args.save_max and self.args.train:
                self.save_model(metrics_dataset, epoch)

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
        if self.args.train:
            self.visual_writer.update_scalars(tag_value=tag_value, step=1)

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))
