"""Hooks for extended functionality during training
Copyright: SchNetPack, 2019
Retrieved from https://github.com/atomistic-machine-learning/schnetpack/tree/dev/src/schnetpack/train/hooks
"""

import os
import time
import numpy as np
import torch

from nff.train.hooks import Hook


class LoggingHook(Hook):
    """Base class for logging hooks.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        mini_batches=1,
    ):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._counter = 0
        self.metrics = metrics
        self.mini_batches = mini_batches
        
    def on_epoch_begin(self, trainer):
        """Log at the beginning of train epoch.

        Args:
            trainer (Trainer): instance of schnetpack.train.trainer.Trainer class.

        """
        # reset train_loss and counter
        if self.log_train_loss:
            self._train_loss = 0.0
            self._counter = 0
        else:
            self._train_loss = None

    def on_batch_end(self, trainer, train_batch, result, loss):
        if self.log_train_loss:
            n_samples = self._batch_size(result)
            self._train_loss += float(loss.data) * n_samples / self.mini_batches
            self._counter += n_samples

    def _batch_size(self, result):
        if type(result) is dict:
            n_samples = list(result.values())[0].size(0)
        elif type(result) in [list, tuple]:
            n_samples = result[0].size(0)
        else:
            n_samples = result.size(0)
        return n_samples

    def on_validation_begin(self, trainer):
        for metric in self.metrics:
            metric.reset()

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        for metric in self.metrics:
            metric.add_batch(val_batch, val_result)


class CSVHook(LoggingHook):
    """Hook for logging training process to CSV files.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
        mini_batches=1,
    ):
        log_path = os.path.join(log_path, "log.csv")
        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches
        )
        self._offset = 0
        self._restart = False
        self.every_n_epochs = every_n_epochs

    def on_train_begin(self, trainer):

        if os.path.exists(self.log_path):
            remove_file = False
            with open(self.log_path, "r") as f:
                # Ensure there is one entry apart from header
                lines = f.readlines()
                if len(lines) > 1:
                    self._offset = float(lines[-1].split(",")[0]) - time.time()
                    self._restart = True
                else:
                    remove_file = True

            # Empty up to header, remove to avoid adding header twice
            if remove_file:
                os.remove(self.log_path)
        else:
            self._offset = -time.time()
            # Create the log dir if it does not exists, since write cannot
            # create a full path
            log_dir = os.path.dirname(self.log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

        if not self._restart:
            log = ""
            log += "Time"

            if self.log_learning_rate:
                log += ",Learning rate"

            if self.log_train_loss:
                log += ",Train loss"

            if self.log_validation_loss:
                log += ",Validation loss"

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                log += str(metric.name)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a+") as f:
                f.write(log + os.linesep)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            ctime = time.time() + self._offset
            log = str(ctime)

            if self.log_learning_rate:
                log += "," + str(trainer.optimizer.param_groups[0]["lr"])

            if self.log_train_loss:
                log += "," + str(self._train_loss / self._counter)

            if self.log_validation_loss:
                log += "," + str(val_loss)

            if len(self.metrics) > 0:
                log += ","

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, "__iter__"):
                    log += ",".join([str(j) for j in m])
                else:
                    log += str(m)
                if i < len(self.metrics) - 1:
                    log += ","

            with open(self.log_path, "a") as f:
                f.write(log + os.linesep)


class TensorboardHook(LoggingHook):
    """Hook for logging training process to tensorboard.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.
        img_every_n_epochs (int, optional):
        log_histogram (bool, optional):

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        every_n_epochs=1,
        img_every_n_epochs=10,
        log_histogram=False,
        mini_batches=1
    ):
        from tensorboardX import SummaryWriter

        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches
        )
        self.writer = SummaryWriter(self.log_path)
        self.every_n_epochs = every_n_epochs
        self.log_histogram = log_histogram
        self.img_every_n_epochs = img_every_n_epochs

    def on_epoch_end(self, trainer):
        if trainer.epoch % self.every_n_epochs == 0:
            if self.log_train_loss:
                self.writer.add_scalar(
                    "train/loss", self._train_loss / self._counter, trainer.epoch
                )
            if self.log_learning_rate:
                self.writer.add_scalar(
                    "train/learning_rate",
                    trainer.optimizer.param_groups[0]["lr"],
                    trainer.epoch,
                )

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:
            for metric in self.metrics:
                m = metric.aggregate()

                if np.isscalar(m):
                    self.writer.add_scalar(
                        "metrics/%s" % metric.name, float(m), trainer.epoch
                    )
                elif m.ndim == 2:
                    if trainer.epoch % self.img_every_n_epochs == 0:
                        import matplotlib.pyplot as plt

                        # tensorboardX only accepts images as numpy arrays.
                        # we therefore convert plots in numpy array
                        # see https://github.com/lanpa/tensorboard-pytorch/blob/master/examples/matplotlib_demo.py
                        fig = plt.figure()
                        plt.colorbar(plt.pcolor(m))
                        fig.canvas.draw()

                        np_image = np.fromstring(
                            fig.canvas.tostring_rgb(), dtype="uint8"
                        )
                        np_image = np_image.reshape(
                            fig.canvas.get_width_height()[::-1] + (3,)
                        )

                        plt.close(fig)

                        self.writer.add_image(
                            "metrics/%s" % metric.name, np_image, trainer.epoch
                        )

            if self.log_validation_loss:
                self.writer.add_scalar("train/val_loss", float(val_loss), trainer.step)

            if self.log_histogram:
                for name, param in trainer._model.named_parameters():
                    self.writer.add_histogram(
                        name, param.detach().cpu().numpy(), trainer.epoch
                    )

    def on_train_ends(self, trainer):
        self.writer.close()

    def on_train_failed(self, trainer):
        self.writer.close()


class PrintingHook(LoggingHook):
    """Hook for logging training process to the screen.

    Args:
        log_path (str): path to directory in which log files will be stored.
        metrics (list): metrics to log; each metric has to be a subclass of spk.Metric.
        log_train_loss (bool, optional): enable logging of training loss.
        log_validation_loss (bool, optional): enable logging of validation loss.
        log_learning_rate (bool, optional): enable logging of current learning rate.
        every_n_epochs (int, optional): epochs after which logging takes place.
        separator (str, optional): separator for columns to be printed

    """

    def __init__(
        self,
        log_path,
        metrics,
        log_epoch=True,
        log_train_loss=True,
        log_validation_loss=True,
        log_learning_rate=True,
        log_memory=True,
        every_n_epochs=1,
        separator=' ',
        time_strf=r'%Y-%m-%d %H:%M:%S',
        str_format=r'{1:>{0}}',
        mini_batches=1
    ):
        log_path = os.path.join(log_path, "log_human_read.csv")
        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches
        )

        self.every_n_epochs = every_n_epochs
        self.log_epoch = log_epoch

        self._separator = separator
        self.time_strf = time_strf
        self._headers = {
            'time': 'Time',
            'epoch': 'Epoch',
            'lr': 'Learning rate',
            'train_loss': 'Train loss',
            'val_loss': 'Validation loss',
            'memory': 'GPU Memory (MB)'
        }
        self.str_format = str_format
        self.log_memory = log_memory

    def print(self, log):
        print(log)
        with open(self.log_path, "a+") as f:
            f.write(log + os.linesep)

    def on_train_begin(self, trainer):

        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log = self.str_format.format(
            len(time.strftime(self.time_strf)),
            self._headers['time']
        )

        if self.log_epoch:
            log += self._separator
            log += self.str_format.format(
                len(self._headers['epoch']), self._headers['epoch']
            )

        if self.log_learning_rate:
            log += self._separator
            log += self.str_format.format(
                len(self._headers['lr']), self._headers['lr']
            )

        if self.log_train_loss:
            log += self._separator
            log += self.str_format.format(
                len(self._headers['train_loss']), self._headers['train_loss']
            )

        if self.log_validation_loss:
            log += self._separator
            log += self.str_format.format(
                len(self._headers['val_loss']), self._headers['val_loss']
            )

        if len(self.metrics) > 0:
            log += self._separator

        for i, metric in enumerate(self.metrics):
            header = str(metric.name)
            log += self.str_format.format(len(header), header)
            log += self._separator

        if self.log_memory:
            log += self.str_format.format(
                len(self._headers['memory']), self._headers['memory']
            )

        self.print(log)

    def on_validation_end(self, trainer, val_loss):
        if trainer.epoch % self.every_n_epochs == 0:

            log = time.strftime(self.time_strf)

            if self.log_epoch:
                log += self._separator
                log += self.str_format.format(
                    len(self._headers['epoch']),
                    '%d' % trainer.epoch
                )

            if self.log_learning_rate:
                log += self._separator
                log += self.str_format.format(
                    len(self._headers['lr']),
                    '%.3e' % trainer.optimizer.param_groups[0]['lr']
                )

            if self.log_train_loss:
                log += self._separator
                log += self.str_format.format(
                    len(self._headers['train_loss']),
                    '%.4f' % (self._train_loss / self._counter)
                )

            if self.log_validation_loss:
                log += self._separator
                log += self.str_format.format(
                    len(self._headers['val_loss']),
                    '%.4f' % val_loss
                )

            if len(self.metrics) > 0:
                log += self._separator

            for i, metric in enumerate(self.metrics):
                m = metric.aggregate()
                if hasattr(m, '__iter__'):
                    log += self._separator.join([str(j) for j in m])
                else:
                    log += self.str_format.format(
                        len(metric.name),
                        '%.4f' % m
                    )

                log += self._separator

            if self.log_memory:
                memory = torch.cuda.max_memory_allocated(device=None) * 1e-6
                log += self.str_format.format(
                    len(self._headers['memory']),
                    '%d' % memory
                )

            self.print(log)

    def on_train_failed(self, trainer):
        self.print('the training has failed')

