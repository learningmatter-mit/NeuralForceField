"""Hooks for extended functionality during training
Copyright: SchNetPack, 2019
Retrieved from https://github.com/atomistic-machine-learning/schnetpack/tree/dev/src/schnetpack/train/hooks
"""

import os
import time
import numpy as np
import torch
import json

from nff.train.hooks import Hook
from nff.train.metrics import RootMeanSquaredError


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
        global_rank=0,
        world_size=1
    ):
        self.log_train_loss = log_train_loss
        self.log_validation_loss = log_validation_loss
        self.log_learning_rate = log_learning_rate
        self.log_path = log_path

        self._train_loss = 0
        self._counter = 0
        self.metrics = metrics
        self.mini_batches = mini_batches

        self.global_rank = global_rank
        self.world_size = world_size
        self.par_folders = self.get_par_folders()
        self.parallel = world_size > 1

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
            self._train_loss += float(loss.data) * \
                n_samples / self.mini_batches
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

    def get_base_folder(self):

        sep = os.path.sep
        base_folder = os.path.join(*self.log_path.split(sep)[:-1])
        if base_folder.endswith(str(self.global_rank)):
            base_folder = os.path.join(*base_folder.split(sep)[:-1])
        if self.log_path.startswith(sep):
            base_folder = sep + base_folder
        return base_folder

    def get_par_folders(self):

        base_folder = self.get_base_folder()
        par_folders = [os.path.join(base_folder, str(i))
                   for i in range(self.world_size)]
        return par_folders
 
    def save_metrics(self, epoch):

        par_folder = self.par_folders[self.global_rank]
        json_file = os.path.join(par_folder, "epoch_{}.json".format(epoch))

        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                dic = json.load(f)
        else:
            dic = {}
        for metric in self.metrics:
            m = metric.aggregate()
            dic[metric.name] = m
        with open(json_file, "w") as f:
            json.dump(dic, indent=4, sort_keys=True)

    def avg_parallel_metrics(self, epoch):

        self.save_metrics(epoch)
        metric_dic = {}

        for metric in self.metrics:
            par_dic = {folder: None for folder in self.par_folders}

            while None in par_dic.values():
                for folder in self.par_folders:
                    path = os.path.join(folder, "epoch_{}.json".format(epoch))
                    try:
                        with open(path, "r") as f:
                            path_dic = json.load(f)
                        par_dic[folder] = path_dic[metric.name]
                    except (json.JSONDecodeError, FileNotFoundError, KeyError):
                        continue

            if isinstance(metric, RootMeanSquaredError):
                metric_val = np.mean(np.array(list(par_dic.values)) ** 2) ** 0.5
            else:
                metric_val = np.mean(list(par_dic.values()))
            metric_dic[metric.name] = metric_val

        return metric_dic

    def aggregate(self, metric, trainer):
        if self.parallel:
            m = self.avg_parallel_metrics(epoch=trainer.epoch)
        else:
            m = metric.aggregate()
        return m


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
        global_rank=0,
        world_size=1
    ):
        log_path = os.path.join(log_path, "log.csv")
        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches,
            global_rank, world_size
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
                m = self.aggregate(metric, trainer)
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
        mini_batches=1,
        global_rank=0,
        world_size=1
    ):
        from tensorboardX import SummaryWriter

        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches,
            global_rank, world_size
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
                m = self.aggregate(metric, trainer)

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
                self.writer.add_scalar(
                    "train/val_loss", float(val_loss), trainer.step)

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
        mini_batches=1,
        global_rank=0,
        world_size=1
    ):

        log_path = os.path.join(log_path, "log_human_read.csv")
        super().__init__(
            log_path, metrics, log_train_loss, log_validation_loss, log_learning_rate, mini_batches,
            global_rank, world_size
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
                m = self.aggregate(metric, trainer)
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
