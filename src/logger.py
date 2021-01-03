from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union


class Logger(object):
    def __init__(self,
                 log_path="logs",
                 prefix="",
                 log_freq=1,
                 warning_level=3,
                 print_to_terminal=True):
        log_path = self.make_simple_log_path(log_path, prefix)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "output.txt")
        self.log_freq = log_freq
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level


        self.log_counter = 0
        self.value_buffer = defaultdict(list)

    def make_simple_log_path(self, log_path, prefix):
        now = datetime.now()
        suffix = now.strftime("%d_%H:%M")
        pid_str = os.getpid()
        return os.path.join(log_path, "{}-{}-{}".format(prefix, suffix, pid_str))

    def log_str(self, content, level=4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path, 'a+') as f:
            f.write("{}:\t{}".format(time_str, content))

    def log_var(self, name, val, index):
        self.tb_writer.add_scalar(name, val, index)

    def track(self, name, val):
        val = val.detach().sum()
        self.value_buffer[name].append(val)

    def write(self, index):
        self.log_counter += 1
        if self.log_counter % self.log_freq == 0:
            for name in self.value_buffer.keys():
                value = sum(self.value_buffer[name]) / len(self.value_buffer[name])
                self.log_var(name, value, index)

            table = self.get_table()
            self.log_str(table)
            for name in self.value_buffer.keys():
                self.value_buffer[name].clear()

    def get_table(self):
        table = defaultdict(defaultdict)
        for name in self.value_buffer.keys():
            value = sum(self.value_buffer[name]) / len(self.value_buffer[name])
            category, info = name.split('/')
            table[category][info] = value
        table_str = '\n'
        for category in table.keys():
            table_str += '{}: \n'.format(category)
            for info in table[category].keys():
                table_str += '\t{}\t{}\n'.format(info, float(table[category][info]))
        return table_str
