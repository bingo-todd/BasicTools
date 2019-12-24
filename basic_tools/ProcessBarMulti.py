import numpy as np
import time
import os

from multiprocessing import Process, Manager,Lock
from multiprocessing.managers import BaseManager


class ProcessBarMulti_base(object):
    def __init__(self,max_value_all,token_all=None,desc_all=None,
                 is_show_resrc=False,is_show_time=False):
        """show process bar of multiple tasks
        Args:
            max_value_all: maximum iteration of each task
            token_all: token of each task
            desc_all: description of each task
            is_show_resrc: whether to display resource
            is_show_time: whether to display elapsed time of last iteration
        Returns:
        """
        n_task = len(max_value_all)
        if token_all is None:
            token_all = ['{}'.format(i) for i in range(n_task)]

        if desc_all is None:
            desc_all = ['task{}'.format(i) for i in range(n_task)]

        bar_all = {}
        for i in range(n_task):
            bar_all[token_all[i]] = {'max_value':max_value_all[i],
                                     'value':0,
                                     'value_last_update':0,
                                     'desc':desc_all[i],
                                     'time_last':0,
                                     'elapsed_time':0}

        self.n_task = n_task
        self.token_all = token_all
        self.bar_all = bar_all
        self.is_show_resrc = is_show_resrc
        self.is_show_time = is_show_time
        self.is_ploted = False

        self._lock = Lock()


    def get_value(self,token=None):
        if token is None:
            return [self.bar_all[token] for token in self.token_all]
        else:
            return self.bar_all[token]['value']


    def is_finesh(self,token=None):
        """ if token is not specified, return True if all tasks have finished
        """
        if token is None:
            for token in self.token_all:
                if not self.is_finish(token):
                    return False
            return True
        else:
             return self.value >= self.max_value


    def _get_n_col_bar(self):
        n_col_terminal = os.get_terminal_size()[0]
        n_col_bar = int(n_col_terminal*0.66)
        return n_col_bar


    def plot_bar(self):
        # finish_symbol = '>'
        # rest_symbol = '='
        status_str = ''
        bar_status_str_all = []

        n_col_bar = self._get_n_col_bar()
        for token in self.token_all:
            bar = self.bar_all[token]
            p = np.float32(bar['value'])/bar['max_value']

            n_finish = np.int16(p*n_col_bar)
            n_left = n_col_bar - n_finish
            bar_status_str = '[{}] {:<6.2f}% {}'.format(f"{n_finish*'>'}{n_left*'='}",
                                                 p*100,
                                                 bar['desc'])
            bar_status_str_all.append(bar_status_str)

        if self.is_ploted:
            print('\033[F'*(self.n_task),flush=True,end='')
        else:
            self.is_ploted = True

        print('\n'.join(bar_status_str_all))


    def update(self,token,value=None):
        """Update process bar of given token
        Args:
            token: token to specify process bar
            value: assign value to process bar directly
        """
        # with self._lock:

        with self._lock:
            if token not in self.token_all:
                raise Exception(f'{token} not in possible tokens {self.token_all}')

            bar = self.bar_all[token]
            if value == None:
                value = bar['value'] + 1

            if value >= bar['max_value']:
                bar['value'] = bar['max_value']
            else:
                bar['value'] = value

            n_col_terminal = os.get_terminal_size()[0]
            n_col_bar = n_col_terminal - 36
            if bar['value']-bar['value_last_update'] < n_col_bar/bar['max_value']:
                return# no need to update

            bar['value_last_update'] = bar['value']
            self.plot_bar()


    def _auto_update(self):
        while(self.value<self.max_value):
            time.sleep(1)
            self.update(value=0)

#
BaseManager.register('ProcessBarMulti_base',ProcessBarMulti_base)
manager = BaseManager()
manager.start()
ProcessBarMulti = manager.ProcessBarMulti_base


def test_process_bar_multi():
    # import queue
    def count_up(pb_share,token,max_value):
        for value in range(max_value):
            # with lock:
            pb_share.update(token)
            time.sleep(0.5)

    n_task = 3
    process_all = []
    max_value_all = [200,100,50]
    pb_share = ProcessBarMulti(max_value_all[:n_task])
    lock = Lock()
    for task_i in range(n_task):
        process = Process(target=count_up,args=(pb_share,str(task_i),
                                                   max_value_all[task_i]))
        process.start()
        process_all.append(process)

    [process.join() for process in process_all]


if __name__ == '__main__':
    test_process_bar_multi()
