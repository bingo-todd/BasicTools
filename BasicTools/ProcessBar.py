import numpy as np
import time
import os


class ProcessBar(object):
    def __init__(self, max_value=100, is_show_resrc=False, title='',
                 is_show_time=False):

        self.max_value = max_value
        self.value = 0.
        self.is_show_resrc = is_show_resrc
        self.is_show_time = is_show_time
        self.title = title

        self._is_ploted = False

    def get_cur_value(self):
        print(self.value)

    def is_finish(self):
        return self.value >= self.max_value

    def _get_n_col_bar(self):
        n_col_terminal = os.get_terminal_size()[0]
        n_col_bar = int(n_col_terminal*0.66)
        return n_col_bar

    def update(self, value=None):
        if value is None:
            self.value = self.value + 1
        else:
            if value >= self.max_value:
                self.value = self.max_value
            else:
                self.value = value

        p = np.float32(self.value)/self.max_value
        n_col_bar = self._get_n_col_bar()
        n_finish = np.int16(p*n_col_bar)
        n_left = n_col_bar - n_finish
        bar_str = f"{n_finish*'>'}{n_left*'='}"
        bar_status_str = '[{}] {:>3.0f}% {}'.format(bar_str,
                                                    p*100,
                                                    self.title)
        if self._is_ploted:
            print('\033[F', flush=True, end='')
        else:
            self._is_ploted = True
        print(bar_status_str)


if __name__ == '__main__':

    pb = ProcessBar(100)
    for i in range(100):
        time.sleep(0.1)
        pb.update()
