import numpy as np
import subprocess
import threading
import time
import os

# from . import query_resrc
import query_resrc

class process_bar(object):
    def __init__(self,max_value=100,is_show_resrc=False,title='',
                 is_show_time=False):

        self.max_value = max_value
        self.value = 0.
        self.is_show_resrc = is_show_resrc
        self.is_show_time = is_show_time
        self.title = title

    def get_cur_value(self):
        print(self.value)


    def is_finish(self):
        return self.value >= self.max_value


    def update(self,value=None):
        if value == None:
            self.value = self.value + 1
        else:
            if value >= self.max_value:
                self.value = self.max_value
            else:
                self.value = np.mod(value,self.max_value)

        p = np.float32(self.value)/self.max_value
        # finish_symbol = '>'
        # rest_symbol = '='
        n_col_termial = os.get_terminal_size()[0]
        # print(n_col_termial)
        # return
        if self.is_show_time:
            if self.value==1:
                self.time_last = time.time()
                time_str=''
            else:
                time_str = 'time used:{:.2f}s'.format(time.time()-self.time_last)
                self.time_last = time.time()
        else:
            time_str=''

        if self.is_show_resrc:
            n_col_bar = n_col_termial - 30
            if n_col_bar > 50:
                n_col_bar = 50
            bar_str = ''.join(('[',
                               '>'*np.int16(p*n_col_bar),
                               '='*(n_col_bar-np.int16(p*n_col_bar)),
                               ']'))
            print(''.join(('\r{0}{1:>3.0%}',
                           '  Cpu:{2[0]:<5.2f}%',
                           '  Mem:{2[1]:<5.2f}%')).format(bar_str,p,
                                                        query_resrc.query_resrc()),
                 flush=True,end='')
        else:
            n_col_bar = n_col_termial - 6
            if n_col_bar > 50:
                n_col_bar = 50
            bar_str = ''.join(('[',
                               '>'*np.int16(p*n_col_bar),
                               '='*(n_col_bar-np.int16(p*n_col_bar)),
                               ']'))
            print('\r{0}{1:>3.2%} {2} {3}'.format(bar_str,p,self.title,time_str),
                  flush=True,end='')

        # auto update resource usage, in case too long interval between updates
        # if self.is_show_resrc and self.value ==1:
        #     thread = threading.Thread(target=self._auto_update)
        #     thread.daemon = True #
        #     thread.start()

        if self.value == self.max_value:
            print('\n')


    def _auto_update(self):
        while(self.value<self.max_value):
            time.sleep(1)
            self.update(value=0)

if __name__ == '__main__':
    # from process_bar import process_bar

    # norm process_bar
    print('norm process_bar')
    p = process_bar(100,is_show_resrc=False) # show current cpu and memory usage
    for i in range(100):
        time.sleep(0.1)
        p.update()

    # process_bar with cpu and mem usage
    print('process_bar with cpu and mem usage')
    p = process_bar(100,is_show_resrc=True) # show current cpu and memory usage
    for i in range(100):
        time.sleep(0.1)
        p.update()
