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



class process_bar_multi(object):
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

        bar_all = {}
        for i in range(n_task):
            bar_all[token_all[i]] = {'max_value':max_value_all[i],
                                     'value':0,
                                     'desc':desc_all[i],
                                     'time_last':0,
                                     'elapsed_time':0}

        self.n_task = n_task
        self.token_all = token_all
        self.bar_all = bar_all
        self.is_show_resrc = is_show_resrc
        self.is_show_time = is_show_time
        self.is_ploted = False


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


    def plot_bar(self):
        # finish_symbol = '>'
        # rest_symbol = '='
        str_status = ''
        str_status_bar_all = []
        n_col_termial = os.get_terminal_size()[0]
        n_col_bar = n_col_terminal - 36

        for token in self.token_all:
            bar = self.bar_all[token]
            p = np.float32(bar['value'])/bar['max_value']

            bar['elapsed_time'] = time.time() - bar['time_last']

            if self.is_show_time:
                str_elapsed_time = '{}s'.format(bar['elapsed_time'])
            else:
                str_elapsed_time = ''
                '
            n_finish = np.int16(p*n_col_bar)
            n_left = np.int16((1-p)*n_col_bar)
            str_status_bar = '[{}] {} {}'.format(f"{n_finish*'>'}{n_left*''}",
                                                 elapsed_time,
                                                 bar['desc'])
            str_status_bar_all.append(str_status_bar)

        str_pos_control = ''
        if self.is_ploted:
            str_pos_control = '\033[f'*self.n_task
        str_status = '{}{}'.format(str_pos_control,
                                   '\n'.join(str_status_bar_all))
        print(str_status,flush=True,end='')


    def update(self,token,value=None):
        """Update process bar of given token
        Args:
            token: token to specify process bar
            value: assign value to process bar directly
        """
        if token no in self.token_all:
            raise Exception('Unknown token')

        bar = self.bar_all[token]
        if value == None:
            bar['value'] = bar['value'] + 1
        else:
            if value >= bar['max_value']:
                bar['value'] = max_value
            else:
                bar['value'] = value
        self.plot_bar()


    def _auto_update(self):
        while(self.value<self.max_value):
            time.sleep(1)
            self.update(value=0)



if __name__ == '__main__':
    # from process_bar import process_bar

    # norm process_bar
    #print('norm process_bar')
    #p = process_bar(100,is_show_resrc=False) # show current cpu and memory usage
    #for i in range(100):
    #    time.sleep(0.1)
    #    p.update()

    ## process_bar with cpu and mem usage
    #print('process_bar with cpu and mem usage')
    #p = process_bar(100,is_show_resrc=True) # show current cpu and memory usage
    #for i in range(100):
    #    time.sleep(0.1)
    #    p.update()


    for i in range(10):
        print('test 1\n',flush=True,end='')
        print('test 2\n',flush=True,end='')
        print('\033[F\033[F test {}'.format(i))
