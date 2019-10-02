import numpy as np
import subprocess

class process_bar(object):
    def __init__(self,max_value=100,is_show_resrc=False):

        self.max_value = max_value
        self.value = 0.
        self.is_show_resrc = is_show_resrc

    def get_cur_value(self):
        print(self.value)

    def update(self,value=None):
        if value == None:
            self.value = self.value + 1
        else:
            self.value = np.mod(value,self.max_value)

        p = np.float32(self.value)/self.max_value
        # finish_symbol = '>'
        # rest_symbol = '='
        if self.is_show_resrc:
            print('\r{0:=<50} process{1:>4.0%} \t Cpu:{2[0]:<.2f}% Mem:{2[1]:<.2f}%'.format('>'*np.int(p*50),p,
                                                           self.query_resrc()),
                 flush=True,end='')
        else:
            print('\r{=:<50} process{:>4.0%}'.format('>'*np.int(p*50),p),
                                                      flush=True,end='')

        if self.value == self.max_value:
            print('\n')


    def query_resrc(self,is_print=False):
        BASE = 1024
        cmd_line = ['free','-m']
        info = subprocess.run(cmd_line,check=True,stdout=subprocess.PIPE).stdout;
        mem_info = info.decode().split('\n')[1]
        mem_info_elems = mem_info.split()
        mem_size_total = np.int(mem_info_elems[1])
        mem_size_used = np.int(mem_info_elems[2])
        mem_used_percentage = mem_size_used/mem_size_total*100

        cmd_line = ['top',"-bn1"]
        info = subprocess.run(cmd_line,check=True,stdout=subprocess.PIPE).stdout;
        info = [line for line in info.decode().split('\n') if 'Cpu(s)' in line][0]
        cpu_used_percentage = np.float(info.split()[1])

        if is_print:
            print('Occupation: cpu {:.2f}% mem {:.2f}%'.format(cpu_used_percentage,
                                                                mem_used_percentage))

        return [cpu_used_percentage,mem_used_percentage]

if __name__ == '__main__':
    from process_bar import process_bar
    # p = process_bar(100)
    p = process_bar(100,is_show_resrc=True) # show current cpu and memory usage
    for i in range(100):
      p.update()
