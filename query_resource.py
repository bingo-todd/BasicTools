import subprocess
import numpy as np

def query_resrc(is_print=False):
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
        print('cpu:{:.2f}%  mem:{:.2f}%'.format(cpu_used_percentage,mem_used_percentage))

    return [cpu_used_percentage,mem_used_percentage]

if __name__ == '__main__':
    query_resrc(is_print=True)
