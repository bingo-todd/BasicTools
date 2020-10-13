import os
import re
import numpy as np
import subprocess
from subprocess import PIPE
from multiprocessing import Queue, Process, Manager, Lock
from BasicTools.ProcessBarMulti import ProcessBarMulti


def get_subprocess_pid(pid):
    p = subprocess.Popen(['pstree', '-pn', f'{pid}'], stdout=PIPE)
    out, err = p.communicate()
    pids = []
    for line in out.split():
        elems = re.split('[()]', line.decode().strip())
        if len(elems) > 2:
            try:
                pid = int(elems[-2])
            except Exception:
                None
        pids.append(pid)
    return pids


def worker(func, lock, tasks_queue, outputs_dict, pb, worker_params,
           pid_father, worker_dir_base):
    cur_pid = os.getpid()
    worker_dir = f'{worker_dir_base}/{cur_pid}'
    if os.path.exists(worker_dir):
        os.system(f'rm -r {worker_dir}')
    os.makedirs(worker_dir)
    while True:
        task = tasks_queue.get()
        if task is None:
            break
        else:
            task_id = task[0]
            try:
                result = func(*task[1:], *worker_params)
            except Exception as e:
                with open(f'{worker_dir_base}/Exception', 'w') as except_file:
                    except_file.write(f'{e}')
                print(f'{e}')
                print(f'log info in {worker_dir_base}')
                print(f'running {func}: kill all subprocess')

                for pid_str in os.listdir(f'{worker_dir_base}'):
                    if pid_str.isdecimal():
                        pid = int(pid_str)
                        if pid != cur_pid:
                            os.system(f'kill {pid} > /dev/null')
                os.system(f'kill {pid_father} > /dev/null')
                break

            with lock:
                outputs_dict[task_id] = result
                if pb is not None:
                    pb.update()
    os.system(f'rm -r {worker_dir}')
    return None


def easy_parallel(func, tasks, n_worker=8, show_process=False,
                  worker_params=None, use_randomstate=False):
    """
    Args:
        func: function to be called in parallel
        tasks: list of list, arguments of func
        n_worker: number of processes
    """

    if len(tasks) < 1:
        return None

    threads = []
    outputs_dict = Manager().dict()
    if show_process:
        pb = ProcessBarMulti([len(tasks)])
    else:
        pb = None

    tasks_queue = Queue()
    [tasks_queue.put([str(task_i), *task])
     for task_i, task in enumerate(tasks)]
    [tasks_queue.put(None) for worker_i in range(n_worker)]

    cur_pid = os.getpid()
    rand_generator = np.random.RandomState(cur_pid)
    while True:
        rand_num = rand_generator.randint(0, 10000)
        worker_dir_base = f'dump/easy_parallel_{rand_num}'
        if not os.path.exists(worker_dir_base):
            break
        os.makedirs(worker_dir_base)

    if worker_params is None:
        worker_params = [[] for worker_i in range(n_worker)]
    else:
        n_worker = len(worker_params)
    if use_randomstate:
        rand_seeds = rand_generator.choice(10000, n_worker)
        [item.append(np.random.RandomState(rand_seed))
         for item, rand_seed in zip(worker_params, rand_seeds)]

    lock = Lock()
    for worker_i in range(n_worker):
        thread = Process(
                target=worker,
                args=(func, lock, tasks_queue, outputs_dict, pb,
                      worker_params[worker_i], cur_pid, worker_dir_base))
        thread.start()
        threads.append(thread)
    [thread.join() for thread in threads]

    with open(f'{worker_dir_base}/result_keys.txt', 'w') as key_record_file:
        key_record_file.write('; '.join(outputs_dict.keys()))
        key_record_file.write(f'n_task {len(tasks)}')

    outputs = [outputs_dict[str(task_i)] for task_i, _ in enumerate(tasks)]

    os.system(f'rm -r {worker_dir_base}')
    return outputs


if __name__ == '__main__':
    import time

    def test_func(i):
        time.sleep(np.random.randint(100, size=1))
        raise Exception(f'i')
        return i+1

    tasks = [[i] for i in range(32)]
    outputs = easy_parallel(test_func, tasks, show_process=True)
    print(outputs)
