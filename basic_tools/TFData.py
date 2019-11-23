import tensorflow as tf
import numpy as np
import os
import pickle
import threading
import time

class TFData(object):
    """data read pipeline
    """
    def __init__(self,*var_shape,sess,batch_size,n_batch_queue,coord,file_reader=None):
        """
        pipeline for feeding data to neural network
        Inputs:
            file_dir: data directory
            shape: shape of data
            batch_size: batch size
            n_batch_queue: number of batches queue hold
            coord:
            _file_reader: function to load data from file, build in function
                         will be used if None
            is_repeat: bool, if True, read all data repeatly, else read all
                       data only once
        """
        n_var = len(var_shape)
        var_list = []
        queue_list = []
        queue_size_list = []
        enqueue_op_list = []

        for i in range(n_var):
            var = tf.compat.v1.placeholder(dtype=tf.float32,shape=var_shape[i])
        # data queue
            queue = tf.queue.FIFOQueue(capacity=batch_size*n_batch_queue,
                                         dtypes=tf.float32,
                                         shapes=tf.TensorShape(var_shape[i][1:]))
            queue_size = queue.size()
            enqueue_op = queue.enqueue_many([var])

            var_list.append(var)
            queue_list.append(queue)
            queue_size_list.append(queue_size)
            enqueue_op_list.append(enqueue_op)

        if file_reader == None:
            self._file_reader = self._file_reader
        else:
            self._file_reader = file_reader

        var_batch = [queue_list[i].dequeue_many(batch_size) for i in range(n_var)]

        #
        self.coord = coord
        self.batch_size = batch_size
        #
        self.n_var = n_var
        self._var_list = var_list
        self.queue_list = queue_list
        self.queue_size_list = queue_size_list
        self._enqueue_op_list = enqueue_op_list

        self.var_batch = var_batch

        self._sess = sess
        self.threads = []
        self.is_epoch_finish = False # sign, indicate whether all files
                                     # have been read, 1 epoch

    def query_if_finish(self):
        if not self._query_if_ready() and self.is_epoch_finish:
            return True
        else:
            return False


    def _query_if_ready(self):
        queue_size_list = self._sess.run(self.queue_size_list)
        if np.min(queue_size_list) > self.batch_size:
            return True
        else:
            return False


    def _empty_queue(self):
        """"""
        for i in range(self.n_var):
            queue_size = self.queue_list[i].size()
            self._sess.run(self.queue_list[i].dequeue_many(queue_size))


    def _file_reader(self,file_dir):
        """wirte your file_read function and pass to TFData
        """
        raise NotImplementedError()


    def _reader_main(self):
        """"""
        stop = False
        while not stop:
            # loop until main prosess finish
            if self.coord.should_stop(): # train finish, terminate file read
                    stop = True
                    break
            try:
                var_list_generator = self._file_reader(self.file_dir)
            except:
                stop = True
                self.coord.request_stop()

            for var_list in var_list_generator:
                # check size
                n_sample_all = [item.shape[0] for item in var_list]
                if np.min(n_sample_all) != np.max(n_sample_all):
                    raise Exception('var shape miss match')

                if self.coord.should_stop(): # train finish, terminate file read
                    stop = True
                    break
                else:
                    for i in range(self.n_var):
                        self._sess.run(self._enqueue_op_list[i],
                                 feed_dict={self._var_list[i]:var_list[i]})

            if not self.is_repeat:
                self.file_dir=None
                self.is_epoch_finish = True
                stop = True


    def start(self,file_dir,n_thread=1,is_repeat=False):
        self.is_epoch_finish = False
        self.is_repeat = is_repeat
        self.file_dir = file_dir
        for _ in np.arange(n_thread):
            self._empty_queue()
            thread = threading.Thread(target=self._reader_main)
            thread.daemon = True #
            thread.start()

            while not self._query_if_ready():
                time.sleep(0.1)

            self.threads.append(thread)
        return self.threads
