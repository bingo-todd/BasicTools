import tensorflow as tf
import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pickle
import threading

class TFData(object):
    """create a thread to read data when needed
    example:
	coord = tf.train.Coordinator()
	train_tfdata = TFData.TFData(self.train_set_dirs,[None,37],[None,37],
	                         self.batch_size,10,coord,file_reader,False)
	train_x_batch,train_y_batch = train_tfdata.dequeue()

	for epoch in range(self.max_epoch):
	    threads = train_tfdata.start_thread(self._sess)# 每个epoch，新开一个读取数据的thread
	    while self._sess.run(train_tfdata.x_queue.size())<self.batch_size:# 等待队列sample个数凑够一个batch
	        time.sleep(0.5)
	    #
	    print('epoch %d'%epoch)
	    while (not train_tfdata.is_epoch_finish) or # 终止条件：1.文件已经遍历一遍；2.队列剩余sample不够一个batch
	    		(self._sess.run(train_tfdata.x_queue.size())>=self.batch_size):
	        train_batch_value = self._sess.run([train_x_batch,train_y_batch])
	        self._sess.run(opt_step,feed_dict={self._x:train_batch_value[0],
	                                           y:train_batch_value[1],
	                                           learning_rate:lr})
	    train_tfdata.empty_queue(self._sess) # 清空队列
    """


    def __init__(self,file_dir,x_shape,y_shape,batch_size,batch_num_queue,coord,data_reader=None,is_repeat=True):
        """
        pipeline for data feed to neural network
        Inputs:
            file_dir: data directory
            x_shape: list, shape of features, first element should be None
            y_shape: list, shape of label, first element should be None
            batch_size: batch_size
            batch_num_queue: queue length in batchs
            coord:
            data_reader: function to load data from file, build in function will be used if None
            is_repeat: bool, if Ture, read all data repeatly, else read all data only once
        """

        self.is_repeat = is_repeat
        # get all data file_paths, shuffle(approximate data shuffle) and repet(epoch)
        self.file_dir = file_dir
        #
        self.coord = coord
        #
        self.x_shape = x_shape
        self.y_shape = y_shape

        self.batch_size = batch_size
        # data queue
        self.x_queue = tf.FIFOQueue(capacity=batch_size*batch_num_queue,
                               dtypes=tf.float32,
                               shapes=self.x_shape[1:])
        self.y_queue = tf.FIFOQueue(capacity=batch_size*batch_num_queue,
                               dtypes=tf.float32,
                               shapes=self.y_shape[1:])

        # enqueue op
        self.x_placeholder = tf.placeholder(dtype=tf.float32,shape=self.x_shape)
        self.y_placeholder = tf.placeholder(dtype=tf.float32,shape=self.y_shape)

        self.x_enqueue = self.x_queue.enqueue_many([self.x_placeholder])
        self.y_enqueue = self.y_queue.enqueue_many([self.y_placeholder])

        #
        self.threads = []

        if data_reader == None:
            self.data_reader = self._data_reader
        else:
            self.data_reader = data_reader

        self.is_epoch_finish = False

        #

    def dequeue(self):
        x_batch = self.x_queue.dequeue_many(self.batch_size)
        y_batch = self.y_queue.dequeue_many(self.batch_size)
        return [x_batch,y_batch]

    def empty_queue(self,sess):
        sess.run(self.x_queue.dequeue_many(self.x_queue.size()))
        sess.run(self.y_queue.dequeue_many(self.y_queue.size()))


    def _data_reader(self,file_dir):
        """read dat file in which both x and y are saved with cPickle"""
        filename_list = os.listdir(self.file_dir)
        # screen data
        filename_filter = lambda filename: len(filename)>3 and filename[-3:]=='dat' and filename[0] !='.'
        filename_list = filter(filename_filter,filename_list)
        # repeate filename_list to enable multiple epoch
        # shuffle filename_list
        np.random.shuffle(filename_list)

        for filename in filename_list:
            filepath =  os.path.join(self.file_dir,filename)
            with file(filepath,'r') as data_file:
                x,y = pickle.load(data_file)
            yield [x,y]



    def _reader_main(self,sess):
        """"""
        stop = False
        while not stop:
            # loop until main prosess finish
            if self.coord.should_stop(): # train finish, terminate file read
                    stop = True
                    break
            try:
                iterator = self.data_reader(self.file_dir)
            except:
                stop = True
                self.coord.request_stop()

            for x_file,y_file in iterator:
                if self.coord.should_stop(): # train finish, terminate file read
                    stop = True
                    break

                sess.run([self.x_enqueue,self.y_enqueue],
                         feed_dict={self.x_placeholder:x_file,self.y_placeholder:y_file})

            if not self.is_repeat:
                self.is_epoch_finish = True
                stop = True

    def start_thread(self,sess,n_thread=1):
        self.is_epoch_finish = False
        for _ in np.arange(n_thread):
            thread = threading.Thread(target=self._reader_main,args=(sess,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads
