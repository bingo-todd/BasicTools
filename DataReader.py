import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cPickle
import threading

class DataReader(object):
    """create a thread to read data when needed
    example:

    coord = tf.train.Coordinator()# 
    reader = DataReader('Data_set/Develop/',
                        x_len=1771,y_len=161,
                        batch_size=100,
                        coord=coord)    
    x_batch,y_batch = reader.dequeue()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init = tf.global_variables_initializer()
    sess.run(init)

    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    reader.start_thread()

    try:
        for i in xrange(100):
            x_batch_value,y_batch_value = sess.run([x_batch,y_batch])
            print x_batch_value.shape,y_batch_value.shape
    except KeyboardInterrupt:
        print()
    finally:
        coord.request_stop()
        coord.join(threads)
    """
    
    
    def __init__(self,file_dir,x_len,y_len,batch_size,coord,file_reader=None):
        """ 
        """
        
        # get all data file_paths, shuffle(approximate data shuffle) and repet(epoch)
        self.file_dir = file_dir
        # 
        self.coord = coord
        #
        self.x_len = x_len
        self.y_len = y_len
        
        self.batch_size = batch_size
        # data queue
        batch_num_queue = 2
        self.x_queue = tf.FIFOQueue(capacity=batch_size*batch_num_queue,
                               dtypes=tf.float32,
                               shapes=[self.x_len])
        self.y_queue = tf.FIFOQueue(capacity=batch_size*batch_num_queue,
                               dtypes=tf.float32,
                               shapes=[self.y_len])
        
        # enqueue op
        self.x_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,self.x_len])
        self.y_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,self.y_len])

        self.x_enqueue = self.x_queue.enqueue_many([self.x_placeholder])
        self.y_enqueue = self.y_queue.enqueue_many([self.y_placeholder])
        
        # 
        self.threads = []
        
        if file_reader == None:
            self.file_reader = self._read_dat
        else:
            self.file_reader = file_reader
            
        #
    
    def dequeue(self):
        x_batch = self.x_queue.dequeue_many(self.batch_size)
        y_batch = self.y_queue.dequeue_many(self.batch_size)
        return [x_batch,y_batch]
    
        
    def _read_dat(self):
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
                x,y = cPickle.load(data_file)
            yield [filepath,x,y]
                
        
        
    def _reader_main(self,sess):
        """"""
        stop = False
        while not stop:
            # loop until main prosess finish
            iterator = self.read_dat()
            for filepath,x_file,y_file in iterator:
                if self.coord.should_stop(): # train finish, terminate file read
                    stop = True
                    break
                    
                sess.run([self.x_enqueue,self.y_enqueue],
                         feed_dict={self.x_placeholder:x_file,self.y_placeholder:y_file})
                
    def start_thread(self,sess,n_thread=1):
        for _ in np.arange(n_thread):
            thread = threading.Thread(target=self._reader_main,args=(sess,))
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
        return self.threads