# Little python toolbox
A collection of little python scripts

<!-- TOC -->

- [Little python toolbox](#little-python-toolbox)
  - [TFData](#tfdata): data pipe for tensorflow
  - [process_bar](#process_bar)
  - [query_resrc](#query_resrc): get cpu and memory usage
  - [reverb-time](#reverb-time): calculte RT60 from room impulse response(RIR)
  - [show_struct](#show_struct): print functions in a given scripts
  - [wav_tools](#wav_tools): functions related to signal process

<!-- /TOC -->

## TFData

  Data pipe for tensorflow

  E.g.

  ```python
  with tf.Session() as sess:
    coord = tf.train.Coordinator()
    train_tfdata = TFData.TFData(train_set_dirs,[None,x_len],[None,y_len],
                                batch_size,N_batch_in_queue,coord,file_reader_func,is_repeat)
    train_x_batch,train_y_batch = train_tfdata.dequeue()
    for epoch in range(max_epoch):
        # open data-reading thread in each epoch
        threads = train_tfdata.start_thread(sess)
        # until at least one batch data has been ready
        while sess.run(train_tfdata.x_queue.size())<batch_size:
            time.sleep(0.5)
        #
        print('epoch %d'%epoch)
        # one epoch finished when:
        #   1. all files has be read;
        #   2. Number of samples is less than one batch
        while (not train_tfdata.is_epoch_finish) or
                  (sess.run(train_tfdata.x_queue.size())>=batch_size):
            train_batch_value = sess.run([train_x_batch,train_y_batch])
            sess.run(opt_step,feed_dict={x:train_batch_value[0],
                                         y:train_batch_value[1],
                                         learning_rate:lr})
        # clear queue
        train_tfdata.empty_queue(self._sess)
  ```

## process_bar

  Process bar, additonally can show cpu and memory percentage

  E.g.

  ```python
  from process_bar import process_bar
  p = process_bar(100,is_show_resrc) # show current cpu and memory usage
  for i in range(100):
    p.update()
  ```
  if `is_show_resrc=False` <pre>|============================                      | process 56%</pre>
  if `is_show_resrc=True`
   <pre>|=======================                           | process 47% 	 Cpu:1.60% Mem:26.78%</pre>

## query_resrc

  Get cpu and memory usage(%)

  E.g.
  ```shell
  python query_resource.py
  ```

  `cpu:1.70%  mem:30.09%`

## reverb-time

  Calculte RT60 from room impulse response(RIR)

## show_struct

  Print functions in a given scripts

  E.g.

  ```shell
  python show_struct py_file_path options
  # options
  # no_doc: not doc
  # tight: not blank line
  ```
  result show in wav_tools

## wav_tools

  Functions related to signal process
  Structure of wav_tools derived by `show_struct`
  ```
  |wav_tools
  |
  |-functions
  |
  |--BRIR_filter(src, BRIR)
  |
  |--BRIR_filter_fft(src, BRIR)
  |
  |--VAD(wav, fs, frame_len=0.02, overlap=0.02, thd=40, is_plot=False)
  |
  |--cal_SNR(tar, ref)
  |
  |--cal_SNR_frame(tar, inter, fs, frame_len=0.02, overlap=0.01, is_plot=False)
  |
  |--cal_ccf_fft(x1, x2, max_delay_size=None)
  |
  |--cal_ccf_frame(wav, fs, frame_len=0.02, overlap=0.01, max_delay=None)
  |
  |--cal_power(data)
  |
  |--cal_snr(tar, interfer, axis=-1)
  |
  |--frame_x(x, frame_size, overlap_size, win_func=numpy.ones, window=None)
  |
  |--gen_noise(ref, SNR)
  |
  |--plot_wav_spec(wav_list, label_list=None, fs=16000, frame_len=0.02, y_axis_type=mel, figsize=None)
  |
  |--set_SNR(tar, ref, SNR)
  |
  |--truncate_speech(wav, fs, frame_len=0.02, is_plot=False)
  |
  |--wav_read(file_path)
  |
  |--wav_write(signal, fs, file_path)
  |
  ```
