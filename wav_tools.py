import wave
import os
import numpy as np
import scipy.signal as dsp_tools
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

class wav_tools:

    _eps = 1e-20 # small value

    @staticmethod
    def wav_read(file_path):
        """ read wav file
        """
        if True:
            x,fs = sf.read(file_path)
        else:
            wav_file = wave.open(file_path,'r')
            sample_num = wav_file.getnframes()
            channel_num = wav_file.getnchannels()
            fs = wav_file.getframerate()

            pcm_data = wav_file.readframes(sample_num)
            x = np.fromstring(data,np.int16)/(2.0**15)
            x = x.reshape([-1,channel_num])
            wav_file.close()

        return [x,fs]


    @staticmethod
    def wav_write(x,fs,file_path):
        """ write wav file,
        """
        if True:
            sf.write(file=file_path,data=x,samplerate=fs)
        else:
            bits_per_sample = 16
            samples = np.asarray(x*(2**(bits_per_sample)),dtype=np.int16)
            sample_num = samples.shape[0]
            if samples.ndim > 1:
                channel_num = samples.shape[1]
            else:
                channel_num = 1

            wav_file = wave.open(file_path,'w')
            wav_file.setparams((channel_num, 2, fs, sample_num, 'NONE', 'not compressed'))
            wav_file.writeframes(samples.tostring())
            wav_file.close()


    # @staticmethod
    # def brir_filter_fft(src,brir):
    #     """ synthesize spatial recording
    #     Args:
    #         src:  sound source
    #         brir: binaural room impulse response,
    #     Returns:
    #         spatial recordings
    #     """
    #
    #     if len(src.shape)>1 and src.shape[1]>1:
    #         raise Exception('input sound source should have only one channel')
    #     signal_len = src.shape[0]
    #     fft_len = 2*signal_len-1
    #     y = np.zeros((signal_len,2),dtype=np.float64)
    #     for channel_i in range(2):
    #         record[:,channel_i] = np.real(np.fft.ifft(np.fft.fft(brir[:,channel_i],fft_len)
    #                                                   *np.fft.fft(src,fft_len)))
    #     return record


    @staticmethod
    def brir_filter(src,brir):
        """ synthesize spatial recording
        Args:
            src:  sound source
            brir: binaural room impulse response,
        Returns:
            spatial recordings
        """
        if len(src.shape)>1 and src.shape[1]>1:
            raise Exception('input sound source should have only one channel')
        signal_len = src.shape[0]
        record = np.zeros((signal_len,2),dtype=np.float64)
        for channel_i in range(2):
            record[:,channel_i] = np.squeeze(dsp_tools.lfilter(brir[:,channel_i],1,src,axis=0))
        return record


    @staticmethod
    def set_SNR(tar,inter,SNR):
        """ scale target signal to a certain SNR
        Args:
            tar: target signal
            inter: reference signal of SNR
            SNR:

        Returns:
            scaled target signal
        """
        power_inter = wav_tools.cal_power(inter)
        power_tar = wav_tools.cal_power(tar)
        gain = np.sqrt(np.float_power(10,float(SNR)/10)/\
                                    (power_tar/power_inter))
        return tar*gain


    @staticmethod
    def cal_power(x):
        """calculate the engergy of given signal
        """
        thd = x.max()/1e5
        x_len = np.count_nonzero(x>thd)
        power = np.sum(np.square(x))/x_len
        return data_power


    @staticmethod
    def frame_data(x,frame_len,shift_len):
        """Frame given data
        Args:
            x: single/multiple channel data
            frame_length: frame length in sample
            shift_len: overlap in sample
        Returns:
            [n_frame,frame_len,n_chann]
        """
        if len(x.shape) == 1: # convert to 2d array for convenience
            x = x[:,np.newaxis]

        n_sample,n_chann = x.shape
        n_frame = np.int(np.floor(np.float32(n_sample-frame_len)/shift_len)+1)

        # window function
        # window = win_func(frame_len)
        # window = np.reshape(window,[-1,1])

        frames = np.zeros((n_frame,frame_len,n_chann))
        for frame_i in range(n_frame):
            start_pos = frame_i*shift_len
            end_pos = frame_i*shift_len+frame_len
            frames[frame_i] = x[start_pos:end_pos]
            # yield frame
        return frames


    @staticmethod
    def _cal_SNR(tar,inter):
        power_tar = wav_tools.cal_power(tar)
        power_inter = wav_tools.cal_power(inter)
        snr = 10*np.log10(power_tar/power_inter)


    @staticmethod
    def cal_SNR(tar,inter,frame_len=None,shift_len=None,is_plot=None):
        """Calculate SNR as a whole, or frame by frame if frame_len is specified
        SNR = 10log10(power_tar/power_inter)
        Args:
            tar: target signal, single channel
            inter: interfere signal, single channel
            frame_len:
            shift_len: if not specified, set to frame_len/2
            if_plot: whether to plot snr of each frames, default to None
        Returns:
            float number or numpy.ndarray
        """
        if frame_len is None:
            snr=wav_tools._cal_SNR(tar,inter)
        else:
            if shift_len is None:
                shift_len = np.int16(frame_len/2)
            if tar.shape[0] != inter.shape[0]:
                raise Exception('tar and inter have different length')

            frames_tar = wav_tools.frame_data(tar,frame_len,shift_len)
            frames_inter = wav_tools.frame_data(inter,frame_len,shift_len)
            n_frame = frames_tar.shape[0]
            snr = np.asarray([wav_tools._cal_SNR(frames_tar[i],frames_inter[i])
                                                    for i in range(n_frame)])
            if is_plot:
                n_sample = tar.shape[0]
                # waveform of tar and inter
                fig = plt.figure()
                axes1 = fig.subplots(1,1)
                t_samples = np.arange(n_sample)/fs
                axes1.plot(time_axis,tar[:n_sample],label='tar')
                axes1.plot(time_axis,inter[:n_sample],label='inter')
                axes1.set_xlabel('time(s)'); axe1.set_ylabel('amp')
                axes1.legend(loc='upper left')
                # SNRs of frames
                axes2 = axes1.twinx()
                axe2.set_ylabel('SNR(dB)')
                # center of frame
                t_frames = np.arange(n_frame)*shift_len+np.int16(frame_len/2)
                axes2.plot(t_frames,snrs,color='red',linewidth=2,label='SNR')
                axes2.legend(loc='upper right')
                plt.tight_layout()
        return snr


    @staticmethod
    def _cal_ccf(x1,x2,max_delay_n=None):
        """calculate cross-crrelation function in frequency domain
        Args:
            x1,x2: single channel signals
            max_delay_n: delay range
        Returns:
            cross-correlation function
        """
        ccf_len = x1.shape[0]+x2.shape[0]-1
        # add hanning window before fft
        wf = np.hanning(n_sample)
        x1 = x1*wf
        x2 = x2*wf

        if max_delay_size is None or max_delay_size<0:
            max_delay_size = n_sample-1

        x1_fft = np.fft.fft(x1,2*ccf_len-1)# equivalent to add zeros
        x2_fft = np.fft.fft(x2,2*ccf_len-1)
        ccf_unshift = np.real(np.fft.ifft(np.multiply(x1_fft,
                                                      np.conjugate(x2_fft))))
        ccf = np.concatenate([ccf_unshift[-max_delay_size:],
                              ccf_unshift[:max_delay_size]],
                             axis=0)
        return ccf


    @staticmethod
    def cal_ccf(tar,inter,max_delay_n=None,frame_len=None,shift_len=None):
        """Calculate cross-correlation function as a whole or frame by frame
        if frame_len is specified
        Args:
            tar: target signal, single channel
            inter: interfere signal, single channel
            frame_len:
            shift_len: if not specified, set to frame_len/2
        Returns:
            numpy.ndarray with shape of [ccf_len] or [n_frame,ccf_len]
        """
        if frame_len is None:
            ccf = wav_tools._cal_ccf(tar,inter,max_delay_n)
        else:
            if shift_len is None:
                shift_len = np.int16(frame_len/2)
            if tar.shape[0] != inter.shape[0]:
                raise Exception('tar and inter have different length')

            frames_tar = wav_tools.frame_data(tar,frame_len,shift_len)
            frames_inter = wav_tools.frame_data(inter,frame_len,shift_len)
            n_frame = frames_tar.shape[0]
            ccf = np.asarray([wav_tools._cal_ccf(frames_tar[i],
                                                 frames_inter[i],
                                                 max_delay_n)
                                    for i in range(n_frame)])
        return ccf


    @staticmethod
    def gen_whitenoise(noise_shape):
        """Generate Gaussian white noise
        Args:
            noise_shape
        Returns:
            white noise
        """
        wn = np.random.normal(0,1,size=ref.shape)
        return wn


    @staticmethod
    def VAD(x,fs,frame_dur=20e-3,shift_dur=None,thd=40,is_plot=False):
        """ Energy based VAD.
            1. Frame data with overlap of 0
            2. Calculte the energy of each frame
            3. Frames with energy below max_energy-thd is regarded as silent frames
        Args:
            x: single channel signal
            frame_dur: frame length in time
            shift_dur: frames shift length in time
            thd: the maximal energy difference between frames, default 40dB
            is_plot: whether to ploting vad result, default False
        Returns:
            vad result of each frame, numpy.ndarray of bool with shape of
            [n_frame]
        """
        frame_len = np.int(frame_dur*fs)
        shift_len = np.int(shift_dur*fs)

        frames = wav_tools.frame_data(x,frame_len,shift_len)
        energy_frames = np.sum(frames**2,axis=1)
        energy_thd = np.max(energy_frames)/(10**(thd/10.0))
        vad_result = np.greater(energy_frames,)

        if is_plot:
            silence_frame_index = np.nonzero(energy_frames<energy_thd)[0]
            plt.figure()
            n_frame = frames.shape[0]
            for frame_i,frame in enumerate(frame_iter):
                start_pos = frame_i*overlap_size
                end_pos = start_pos+overlap_size
                if vad_result[frame_i,1] == True:
                    color = 'red'
                else:
                    color = 'blue'
                plt.plot(np.arange(start_pos,end_pos)/fs,frame,color=color)

            plt.xlabel('time(s)')
            plt.ylabel('amp')

        return vad_result


    # @staticmethod
    # def truncate_speech(wav,fs,frame_len=20e-3,is_plot=False):
    #     """
    #     """
    #     frame_size = int(frame_len*fs)
    #     n_frame = int(wav.shape[0]/frame_size)
    #     wav_croped = wav[:frame_size*n_frame]
    #     frames = np.reshape(wav_croped,(frame_size,n_frame),order='F')
    #     energy_frames = np.sum(frames**2,axis=0)
    #
    #     # find start and end time of speech
    #     thd = np.max(energy_frames)/(10**4)# 40 dB
    #     vad_frame_index = np.nonzero(energy_frames>thd)[0]
    #     start_frame_index = vad_frame_index[0]
    #     end_frame_index = vad_frame_index[-1]
    #
    #     # clip out silent segments at begining and ending
    #     wav_truncated = np.reshape(frames[:,start_frame_index:end_frame_index+1],
    #                                 [-1,1],order='F')
    #
    #     return wav_truncated


    @staticmethod
    def truncate_data(x,truncate_type="both",eps=1e-5):
        """truncate small-value sample in the first dimension
        Args:
            x: data to be truncated
            truncate_type: specify parts to be cliped, options: begin,end,both(default)
            eps: amplitude threshold
        Returns:
            data truncated
        """
        valid_sample_pos = np.nonzero(x>eps)[0]
        start_pos = 0
        end_pos = x.shape[0]
        if truncate_type in ['begin','both']:
            start_pos = np.min(valid_sample_pos)
        if truncate_type in ['end','both']:
            end_pos = np.max(valid_sample_pos)
        return x[start_pos:end_pos+1]


    @staticmethod
    def plot_wav_spec(wav_list,label_list=None,
                      fs=16000,frame_len=20e-3,
                      y_axis_type='mel',figsize=None):
        """plot spectrogram of given len
        Args:
            wav_list: list of numpy 1d arrays
            label_list: labels of each wav
            fs: sample frequency
            frame_len:
            y_axis_type: options 'mel'
            figsize: specify the size of figure
        """

        N_wav = len(wav_list)
        if label_list is not None:
            if len(label_list) is not N_wav:
                raise Exception('wrong number of label')
        else:
            label_list = ['']*N_wav

        if figsize is None:
            plt.figure(figsize=[4*N_wav,6])

        frame_size = int(frame_len*fs)
        for wav_i,[wav,wav_name] in enumerate(zip(wav_list,label_list)):
            plt.subplot(2,N_wav,wav_i+1)
            librosa.display.waveplot(wav)
            plt.title(wav_name)

            plt.subplot(2,N_wav,N_wav+wav_i+1)
            stft = librosa.stft(wav,fft_len=frame_size)
            amp_db_stft = librosa.amplitude_to_db(np.abs(stft))
            librosa.display.specshow(amp_db_stft,sr=fs, x_axis='time', y_axis=y_axis_type)
        plt.tight_layout()


    @staticmethod
    def cal_GCC_PHAT(x,win_f=np.ones,max_delay_n=None):
        """Generalized cross-correlation phase transform
        Args:
            x: 2-channel data
            win_f: window function
        Returns:
            gcc-phat in [n_frame,-1]
        """
        # print(x)
        n_sample,n_chann=x.shape
        if n_chann != 2:
            raise Exception('only 2-channel data is supported')

        fft_len = 2*n_sample-1

        # 计算之前先加窗
        window = win_f(n_sample)
        x1_fft = np.fft.fft(np.multiply(x[:,0],window),n=fft_len)
        x2_fft = np.fft.fft(np.multiply(x[:,1],window),n=fft_len)
        gcc_fft = np.multiply(x1_fft,np.conj(x2_fft))
        gcc_phat_raw = np.real(np.fft.ifft(np.divide(gcc_fft,
                                                     np.abs(gcc_fft)+wav_tools._eps),
                                          fft_len))

        half_gcc_len = np.int16(fft_len/2)
        if max_delay_n is None:
            max_delay_n = half_gcc_len
        gcc_phat = np.concatenate((gcc_phat_raw[-max_delay_n:],
                                   gcc_phat_raw[:max_delay_n+1]))
        return gcc_phat


#
if __name__ == '__main__':

    # truncate_data
    print('truncate_data function test')
    x = np.zeros((2,10))
    x[0,3:7] = 1
    x_truncated = wav_tools.truncate_data(x.T).T
    print('x {}\n after truncation \n{}'.format(x,x_truncated))
