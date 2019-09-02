import wave
import os
import numpy as np
import scipy.signal as dsp_tools
import matplotlib.pyplot as plt
import librosa
import librosa.display

class wav_tools:

    @staticmethod
    def wav_read(file_path):
        """ read wav file, better use soundfile instead
        """
        wav_file = wave.open(file_path,'r')
        sample_num = wav_file.getnframes()
        channel_num = wav_file.getnchannels()
        fs = wav_file.getframerate()

        data = wav_file.readframes(sample_num)
    #   print wav_file.getparams()
        samples = np.fromstring(data,np.int16)
        signal = samples/(2.0**15)
        signal = signal.reshape([-1,channel_num])
        wav_file.close()
        return [signal,fs]

    @staticmethod
    def wav_write(signal,fs,file_path):
        """ write wav file, better use soundfile instead
        """
        bits_per_sample = 16
        samples = np.asarray(signal*(2**(bits_per_sample)),dtype=np.int16)
        sample_num = samples.shape[0]
        if samples.ndim > 1:
            channel_num = samples.shape[1]
        else:
            channel_num = 1

        wav_file = wave.open(file_path,'w')
        wav_file.setparams((channel_num, 2, fs, sample_num, 'NONE', 'not compressed'))
        wav_file.writeframes(samples.tostring())
        wav_file.close()


    @staticmethod
    def BRIR_filter_fft(src,BRIR):
        """ synthesize spatial recording
        Args:
            src: 1 channel sound source
            BRIR: binaural room impulse response, two channel
        Returns:
            two channel spatial recordings
        """

        if len(src.shape)>1 and src.shape[1]>1:
            raise Exception('input sound source should have only one channel')
        signal_len = src.shape[0]
        N_fft = 2*signal_len-1
        record = np.zeros((signal_len,2),dtype=np.float64)
        for channel_i in range(2):
            record[:,channel_i] = np.real(np.fft.ifft(np.fft.fft(BRIR[:,channel_i],N_fft)
                                                      *np.fft.fft(src,N_fft)))
        return record


    @staticmethod
    def BRIR_filter(src,BRIR):
        """ synthesize spatial recording
        Args:
            src: 1 channel sound source
            BRIR: binaural room impulse response, two channel
        Returns:
            two channel spatial recordings
        """

        if len(src.shape)>1 and src.shape[1]>1:
            raise Exception('input sound source should have only one channel')
        signal_len = src.shape[0]
        record = np.zeros((signal_len,2),dtype=np.float64)

        for channel_i in range(2):
            record[:,channel_i] = np.squeeze(dsp_tools.lfilter(BRIR[:,channel_i],1,src,axis=0))
        return record


    @staticmethod
    def set_SNR(tar,ref,SNR):
        """ scale target signal to a certain SNR
        Args:
            tar: target signal
            ref: reference signal of SNR
            SNR:

        Returns:
            scaled target signal
        """

        ref_len = np.count_nonzero(ref>ref.max()/1000000)
        power_ref = np.sum(np.square(ref))/ref_len

        tar_len = np.count_nonzero(tar>tar.max()/1000000)
        power_tar = np.sum(np.square(tar))/tar_len

        power_gain = np.float_power(10,float(SNR)/10)/(power_tar/power_ref)

        return tar*np.sqrt(power_gain)


    @staticmethod
    def cal_power(data):
        """calculate the engergy of given signal"""
        # data_len = np.count_nonzero(data>data.max()/1000000)
        data_power = np.sum(np.square(data))/data_len
        return data_power


    @staticmethod
    def cal_SNR(tar,ref):
        """calculate the SNR of tar and ref
            SNR = 10log(sum(tar**2)/sum(ref**2))
        """
        tar_len = np.count_nonzero(tar>tar.max()/1000000)
        tar_power = np.sum(np.square(tar))/tar_len

        ref_len = np.count_nonzero(ref>ref.max()/1000000)
        ref_power = np.sum(np.square(ref))/ref_len

        return 10*np.log10(tar_power/ref_power)


    @staticmethod
    def frame_x(x_list,frame_size,overlap_size):
        N_sample = np.min([x.shape[0] for x in x_list])
        N_frame = np.int(np.floor(np.float32(N_sample-frame_size)/overlap_size)+1)
        for frame_i in range(N_frame):
            start_pos = frame_i*overlap_size
            end_pos = frame_i*overlap_size+frame_size
            yield [x[start_pos:end_pos] for x in x_list]


    @staticmethod
    def cal_instant_SNR(tar,inter,fs,frame_len=20e-3,overlap=10e-3,is_plot=False):
        """
        Calculate SNR frame by frame
        Args:
            tar: target signal, single channel
            inter: interference signal, single channel
            frame_size: frame duration (s)
            overlap: overlap (s)
        Returns:
            SNR of each frame. shape:[N_frame]
        """
        tar = np.squeeze(tar)
        inter = np.squeeze(inter)
        if len(tar.shape)>1 or len(inter.shape)>1:
            raise Exception('tar or inter should only have one dimension')

        frame_size = np.int(fs*frame_len)
        overlap_size = np.int(fs*overlap)
        frame_iter = wav_tools.frame_x([tar,inter],frame_size,overlap_size)
        snrs = np.asarray([wav_tools.cal_SNR(tar_frame,inter_frame)
                            for tar_frame,inter_frame in frame_iter])

        if is_plot:
            N_sample = np.min([tar.shape[0],inter.shape[0]])
            N_frame = snrs.shape[0]

            fig = plt.figure(dpi=120)
            axe1 = fig.subplots(1,1)
            time_axis = np.arange(N_sample)/fs
            axe1.plot(time_axis,tar[:N_sample],label='target')
            axe1.plot(time_axis,inter[:N_sample],label='interference')
            axe1.set_xlabel('time(s)'); axe1.set_ylabel('amp')
            axe1.legend(loc='upper left')

            axe2 = axe1.twinx()
            axe2.set_xlabel('time(s)'); axe2.set_ylabel('SNR(dB)')
            axe2.plot((np.arange(N_frame)+1)*overlap,snrs,color='red',linewidth=2,label='SNR')# center of frame
            axe2.legend(loc='upper right')

            plt.tight_layout()

        return snrs


    @staticmethod
    def cal_ccf_fft(x1,x2,max_delay_size=None):
        """calculate cross-crrelation function in frequency domain, which is more
        efficient than calculating directly
        Args:
            x1,x2: single channel signals
        Returns:
            cross-correlation function
        """

        N_sample = x1.shape[0]+x2.shape[0]-1
        # add hanning window before fft
        # wf = np.hanning(N_sample)
        # x1 = x1*wf
        # x2 = x2*wf

        if max_delay_size is None or max_delay_size<0:
            max_delay_size = N_sample-1

        X1 = np.fft.fft(x1,2*N_sample-1)# equivalent to add zeros
        X2 = np.fft.fft(x2,2*N_sample-1)
        ccf_unshift = np.real(np.fft.ifft(np.multiply(X1,np.conjugate(X2))))
        ccf = np.concatenate([ccf_unshift[-max_delay_size:],ccf_unshift[:max_delay_size]],axis=0)

        return ccf


    @staticmethod
    def cal_instant_ccf(wav,fs,frame_len=20e-3,overlap=10e-3,max_delay=None):
        """

        """
        frame_size = np.int(fs*frame_len)
        overlap_size = np.int(fs*overlap)
        if max_delay is not None:
            max_delay_size = np.int(fs*max_delay)
        else:
            max_delay_size = None
            
        frame_iter = wav_tools.frame_x([wav],frame_size,overlap_size)
        ccfs = np.asarray([wav_tools.cal_ccf_fft(frame,frame,max_delay_size)/(np.sum(frame**2))
                                for [frame] in frame_iter])
        return ccfs


    @staticmethod
    def gen_noise(ref,SNR):
        """
        Generate white noise with desired SNR

        Args:
            SNR: target to reference ration
            ref: reference signal of SNR
        Returns:
            white noise with desired SNR
        """
        wn_raw = np.random.normal(0,1,size=ref.shape[0])
        wn = set_SNR(wn_raw,ref,SNR)
        return wn

    @staticmethod
    def VAD(wav,fs,frame_len=20e-3,overlap=20e-3,thd=40, is_plot=False):
        """ Energy based VAD.
            1. Frame data with overlap of 0
            2. Calculte the energy of each frame
            3. Frames with energy below max_energy-thd is regarded as silent frames
        Args:
            wav: single channel signal
            fs: sample frequency
            frame_len: default value 20e-3
            thd: the maximal energy difference between frames, default to 40dB
            is_plot: False
        Returns:
            vad result of each frame [N_frame,2], 0: frame start position, 1: frame end position
        """

        frame_size = np.int(fs*frame_len)
        overlap_size = np.int(fs*overlap)

        frame_iter = wav_tools.frame_x([wav],frame_size,overlap_size)
        energy_array = np.asarray([np.sum(frame**2) for [frame] in frame_iter],dtype=np.float32).reshape([-1,])
        max_energy = np.max(energy_array)

        N_frame = energy_array.shape[0]
        vad_result = np.zeros((N_frame,2))
        vad_result[:,0] = np.arange(N_frame)*overlap
        vad_result[:,1] = np.greater(energy_array,max_energy/(10**(thd/10.0)))

        if is_plot:
            silence_frame_index = np.nonzero(energy_array<thd)[0]
            plt.figure(dpi=120)

            frame_iter = wav_tools.frame_x([wav],frame_size,overlap_size)
            for frame_i,[frame] in enumerate(frame_iter):
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


    @staticmethod
    def truncate_speech(wav,fs,frame_len=20e-3,is_plot=False):
        """ energy based VAD,
        """
        frame_size = int(frame_len*fs)
        N_frame = int(wav.shape[0]/frame_size)

        wav_croped = wav[:frame_size*N_frame]

        frames = np.reshape(wav_croped,(frame_size,N_frame),order='F')
        energy_frames = np.sum(frames**2,axis=0)

        thd = np.max(energy_frames)/(10**4)# 40 dB
        vad_frame_index = np.nonzero(energy_frames>thd)[0]
        start_frame_index = vad_frame_index[0]
        end_frame_index = vad_frame_index[-1]

        wav_truncated = np.reshape(frames[:,start_frame_index:end_frame_index+1],[-1,1],order='F')

        return wav_truncated


    @staticmethod
    def frame_data(x,frame_size,shift_len,wind_f=None):
        """
        Input:
            x: data to be framed, 1/2 dimension array(2d:[data_len,channel_num])
            frame_size
            shift_len
            wind_f: 1 dimension array with size of frame_size
        Output:
            framed_data: [N_frame,frame_size,channel_num]
        """
        x_len = x.shape[0]

        if len(x.shape)==1:
            channel_num=1
            x.shape = [x_len,1]
        elif len(x.shape)==2:
            channel_num = x.shape[1]
        else:
            raise Exception('x should have two dimensions at most')

        N_frame=np.int16(np.floor((x_len-frame_size)/shift_len))+1
        x_framed=np.zeros((N_frame,frame_size,channel_num),dtype=np.float32)
    #     wind_f = np.hanning(frame_size)[:,np.newaxis]
        for frame_i in range(N_frame):
            frame_pos=frame_i*shift_len
            x_framed[frame_i,:,:]=x[frame_pos:frame_pos+frame_size,:]

        if wind_f is not None:
            x_framed = np.multiply(x_framed,wind_f[np.newaxis,:,np.newaxis])

        return np.squeeze(x_framed)

    @staticmethod
    def cal_snr(tar,interfer,axis=-1):
        if tar.shape != interfer.shape:
            raise Exception('tar and interfer must have the same shape')
        epsilon = 1e-20
        tar_energy = np.sum(tar**2,axis)+epsilon
        interfer_energy = np.sum(interfer**2,axis)+epsilon
        snr = 10*np.log10(np.divide(tar_energy,interfer_energy))

        return snr


    @staticmethod
    def plot_wav_spec(wav_list,label_list=None,
                      fs=16000,frame_len=20e-3,
                      y_axis_type='mel',figsize=None):
        """
        plot spectrogram of given len
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
            stft = librosa.stft(wav,n_fft=frame_size)
            amp_db_stft = librosa.amplitude_to_db(np.abs(stft))
            librosa.display.specshow(amp_db_stft,sr=fs, x_axis='time', y_axis=y_axis_type)
        plt.tight_layout()
