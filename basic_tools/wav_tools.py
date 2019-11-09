import wave
import os
import numpy as np
import scipy.signal as dsp_tools
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

import sys
sys.path.append('/home/st/Work_Space/module_st/basic-toolbox-develop/basic_tools')
import fft
import plot_tools


def wav_read(fpath,tar_fs=None):
    """ read wav file, implete with soundfile
    """
    if True:
        x,fs = sf.read(fpath)
        if tar_fs is not None and tar_fs != fs:
            x=resample(x,fs,tar_fs)
            fs = tar_fs
    else:
        wav_file = wave.open(fpath,'r')
        sample_num = wav_file.getnframes()
        channel_num = wav_file.getnchannels()
        fs = wav_file.getframerate()

        pcm_data = wav_file.readframes(sample_num)
        x = np.fromstring(data,np.int16)/(2.0**15)
        x = x.reshape([-1,channel_num])
        wav_file.close()

    return [x,fs]


def wav_write(x,fs,fpath):
    """ write wav file,  implete with soundfile
    """
    if True:
        sf.write(file=fpath,data=x,samplerate=fs)
    else:
        bits_per_sample = 16
        samples = np.asarray(x*(2**(bits_per_sample)),dtype=np.int16)
        sample_num = samples.shape[0]
        if samples.ndim > 1:
            channel_num = samples.shape[1]
        else:
            channel_num = 1

        wav_file = wave.open(fpath,'w')
        wav_file.setparams((channel_num, 2, fs, sample_num, 'NONE', 'not compressed'))
        wav_file.writeframes(samples.tostring())
        wav_file.close()


def resample(x,orig_fs,tar_fs):
    """ resample signal, implete with librosa
    Args:
        x: signal, resampling in the first dimension
        orig_fs: original sample frequency
        tar_fs: target sample frequency
    Returns:
        resampled data
    """
    x_result = librosa.resample(x.T,orig_fs,tar_fs)
    return x_result.T


def brir_filter(x,brir,is_gpu=False,gpu_index=0):
    """ synthesize spatial recording
    Args:
        x: single-channel signal
        brir: binaural room impulse response
    Returns:
        spatialized signal
    """
    if len(x.shape)>1 and x.shape[1]>1:
        raise Exception('x has mutliple channels')
    signal_len = x.shape[0]
    y = np.zeros((signal_len,2),dtype=np.float64)
    if is_gpu:
        import filter_gpu
        filter_gpu_obj = filter_gpu.filter_gpu(gpu_index=gpu_index)
        for channel_i in range(2):
            y[:,channel_i] = filter_gpu_obj.filter(x=x,coef=brir[:,channel_i])
    else:
        for channel_i in range(2):
            y[:,channel_i] = np.squeeze(dsp_tools.lfilter(brir[:,channel_i],1,
                                                          x,axis=0))
    return y


def cal_power(x):
    """calculate the engergy of given signal
    """
    theta = x.max()/1e5
    x_len = np.count_nonzero(x>theta)
    power = np.sum(np.square(x))/x_len
    return power


def frame_data(x,frame_len,overlap):
    """parse data into frames
    Args:
        x: single/multiple channel data
        frame_len: frame length in sample
        overlap: overlap in sample
    Returns:
        [n_frame,frame_len,n_chann]
    """
    if frame_len <= 1:
        return x

    # ensure x is 2d array
    if len(x.shape) == 1:
        x = x[:,np.newaxis]

    n_sample,n_chann = x.shape
    n_frame = np.int(np.floor(np.float32(n_sample-frame_len)/overlap)+1)
    frames = np.zeros((n_frame,frame_len,n_chann))
    for frame_i in range(n_frame):
        frame_slice = slice(frame_i*overlap,frame_i*overlap+frame_len)
        frames[frame_i] = x[frame_slice]
    return np.squeeze(frames)


def set_snr(x,ref,snr):
    """ scale signal to a certain snr relative to ref
    Args:
        x: signal to be scaled
        ref: reference signal
        snr:
    Returns:
        scaled target signal
    """
    power_x = cal_power(x)
    power_ref = cal_power(ref)
    coef = np.sqrt(np.float_power(10,float(snr)/10)/\
                                (power_x/power_ref))
    return x*coef


def _cal_snr(tar,inter):
    """sub-function of cal_snr"""
    power_tar = cal_power(tar)
    power_inter = cal_power(inter)
    snr = 10*np.log10(power_tar/power_inter)


def cal_snr(tar,inter,frame_len=None,overlap=None,is_plot=None):
    """Calculate snr of entire signal, frames if frame_len is
    specified.
                snr = 10log10(power_tar/power_inter)
    Args:
        tar: target signal, single channel
        inter: interfere signal, single channel
        frame_len:
        overlap: if not specified, set to frame_len/2
        if_plot: whether to plot snr of each frames, default to None
    Returns:
        float number or numpy.ndarray
    """
    if frame_len is None:
        snr=_cal_snr(tar,inter)
    else:
        if overlap is None:
            overlap = np.int16(frame_len/2)

        # signal length check
        if tar.shape[0] != inter.shape[0]:
            raise Exception('tar and inter do not have the same length,\
                             tar:{}, inter:{}'.format(tar.shape[0],
                                                      inter.shape[0]))

        frames_tar = frame_data(tar,frame_len,overlap)
        frames_inter = frame_data(inter,frame_len,overlap)
        n_frame = frames_tar.shape[0]
        snr = np.asarray([_cal_snr(frames_tar[i],frames_inter[i])
                                                for i in range(n_frame)])
        if is_plot:
            n_sample = tar.shape[0]
            # waveform of tar and inter
            fig = plt.figure()
            ax1 = fig.subplots(1,1)
            t_samples = np.arange(n_sample)/fs
            ax1.plot(time_axis,tar[:n_sample],label='tar')
            ax1.plot(time_axis,inter[:n_sample],label='inter')
            ax1.set_xlabel('time(s)'); axe1.set_ylabel('amp')
            ax1.legend(loc='upper left')

            # snrs of frames
            ax2 = ax1.twinx()
            axe2.set_ylabel('snr(dB)')
            # time: center of frame
            t_frames = np.arange(n_frame)*overlap+np.int16(frame_len/2)
            ax2.plot(t_frames,snrs,color='red',linewidth=2,label='snr')
            ax2.legend(loc='upper right')
            plt.tight_layout()
    return snr


def gen_whitenoise(n_sample,power=1):
    """Generate Gaussian white noise
    Args:
        n_sample
        power
    Returns:
        white noise
    """
    wn = np.random.normal(0,1,size=n_sample)
    coef = np.sqrt(power/(np.sum(wn**2/n_sample)))
    return wn*coef


def vad(x,fs,frame_len,overlap=None,theta=40,is_plot=False):
    """ Energy based vad.
        1. Frame data with overlap of 0
        2. Calculte the energy of each frame
        3. Frames with energy below max_energy-theta is regarded as silent frames
    Args:
        x: single channel signal
        frame_len: frame length
        overlap: frames shift length in time
        theta: the maximal energy difference between frames, default 40dB
        is_plot: whether to ploting vad result, default False
    Returns:
        vad_labels, as well as figures of vad_labesl if is_plot is ture
    """
    if overlap is None:
        overlap = frame_len

    frames = frame_data(x,frame_len,overlap)
    energy_frames = np.sum(frames**2,axis=1)
    energy_thd = np.max(energy_frames)/(10**(theta/10.0))
    vad_labels = np.greater(energy_frames,energy_thd)

    if is_plot and (frame_len==overlap):
        # if dpi is low, speech line and silence line will be overlapped
        fig = plt.figure(dpi=500)
        ax = fig.subplots(1,1)
        line_speech = None
        line_silence = None
        n_frame = frames.shape[0]
        for frame_i in range(n_frame):
            frame = frames[frame_i]
            start_pos = frame_i*overlap
            end_pos = start_pos+frame_len
            if vad_labels[frame_i] == True:
                [line_speech] = ax.plot(np.arange(start_pos,end_pos),frame,
                                        linewidth=1,color='red')
            else:
                # np.arange(start_pos,end_pos)/fs,
                [line_silence] = ax.plot(np.arange(start_pos,end_pos),frame,
                                         linewidth=1,color='blue')
        if line_speech is not None:
            line_speech.set_label('speech')
        if line_silence is not None:
            line_silence.set_label('silence')
        ax.legend()
        ax.set_xlabel('time(s)')
        ax.set_ylabel('amp')
        ax.set_title('threshold={}dB'.format(theta))
        return [vad_labels,fig]
    else:
        return vad_labels


def truncate_data(x,type="both",eps=1e-5):
    """truncate small-value sample in the first dimension
    Args:
        x: data to be truncated
        type: specify parts to be cliped, options: begin,end,both(default)
        eps: amplitude threshold
    Returns:
        data truncated
    """
    valid_sample_pos = np.nonzero(x>eps)[0]
    start_pos = 0
    end_pos = x.shape[0]
    if type in ['begin','both']:
        start_pos = np.min(valid_sample_pos)
    if type in ['end','both']:
        end_pos = np.max(valid_sample_pos)
    return x[start_pos:end_pos+1]


def hz2erbscal(self,freq):
    """convert Hz to ERB scale"""
    return 21.4*np.log10(4.37*freq/1e3+1)


def erbscal2hz(self,erb_num):
    """convert ERB scale to Hz"""
    return (10**(erb_num/21.4)-1)/4.37*1e3


def cal_erb(self,cf):
    """calculate the ERB(Hz) of given center frequency based on equation
    given by Glasberg and Moore
    Args
        cf: center frequency Hz, single value or numpy array
    """
    return 24.7*(4.37*cf/1000+1.0)


def cal_bw(self,cf):
    """calculate the 3-dB bandwidth
    Args
        cf: center frequency Hz, single value or numpy array
    """
    erb = self.cal_ERB(cf)
    return 1.019*erb


def test():
    print('test')
    wav_fpath = 'resource/tar.wav'
    print(wav_fpath)
    data,fs = wav_read(wav_fpath)

    # vad
    # vad_labels,fig = vad(data,fs,frame_dur=20e-3,is_plot=True,theta=20)
    # savefig.savefig(fig,fig_name='vad',fig_dir='./images/wav_tools')

    # wave
    fig = plot_wav_spec(data)
    plot_tools.savefig(fig,name='wav_spec',dir='./images/wav_tools')


if __name__ == '__main__':
    test()
