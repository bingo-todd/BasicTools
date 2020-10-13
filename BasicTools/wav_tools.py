import wave
import os
import numpy as np
import scipy.signal as dsp_tools
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

# from . import plot_tools


def read_wav(wav_path, tar_fs=None):
    """ read wav file, implete with soundfile
    """
    wav_path = os.path.expanduser(wav_path)
    if True:
        x, fs = sf.read(wav_path)
        if tar_fs is not None and tar_fs != fs:
            x = resample(x, fs, tar_fs)
            fs = tar_fs
    else:
        wav_file = wave.open(wav_path, 'r')
        sample_num = wav_file.getnframes()
        channel_num = wav_file.getnchannels()
        fs = wav_file.getframerate()

        pcm_data = wav_file.readframes(sample_num)
        x = np.fromstring(pcm_data, np.int16)/(2.0**15)
        x = x.reshape([-1, channel_num])
        wav_file.close()

    return [x.astype(np.float32), fs]


def write_wav(x, fs, wav_path):
    """ write wav file,  implete with soundfile
    """
    wav_path = os.path.expanduser(wav_path)
    if True:
        sf.write(file=wav_path, data=x, samplerate=fs)
    else:
        bits_per_sample = 16
        samples = np.asarray(x*(2**(bits_per_sample)), dtype=np.int16)
        sample_num = samples.shape[0]
        if samples.ndim > 1:
            channel_num = samples.shape[1]
        else:
            channel_num = 1

        wav_file = wave.open(wav_path, 'w')
        wav_file.setparams((channel_num, 2, fs, sample_num, 'NONE',
                            'not compressed'))
        wav_file.writeframes(samples.tostring())
        wav_file.close()


def resample(x, orig_fs, tar_fs):
    """ resample signal, implete with librosa
    Args:
        x: signal, resampling in the first dimension
        orig_fs: original sample frequency
        tar_fs: target sample frequency
    Returns:
        resampled data
    """
    x_result = librosa.resample(x.T, orig_fs, tar_fs)
    return x_result.T


def brir_filter(x, brir):
    """ synthesize spatial recording
    Args:
        x: single-channel signal
        brir: binaural room impulse response
    Returns:
        spatialized signal
    """
    if (len(x.shape) > 1) and (x.shape[1] > 1):
        raise Exception('x has mutliple channels')
    signal_len = x.shape[0]
    y = np.zeros((signal_len, 2), dtype=np.float64)
    for channel_i in range(2):
        y[:, channel_i] = np.squeeze(
                            dsp_tools.lfilter(brir[:, channel_i], 1,
                                              x, axis=0))
    return y


def cal_power(x):
    """calculate the engergy of given signal
    """
    theta = x.max()/1e5
    x_len = np.count_nonzero(x > theta)
    power = np.sum(np.square(x))/x_len
    return power


def frame_data(x, frame_len, shift_len):
    """parse data into frames
    Args:
        x: single/multiple channel data
        frame_len: frame length in sample
        shift_len: shift_len in sample
    Returns:
        [n_frame,frame_len,n_chann]
    """
    x = np.asarray(x)

    if frame_len < 1:
        return x
    if frame_len == 1:
        return np.expand_dims(x, axis=1)

    # ensure x is 2d array
    n_dim = len(x.shape)
    if n_dim == 1:
        x = x[:, np.newaxis]

    n_sample, *sample_shape = x.shape
    n_frame = np.int(np.floor(np.float32(n_sample-frame_len)/shift_len)+1)
    frame_all = np.zeros((n_frame, frame_len, *sample_shape))
    for frame_i in range(n_frame):
        frame_slice = slice(frame_i*shift_len, frame_i*shift_len+frame_len)
        frame_all[frame_i] = x[frame_slice]

    if n_dim == 1:
        frame_all = np.squeeze(frame_all)
    return frame_all


def set_snr(x, ref, snr):
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
    coef = np.sqrt(np.float_power(10, float(snr)/10)
                   / (power_x / power_ref))
    return coef*x.copy()


def _cal_snr(tar, inter):
    """sub-function of cal_snr"""
    power_tar = cal_power(tar)
    power_inter = cal_power(inter)
    snr = 10*np.log10(power_tar/power_inter)
    return snr


def cal_snr(tar, inter, frame_len=None, shift_len=None, is_plot=None):
    """Calculate snr of entire signal, frames if frame_len is
    specified.
                snr = 10log10(power_tar/power_inter)
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
        snr = _cal_snr(tar, inter)
    else:
        if shift_len is None:
            shift_len = np.int16(frame_len/2)

        # signal length check
        if tar.shape[0] != inter.shape[0]:
            raise Exception('tar and inter do not have the same length,\
                             tar:{}, inter:{}'.format(tar.shape[0],
                                                      inter.shape[0]))

        frame_all_tar = frame_data(tar, frame_len, shift_len)
        frame_all_inter = frame_data(inter, frame_len, shift_len)
        n_frame = frame_all_tar.shape[0]
        snr = np.asarray([_cal_snr(frame_all_tar[i],
                                   frame_all_inter[i])
                          for i in range(n_frame)])
        if is_plot:
            n_sample = tar.shape[0]
            # waveform of tar and inter
            fig = plt.figure()
            ax1 = fig.subplots(1, 1)
            time_axis = np.arange(n_sample)
            ax1.plot(time_axis, tar[:n_sample], label='tar')
            ax1.plot(time_axis, inter[:n_sample], label='inter')
            ax1.set_xlabel('time(s)')
            ax1.set_ylabel('amp')
            ax1.legend(loc='upper left')

            # snrs of frames
            ax2 = ax1.twinx()
            ax2.set_ylabel('snr(dB)')
            # time: center of frame
            frame_t_all = np.arange(n_frame)*shift_len+np.int16(frame_len/2)
            ax2.plot(frame_t_all, snr, color='red', linewidth=2,
                     label='snr')
            ax2.legend(loc='upper right')
    return snr


def gen_wn(shape, ref=None, energy_ratio=0, power=1):
    """Generate Gaussian white noise with either given energy ration related
    to ref signal or given power
    Args:
        shape: the shape of white noise to be generated,
        ref: reference signal
        energy_ratio: energy ration(dB) between white noise and reference
            signal, default to 0 dB
        power:
    Returns:
        white noise
    """
    wn = np.random.normal(0, 1, size=shape)
    if ref is not None:
        wn = set_snr(wn, ref, energy_ratio)
    else:
        power_orin = np.sum(wn**2, axis=0)/shape[0]
        coef = np.sqrt(power/power_orin)
        wn = wn*coef
    return wn


def truncate_data(x, trunc_type="both", eps=1e-5):
    """truncate small-value sample in the first dimension
    Args:
        x: data to be truncated
        trunc_type: specify parts to be cliped, options: begin,end,
            both(default)
        eps: amplitude threshold
    Returns:
        data truncated
    """
    valid_sample_pos = np.nonzero(x > eps)[0]
    start_pos = 0
    end_pos = x.shape[0]
    if type in ['begin', 'both']:
        start_pos = np.min(valid_sample_pos)
    if type in ['end', 'both']:
        end_pos = np.max(valid_sample_pos)
    return x[start_pos:end_pos+1]


def hz2erbscal(freq):
    """convert Hz to ERB scale"""
    return 21.4*np.log10(4.37*freq/1e3+1)


def erbscal2hz(erb_num):
    """convert ERB scale to Hz"""
    return (10**(erb_num/21.4)-1)/4.37*1e3


def cal_erb(cf):
    """calculate the ERB(Hz) of given center frequency based on equation
    given by Glasberg and Moore
    Args
        cf: center frequency Hz, single value or numpy array
    """
    return 24.7*(4.37*cf/1000+1.0)


def test():
    print('test')
    wav_fpath = 'resource/tar.wav'
    print(wav_fpath)
    data, fs = read_wav(wav_fpath)

    # vad
    # vad_flag_all,fig = vad(data,fs,frame_dur=20e-3,is_plot=True,theta=20)
    # savefig.savefig(fig,fig_name='vad',fig_dir='./images/wav_tools')

    # wave
    # fig = plot_wav_spec(data)
    # plot_tools.savefig(fig,name='wav_spec',dir='./images/wav_tools')


if __name__ == '__main__':
    test()
