import simpleaudio as sa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

def play_audio(audio_data, fs_Hz):
    ''' Function to convert a numpy array to 16 bit integer array
        and serialize it to play it with simpleaudio.
        Args: 
            audio_data (1d-numpy array): array holding the audio samples.
            fs_Hz (scalar): Sampling frequency.

        Returns: 
            None.
    '''
    audio_data_16bit = audio_data
    nan_indices = np.argwhere(np.isnan(audio_data_16bit))
    if(nan_indices.size > 0):
        print("Error: there are %d NaN values in the waveform. Play aborted." %(nan_indices.size))
        return

    max_value = np.max(abs(audio_data))

    audio_data_16bit = audio_data_16bit*32767/max_value
    audio_data_16bit = audio_data_16bit.astype(np.int16)
    play_obj = sa.play_buffer(audio_data_16bit, 1, 2, fs_Hz)
    play_obj.wait_done()
    
    return audio_data_16bit


def normalize_loudness(file_path, target_loudness=-23.0):
    """
    this function reads an audio file, normalize and overwrites the old audio file, Normalization ITU-R Standard
    Args:
        file_path (str): file path to the original sound file.
        target_loudness (float): defines the target loudnes in LUFS, default=-23 LUFS.

    Returns:
        None.

    """

    audio_data, fs_Hz = sf.read(file_path)
    meter = pyln.Meter(fs_Hz)
    loudness = meter.integrated_loudness(audio_data)
    audio_data_norm = pyln.normalize.loudness(audio_data, loudness, target_loudness)
    sf.write(file_path, audio_data_norm, fs_Hz)
    print("Normalized from {:1.4f} to {:1.4f} loudness.".format(loudness,target_loudness))

