
''' This script calculates the mel spectrograms from a given set of audio files
    according to the method used by Tacotron2.
'''

import sys
import os
import argparse
from scipy.io.wavfile import read
from src.fileio import file_io_functions
from src.fileio import import_rs_data
sys.path.insert(0, 'src\\extern\\tacotron2')
import torch
import matplotlib.pyplot as plt
from src.extern.tacotron2.layers import TacotronSTFT
import logging
import librosa
import numpy as np
import glob
from tqdm import tqdm

def main():
    #################### INPUT ########################################

    path_to_corpus = "C:\\Users\\chris\\Documents\\Institut\\corpora\\speech_synthesis\\corpus_six\\S004"
    final_audio_sampling_rate_Hz = 22000

    ###################################################################

    # Check for cmd line arguments, if the script is used from there.
    parser = argparse.ArgumentParser(description="This script analyses all available wve files and calculates the corresponding mel spectrograms," +
                                                 "based on tacotron")
    parser.add_argument("--path", 
                        help="The system path up to where the radar_files & audio_files folders are located.",
                        default="PATH")

    parser.add_argument("--sr", 
                        help="The final sampling rate of the audio signals.",
                        default='SR')

    args = parser.parse_args()
    parsed_path = args.path
    if(parsed_path == "PATH"):
        print("No cmd line path provided. The path provided in the script will be used.")
    else:
        print("Provided corpus path {:s}".format(parsed_path))
        path_to_corpus = parsed_path

    parsed_sr = args.sr
    if(parsed_sr == 'SR'):
        print("No sampling rate passed to parser. Will use the one provided in the script.")
    else:
        final_audio_sampling_rate_Hz = parsed_sr
    
    if(final_audio_sampling_rate_Hz%100 != 0):
        raise ValueError("The final audio sampling rate needs to be divisible by 100 (the radar frame rate).")

    print("Sampling rate : {:d} Hz".format(final_audio_sampling_rate_Hz))

    filter_length = 1024 # FFT/window size
    hop_length = final_audio_sampling_rate_Hz//100 # Align with radar spectra
    win_length = filter_length
    mel_fmin = 0.0
    mel_fmax = 8000.0
    num_mel_coeffs = 80

    MAX_WAV_VALUE = 32768.0

    stft = TacotronSTFT(filter_length=filter_length, 
                        hop_length=hop_length, 
                        win_length=win_length,
                        n_mel_channels=num_mel_coeffs, 
                        sampling_rate=final_audio_sampling_rate_Hz, 
                        mel_fmin=mel_fmin,
                        mel_fmax=mel_fmax)

    logging.basicConfig(level=logging.INFO) # force=True
    logger = file_io_functions.setup_logger("logger", "mel_spectrograms_calculation.log")

    output_folder_name = "tacotron_mel_power_spectrogram"

    # Create the output folder for the mel-spectrograms.
    folder_path = os.path.join(path_to_corpus, output_folder_name)
    if not os.path.isdir(folder_path):
        print("Creating folder %s" % (folder_path))
        os.mkdir(folder_path)

    # Also create a folder for the numpy files.
    folder_path = os.path.join(path_to_corpus, output_folder_name, "npy")
    if not os.path.isdir(folder_path):
        print("Creating folder %s" % (folder_path))
        os.mkdir(folder_path)

    audio_files_path = os.path.join(path_to_corpus,"audio_files","*.wav")
    full_audio_file_names = glob.glob(audio_files_path)

    for file_index, audio_file_name in tqdm(enumerate(full_audio_file_names)):

        full_audio_file_name = os.path.join(path_to_corpus, "audio_files", audio_file_name)
        # Get the session id and subject id from the file name.
        subject_id, session_id, sentence_number = file_io_functions.get_sentence_info(full_audio_file_name)
        file_preemble = "{:s}_{:s}".format(subject_id, session_id)

        # print("Calculating mel spectrogram for file {:s}".format(full_audio_file_name))

        # Calculate the mel spectrogram.
        read_fs_Hz, audio_raw = read(full_audio_file_name)

        if(read_fs_Hz != final_audio_sampling_rate_Hz):
            raise ValueError("Error: sampling rate mismatch, Read audio: {:d} Hz, User-specified: {:d} Hz.".format(read_fs_Hz, final_audio_sampling_rate_Hz))

        audio_signal = audio_raw/MAX_WAV_VALUE
        
        min_audio_value = np.min(audio_signal)
        max_audio_value = np.max(audio_signal)

        if(min_audio_value < -1 or max_audio_value > 1):
            print("Adjusted Min/Max signal values: {:1.5f}/{:1.5f} to be within [-1,1]".format(min_audio_value, max_audio_value))
            audio_signal = (2*(audio_signal - min_audio_value)/(max_audio_value - min_audio_value) - 1)*0.999
    
        # audio_signal, fs_Hz = librosa.load(full_audio_file_name, sr=final_audio_sampling_rate_Hz)
        # Note: both audio loading methods should result in very similar results

        audio_signal = torch.from_numpy(audio_signal).float().unsqueeze(0)
        mel_tacotron = stft.mel_spectrogram(audio_signal).squeeze(0).T.numpy() # needs to be [num_frames, num_mel_coeffs] for write out
        num_mel_spectra, _  = mel_tacotron.shape

        # Load the corresponding radar frames for length checking.
        full_radar_file_name = os.path.join(path_to_corpus, "radar_files", "{}_{}_{}.bin".format(subject_id, session_id, sentence_number))
        radar_data = import_rs_data.import_rs_binary_file(full_radar_file_name)

        if(abs(radar_data.num_total_frames - num_mel_spectra) > 1):
            logger.error("Frame offset between radar frames (%d) and vocoder parameter frames (%d) > 1 detected" % (radar_data.num_total_frames, num_mel_spectra)
                                + "The radar sequence must be equal to or contain one frame more than the mel spectrogram at most."
                                + "File %s was skipped" % (full_radar_file_name))
            continue
        else:
            # Pad the vocoder parameters by one frame if an offset exists and the files were loaded correctly.
            if(radar_data.num_total_frames != num_mel_spectra):
                if(radar_data.num_total_frames > num_mel_spectra):
                    # Pad the mel spectrogram.
                    mel_tacotron_padded = np.empty((radar_data.num_total_frames, num_mel_coeffs))
                    mel_tacotron_padded[0:num_mel_spectra,:] = mel_tacotron
                    mel_tacotron_padded[num_mel_spectra,:] = mel_tacotron[-1,:] # pad last value.
                    mel_tacotron = mel_tacotron_padded
                    logger.info("Padded mel spectrogram by 1 frame for file %s." % (full_radar_file_name))
                if(num_mel_spectra > radar_data.num_total_frames):
                    # Crop the mel spectrogram by one.
                    mel_tacotron = mel_tacotron[0:num_mel_spectra-1,:]
                    logger.info("Cropped mel spectrogram by 1 frame for file %s." % (full_radar_file_name))

        mel_spectrogram_file_name = "{:s}_{:s}_tacotron_mel_power_spectrogram.csv".format(file_preemble, sentence_number)
        full_mel_spectrogram_file_name = os.path.join(path_to_corpus, output_folder_name, mel_spectrogram_file_name)
        np.savetxt(full_mel_spectrogram_file_name, mel_tacotron, delimiter=',')

        # Also save the mel spectrograms in numpy format.
        mel_spectrogram_file_name = mel_spectrogram_file_name.split('.')[0] + ".npy"
        full_mel_spectrogram_file_name = os.path.join(path_to_corpus, output_folder_name, "npy", mel_spectrogram_file_name)
        mel_tacotron = mel_tacotron.T
        mel_tacotron = mel_tacotron[np.newaxis, :, :]
        np.save(full_mel_spectrogram_file_name, mel_tacotron) # Save as [1, num_mels, seq_length]

if __name__ == "__main__":
    main()