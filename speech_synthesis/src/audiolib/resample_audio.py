''' This script resamples a list of audio files to a given sample rate.
'''
import sys
import os
import argparse
import librosa
import soundfile
import glob
from tqdm import tqdm


def files_to_list(file_name):
    """
    Takes a text file of filenames and makes a list of filenames.
    (c) https://github.com/NVIDIA/waveglow.
    """
    with open(file_name, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def main():
    path_to_corpus = "D:\\corpus_six\\S004\\SES03"
    final_audio_sampling_rate_Hz = 22000
    
    # Check for cmd line arguments, if the script is used from there.
    parser = argparse.ArgumentParser(description="This script analyses all available wave files and calculates the corresponding mel spectrograms," +
                                                 "based on tacotron's function.")
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
        print("Provided corpus path {:s}".format(final_audio_sampling_rate_Hz))

    parsed_sr = args.sr
    if(parsed_sr == 'SR'):
        print("No sampling rate passed to parser. Will use the one provided in the script.")
    else:
        final_audio_sampling_rate_Hz = parsed_sr
    
    print("Sampling rate : {:d} Hz".format(final_audio_sampling_rate_Hz))

    output_folder_name = "audio_files_resampled"

    audio_files_list_name = os.path.join(path_to_corpus, "audio_files","audio*")
    audio_files_list_name = glob.glob(audio_files_list_name)[0]

    audio_file_names = files_to_list(audio_files_list_name)

    # Save the resampled audio.
    folder_path = os.path.join(path_to_corpus, output_folder_name)
    if not os.path.isdir(folder_path):
        print("Creating folder %s" % (folder_path))
        os.mkdir(folder_path)

    for file_index, audio_file_name in tqdm(enumerate(audio_file_names)):

        full_audio_file_name = os.path.join(path_to_corpus, "audio_files", audio_file_name)

        # Load and resample the audio signal.
        audio_signal, fs_Hz = librosa.load(full_audio_file_name, sr=final_audio_sampling_rate_Hz)
        assert(fs_Hz == final_audio_sampling_rate_Hz)

        full_audio_resampled_file_name = os.path.join(path_to_corpus, output_folder_name, audio_file_name)
        soundfile.write(full_audio_resampled_file_name, audio_signal, final_audio_sampling_rate_Hz)


if __name__=="__main__":
    main()