''' 
    Small script to normalize the loundness of the audio files. 
'''

import string
import argparse
import os
import glob
from src.audiolib import audio
import sys
from tqdm import tqdm

def main():
    target_loudness_lufs = -30.0
    path_to_audio_files = "C:\\Users\\chris\\Documents\\Institut\\corpora\\speech_synthesis\\corpus_six\\S004\\audio_files"

    # Check for cmd line arguments, if the script is used from there.
    parser = argparse.ArgumentParser(description="Normalizes the loundless of a list of audio files.")
    parser.add_argument("--path", 
                        help="The system path up to the audio files folder.",
                        default="PATH")

    # Parse the input arguments from the command line.
    args = parser.parse_args()
    parsed_path = args.path
    if(parsed_path == "PATH"):
        print("No cmd line path provided. The path provided in the script will be used.")
    else:
        path_to_audio_files = parsed_path
        print("Provided corpus path {:s}".format(path_to_audio_files))

    # Check if audio files can safetly be overridden.
    is_backup_check_completed = False
    print("WARNING: Normalizing the audio files replaces the original file. Please backup the"
          + "original audio files first. Do you want to proceed? (y/n)")
    while not is_backup_check_completed:
        input_string = input("Decision: ")
        if(input_string == "y"):
            is_backup_check_completed = True
        elif(input_string == 'n'):
            sys.exit()
        else:
            print("Invalid input. Enter 'y' or 'n'")

    # Collect the file information.
    file_info = glob.glob(os.path.join(path_to_audio_files, "S0*"))

    if not file_info:
        raise IOError("Error: No files found at {:s}".format(os.path.join(path_to_audio_files, "S0*")))

    backup_folder_name = "audio_files_original".format(target_loudness_lufs)
    backup_folder_path = os.path.join(path_to_audio_files, backup_folder_name)

    for full_file_name in tqdm(file_info):
        audio.normalize_loudness(file_path=full_file_name, 
                                       target_loudness=target_loudness_lufs)
    

if __name__ == "__main__":
    main()