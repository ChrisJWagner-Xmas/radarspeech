Training and label file locations are stored as
.csv lists. This makes it easier to expand to 
a large dataset compared to saving them as a binary corpus
that is loaded once.

The absolute path to all files is stored in absolute_path_to_files.txt

File structure:

/audio_files
  |
  -- S00x_SES0x_sentence1_audio_file.wav
  -- S00x_SES0x_sentence2_audio_file.wav
  -- ...
  -- S00x_SES0x_sentenceN_audio_file.wav

/radar_files
  |
  -- S00x_SES0x_sentence1_radar_file.wav
  -- S00x_SES0x_sentence2_radar_file.wav
  -- ...
  -- S00x_SES0x_sentenceN_radar_file.wav

/vocoder_files
  |
  -- S00x_SES0x_sentence1_vocoder _file.wav
  -- S00x_SES0x_sentence2_vocoder _file.wav
  -- ...
  -- S00x_SES0x_sentenceN_vocoder _file.wav
