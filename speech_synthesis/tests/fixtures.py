def return_test_data_file_names():
    ''' 
        Returns the file names for some test data files.
    '''
    
    radar_files_names = ['S001_SES01_s01.bin',
                        'S001_SES01_s02.bin',
                        'S001_SES01_s03.bin',
                        'S001_SES01_s04.bin',
                        'S001_SES01_s05.bin',    
                        'S001_SES01_s06.bin',    
                        'S001_SES01_s07.bin',    
                        'S001_SES01_s08.bin',    
                        'S001_SES01_s09.bin',    
                        'S001_SES01_s10.bin']    

    vocoder_params_f0 = 	['S001_SES01_s01_f0.csv',
                             'S001_SES01_s02_f0.csv',
                             'S001_SES01_s03_f0.csv',
                             'S001_SES01_s04_f0.csv',
                             'S001_SES01_s05_f0.csv',
                             'S001_SES01_s06_f0.csv',
                             'S001_SES01_s07_f0.csv',
                             'S001_SES01_s08_f0.csv',
                             'S001_SES01_s09_f0.csv',
                             'S001_SES01_s10_f0.csv']

    vocoder_params_spectrogram = ['S001_SES01_s01_spectrogram.csv',
                                  'S001_SES01_s02_spectrogram.csv',
                                  'S001_SES01_s03_spectrogram.csv',
                                  'S001_SES01_s04_spectrogram.csv',
                                  'S001_SES01_s05_spectrogram.csv',
                                  'S001_SES01_s06_spectrogram.csv',
                                  'S001_SES01_s07_spectrogram.csv',
                                  'S001_SES01_s08_spectrogram.csv',
                                  'S001_SES01_s09_spectrogram.csv',
                                  'S001_SES01_s10_spectrogram.csv']

    vocoder_params_coarse_ap = ['S001_SES01_s01_coarse_ap.csv',
                                'S001_SES01_s02_coarse_ap.csv',
                                'S001_SES01_s03_coarse_ap.csv',
                                'S001_SES01_s04_coarse_ap.csv',
                                'S001_SES01_s05_coarse_ap.csv',
                                'S001_SES01_s06_coarse_ap.csv',
                                'S001_SES01_s07_coarse_ap.csv',
                                'S001_SES01_s08_coarse_ap.csv',
                                'S001_SES01_s09_coarse_ap.csv',
                                'S001_SES01_s10_coarse_ap.csv']

    vocoder_params_mfcc = ['S001_SES01_s01_mfcc.csv',
                           'S001_SES01_s02_mfcc.csv',
                           'S001_SES01_s03_mfcc.csv',
                           'S001_SES01_s04_mfcc.csv',
                           'S001_SES01_s05_mfcc.csv',
                           'S001_SES01_s06_mfcc.csv',
                           'S001_SES01_s07_mfcc.csv',
                           'S001_SES01_s08_mfcc.csv',
                           'S001_SES01_s09_mfcc.csv',
                           'S001_SES01_s10_mfcc.csv']

    vocoder_params_files_names = {'f0' : vocoder_params_f0,
                                  'coarse_ap' : vocoder_params_coarse_ap,
                                  'spectrogram' : vocoder_params_spectrogram,
                                  'mfcc' : vocoder_params_mfcc,
                                  }

    file_paths = {'radar_files': radar_files_names,
                  'input_params_files': vocoder_params_files_names}

    return file_paths


def return_all_vocoder_parameter_and_dims():
    param_names = ['f0', 'coarse_ap', 'spectrogram', 'mfcc']
    param_dims = [1,5,1025,35]
    return dict(zip(param_names, param_dims))


def return_test_sequence_lengths():
    return [66, 75, 78, 68, 48, 45, 60, 68, 55, 62]

