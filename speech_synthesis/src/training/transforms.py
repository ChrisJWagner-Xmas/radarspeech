import torch
import numpy as np
import matplotlib.pyplot as plt

# Constants.
SPECTRAL_MAG_MIN_VALUE_DB = -45.0 # Approximate noise floor for the measurements
SPECTRAL_MAG_MAX_VALUE_DB = 6.03  # Maximum of the "s-parameter" for the hardware

# Functions.
def calc_delta_sequence(sequence):
    ''' Calculates the delta sequence of the input sequence (finite forward differences).
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        delta_sequence (2-d tensor): output sequence
    Note: 
        This function has been unit-tested in the command_word_classification code base.
    '''
    num_frames, num_steps = sequence.shape

    # Allocate new sequence tensor to not modify the input sequence.
    delta_sequence = torch.empty((num_frames, num_steps), dtype=sequence.dtype)

    for frame_index in range(1, num_frames):
        delta_sequence[frame_index-1, :] = sequence[frame_index, :] - sequence[frame_index-1, :]
    delta_sequence[-1, :] = delta_sequence[-2, :] # pad the last frame.

    return delta_sequence


def calc_magnitude(sequence):
    ''' Calculate the linear magnitude of a complex sequence
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        sequence (2-d tensor): magnitude sequence.
    '''
    return torch.abs(sequence)


def calc_magnitude_db(sequence):
    ''' Calculate the magnitude in [dB] of a complex sequence
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        sequence (2-d tensor): magnitude sequence.
    '''
    sequence = 20*torch.log10(torch.abs(sequence))
    sequence[sequence < SPECTRAL_MAG_MIN_VALUE_DB] = SPECTRAL_MAG_MIN_VALUE_DB
    return sequence


def calc_phase_angle(sequence):
    ''' Calculate the phase angle of a complex sequence
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        sequence (2-d tensor): phase angle sequence.
    '''
    return torch.angle(sequence)


def calc_delta_magnitude(sequence):
    ''' Calculates the linear delta magnitude of a complex sequence
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        sequence (2-d tensor): delta magnitude sequence.
    '''
    sequence = calc_delta_sequence(sequence)
    return torch.abs(sequence)


def calc_delta_magnitude_db(sequence):
    ''' Calculates the delta magnitude in [dB] of a complex sequence
    Args:
        sequence (2-d tensor): input sequence
    Returns:
        sequence (2-d tensor): delta magnitude sequence.
    '''
    sequence = calc_delta_sequence(sequence)
    sequence = 20*torch.log10(torch.abs(sequence))
    sequence[sequence < SPECTRAL_MAG_MIN_VALUE_DB] = SPECTRAL_MAG_MIN_VALUE_DB

    return sequence


def calc_delta_phase(sequence):
    sequence = calc_delta_sequence(sequence)
    return torch.angle(sequence)


def _get_impulse_response_function(imp_resp_win_len):

    ''' Getter function for the parameterized impulse response
        calculation function.
    Args:
        imp_resp_win_len (int): length of the impulse response in samples,
                                centered at the center of the impulse response (length N) from
                                [N/2 - imp_resp_win_len: N/2+imp_resp_win_len]
    Returns:
        calc_impulse_response (fcn): impulse response calculation function.
    '''
    if(imp_resp_win_len % 2 != 0):
        raise ValueError("Impulse response window length must be even.")
    if(imp_resp_win_len < 0):
        raise ValueError("Impulse response window length must be >= 0.")

    def calc_impulse_response(sequence):
        '''
            Calculates the impulse response from the radar spectrum.
            A spectrum of N points results in an impulse length of 2*N-2.
            This impulse response is then cropped to its second half and w.r.t.
            to the passed impulse response length, i.e., [N-1:N-1+imp_resp_win_len].
        Args:
            sequence (2-d tensor): input sequence.
        Returns:
            sequence (2-d tensor): impulse responses over time.
        '''
        num_frames, num_steps = sequence.shape

        if(imp_resp_win_len > (num_steps*2-2)/2):
            raise ValueError("Impulse response window length must be <= %d but is %d." % ((num_steps*2-2)/2, imp_resp_win_len))

        # Allocate new sequence tensor to not modify the input sequence.
        impulse_response = torch.empty((num_frames, num_steps*2-2), dtype=torch.float32)

        for frame_index in range(num_frames):
            # Convert to numpy for irfft b.c. HPC does not support torch.fft.irfft()...
            frame = np.fft.irfft(sequence[frame_index,:].numpy())
            impulse_response[frame_index,:] = torch.from_numpy(frame)

        # Crop the impulse response to the specified window length.
        impulse_response = impulse_response[:,num_steps-1-imp_resp_win_len//2:num_steps-1+imp_resp_win_len//2]

        return impulse_response

    return calc_impulse_response


def _get_calc_delta_impulse_response_function(imp_resp_win_len):

    imp_resp_func = _get_impulse_response_function(imp_resp_win_len)
    
    def calc_delta_impulse_response(sequence):
        sequence = imp_resp_func(sequence)
        sequence = calc_delta_sequence(sequence)
        return sequence

    return calc_delta_impulse_response


def _get_crop_function(freq_end_index):   

    def crop_sequence(sequence):
        '''
            Args: 
                sequence (2-d tensor): sequence to be cropped.
            Returns:
                cropped sequence (2-d tensor).
        '''
        return sequence[:,:freq_end_index]

    return crop_sequence


def _get_normalization_function(norm_type, norm_interval):
    ''' Parameterize the normalization function.
    Args:
        norm_interval (list(int)): list with [lower bound, upper bound]
    '''
    def normalize_to_sequence(sequence):
        ''' Normalize the sequence with respect to the maximum and minimum of the whole sequence.
            Args: 
                sequence (2-d tensor): sequence to be cropped.
            Returns:
                normalized sequence (2-d tensor) on a sequence basis.
        '''
        max_value = torch.max(sequence)
        min_value = torch.min(sequence)
        sequence = (norm_interval[1] - norm_interval[0]) \
                            *(sequence - min_value)/(max_value - min_value) + norm_interval[0]

        return sequence

    def normalize_to_frame(sequence):
        ''' Normalize the sequence to the provided member interval array for each frame individually.
            Note: in initial tests this caused the lstm output to return NaNs, so currently not recommended.
            Args: 
                sequence (2-d tensor): sequence to be cropped.
            Returns:
                normalized sequence (2-d tensor) on a frame-by-frame basis.
        '''
        max_values = torch.max(sequence,1).values # normalize across the feature dimension ([seq_length, num_features_per_transform]).
        min_values = torch.min(sequence,1).values # -"-
        sequence = torch.sub(sequence, min_values[:,None])
        sequence = torch.div(sequence, (max_values - min_values)[:,None])
        norm_factor = (self.upper_norm_interval - self.lower_norm_interval)
        sequence = norm_factor*sequence + self.lower_norm_interval
        return sequence

    def no_normalization(sequence):
        pass

    if norm_type == 'sequence':
        return normalize_to_sequence
    elif norm_type == 'frame':
        return normalize_to_frame
    elif norm_type is None:
        return no_normalization
    else:
        raise ValueError("norm_type needs either \'frame\', \'sequence\' or None")


def _get_ltas_subtraction_function(spectrum_index, ltas_list):
    '''Parameterize the ltas subtraction function.
    Args:
        ltas_list (list(1-d tensor)): List of the LTAS for each measured spectrum ([S12, S32])
        spectrum_index (int): spectrum index of the [S12, S32] ltas list.
    '''
    def subtract_ltas(sequence):
        seq_length, num_freqs = sequence.shape
        return sequence - ltas_list[spectrum_index][:num_freqs]

    return subtract_ltas


class ParameterizedTransformFunction():
    ''' Class to parameterize the transform function passed to the dataset object.

    Args:
        spectrum_keys (list(str)): list of strings specifying which spectrum_keys to use and in which order.
        transform_keys (list(str)): list of strings specifying the transform_keys applied to each spectrum 
            in the spectrum_keys list.
        feature_seq_lens (1-d array): Length's of every distinct feature sequence (e.g., magnitude, delta magnitude etc.)
        norm_intervals (list([int,int])): Specify whether to normalize every transformed spectrum. E.g.: [[-1,1],[0,1]]
        norm_type (str): Normalize each sequence 'frame' wise or 'sequence' wise. Note: 'frame' wise is currently not recommended.
        ltas_list (list(1-d tensor)): magnitude LTAS for all measured spectra (currently S12, S32)
        use_ltas_subtraction (bool): Decide whether to use LTAS subtraction.
    '''

    def __init__(self, spectrum_keys, transform_keys, num_features_per_transform, norm_intervals, norm_type, ltas_list, use_ltas_subtraction):
        self.spectrum_keys = spectrum_keys
        self.num_channels = len(spectrum_keys)
        self.spectrum_indices = [] # Maps: S12 -> 0, S32 -> 1
        self.transform_keys = transform_keys
        self.norm_intervals = norm_intervals
        self.norm_type = norm_type
        self.ltas_list = ltas_list
        self.ltas_list_db = []
        self.num_features_per_transform = num_features_per_transform

        for (index, ltas) in enumerate(self.ltas_list):
            if ltas is not None:
                ltas = 20*torch.log10(ltas)
                ltas[ltas < SPECTRAL_MAG_MIN_VALUE_DB] = SPECTRAL_MAG_MIN_VALUE_DB
                self.ltas_list_db.append(ltas)
            else:
                self.ltas_list_db.append(None)

        self.use_ltas_subtraction = use_ltas_subtraction    
        self.allowed_spectrum_keywords = ['S12', 'S32']
        self.allowed_transform_options = ['mag','mag_db','phase','mag_delta','mag_db_delta','phase_delta','impulse_response','impulse_response_delta']
        self.transform_pipeline = []

        self.num_pairs = 0 # Number of [spectrum_keys,transform]-pairs passed to the transform function.

        if(len(self.spectrum_keys) != len(self.transform_keys)):
            raise ValueError("Error: length of spectrum keys needs to be equal to length of transform keys.")

        # Check the entered spectrum keys.
        for keyword in self.spectrum_keys:
            is_key_valid_list = [keyword == allowed_keyword for allowed_keyword in self.allowed_spectrum_keywords]
            if not(any(is_key_valid_list)):
                raise ValueError("Error: allowed keys in spectrum_keys_and_transform_keys are: %s. Passed: %s" % (self.allowed_spectrum_keywords, keyword))
            # Get the numeric index of the spectrum keyword. S12: 0, S32: 1
            self.spectrum_indices.append([index for index, key in enumerate(self.allowed_spectrum_keywords) if (key == keyword)][0])
  
        for keyword in self.transform_keys:
            is_transform_valid_list = [keyword == allowed_transform for allowed_transform in self.allowed_transform_options]
            if not(any(is_transform_valid_list)):
                raise ValueError("Error: allowed keys in spectrum_keys_and_transform_keys are: %s. Passed: %s" % (self.allowed_transform_options, keyword))
        
        self.num_pairs = len(self.spectrum_keys)

        # Check the number of normalization intervals.
        if(len(self.norm_intervals) is not self.num_pairs):
            raise ValueError("Error: %d normalization intervals provided for %d transform pairs." % (len(self.norm_intervals), self.num_pairs))

        self.transform_pipeline = self._get_transform_pipeline()


    def transform(self, sequence_list):
        ''' Transforms the selected spectrograms according with the initialized transform pipeline.
            Args:
                sequence_list (list(2-d tensor)): a list of all possible measured spectrograms
                    (S12, S32) for a single sample.
                reshape_sequence_to_2d (bool): flatten the feature sequence from a 3-d tensor to a 2-d tensor along the time axis.
            Returns:
                feature vector (3-d tensor): stack of feature sequences [num_channels, seq_len, num_features_per_transform]
        '''
        num_frames, _ = sequence_list[0].shape
        # feature_sequence = torch.empty(num_frames, self.num_features_per_transform)
        feature_seq_len = self.num_features_per_transform 
        feature_sequences = torch.empty(self.num_channels, num_frames, feature_seq_len)
        
        for (seq_index, spectrum_index) in enumerate(self.spectrum_indices):
            sequence = sequence_list[spectrum_index]
            # Transform each sequence with its individual transform pipeline.
            for transform_fcn in self.transform_pipeline[seq_index]:
                # print("Transform: %s" % transform_fcn.__name__)
                sequence = transform_fcn(sequence)

            feature_sequences[seq_index,:,:] = sequence
            
        
        '''
            start_index = 0
            stop_index = 0
            # If selected, flatten the feature sequence (for lstms, e.g.).
            feature_sequences = feature_sequences.permute(1,0,2).reshape(num_frames,-1)
            plt.plot(feature_sequences.T)
            plt.show()
        '''

        return feature_sequences


    def _get_transform_pipeline(self):
        ''' Initialize the transform function with the provided transform keyword list. 
        Args:
            None.
        Returns:
            transform_pipeline (list(list(fcns)): a list of transform functions for each sequence that is to be transformed.
        Note:
            All transforms use the spectrum cropped to [0:num_features_per_transform], even the impulse
            response, which could have a total length of num_freqs*2-2. E.g., a feature size of 84 from a spectrum of 128 
            frequency points yields an impulse response of length 84 ([85:170]), centered around index 127. This is to
            make it compatible with other features (mag, phase) and the CNN-LSTM architecture, which needs equal feature
            sizes on all channels.
        '''
        transform_pipeline = []

        # Pick the transform_keys according to the key list and append the corr. transform.
        for (index, key) in enumerate(self.transform_keys):
            transform_list = []
            if key == 'mag':
                transform_list.append(calc_magnitude)
                if(self.use_ltas_subtraction):
                    transform_list.append(_get_ltas_subtraction_function(self.spectrum_indices[index], self.ltas_list))
            elif key == 'mag_db':
                transform_list.append(calc_magnitude_db)
                if(self.use_ltas_subtraction):
                    transform_list.append(_get_ltas_subtraction_function(self.spectrum_indices[index], self.ltas_list_db))
            elif key == 'mag_delta':
                transform_list.append(calc_delta_magnitude)
            elif key == 'mag_db_delta':
                transform_list.append(calc_delta_magnitude_db)
            elif key == 'phase':
                transform_list.append(calc_phase_angle)
            elif key == 'phase_delta':
                transform_list.append(calc_delta_phase)
            elif key == 'impulse_response':
                transform_list.append(_get_impulse_response_function(self.num_features_per_transform))
            elif key == 'impulse_response_delta':
                transform_list.append(_get_calc_delta_impulse_response_function(self.num_features_per_transform))
            else:
                raise ValueError("Error: transform %s is not implemented." % (key))
            # Crop the sequence.
            transform_list.append(_get_crop_function(self.num_features_per_transform))
            # Normalize.
            transform_list.append(_get_normalization_function(self.norm_type, self.norm_intervals[index]))

            transform_pipeline.append(transform_list)

        return transform_pipeline