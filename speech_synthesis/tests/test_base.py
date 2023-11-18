from src.datasets import rs_synthesis_dataset
import torch
import pytest
import itertools
import os
from tests import fixtures
from src.models import RsVocoderSynthesisLstmModel
from src.models import CnnRnnModel
from src.training import train_and_evaluate
from src.training import transforms

### Input #########################

data_file_path = os.path.join(os.getcwd(), "tests\\test_data") # For pytest from console: pytest tests/test_base.py

###################################

# Test the dataset.
def test_create_data_set():

    # Test, whether the dataset object is correctly populated with the file names
    # and locations, both when loaded from the files list and without.
    # Note: for HPC usage, only the files list is safe atm!

    all_vocoder_params_and_dims = fixtures.return_all_vocoder_parameter_and_dims()
    all_vocoder_parameter_names = list(all_vocoder_params_and_dims.keys())

    list_permutations = itertools.permutations(all_vocoder_parameter_names)

    for is_load_file_path_from_list in [True, False]:
        for vocoder_param_names in list_permutations:
            print(vocoder_param_names)
            dataset = rs_synthesis_dataset.RsSynthesisDataset(data_file_path,
                                                              is_load_file_path_from_list=is_load_file_path_from_list,
                                                              input_parameter_list=vocoder_param_names,
                                                              fixed_lag_frames=-10)

            # Load the file name list.
            file_names = fixtures.return_test_data_file_names()
            num_files = len(file_names['radar_files'])

            for file_index in range(num_files):
                # Check the stored radar file names.
                file_path = os.path.join(data_file_path, "radar_files", file_names['radar_files'][file_index])
                file_name = file_path.split("\\")[-1]
                assert(dataset.radar_sequence_file_names_list[file_index] == file_name)
                # Check the stored parameter names.
                for vocoder_param_name in vocoder_param_names:
                    file_path = os.path.join(data_file_path, 
                                             "input_params_files", 
                                             vocoder_param_name, # sub-folder name
                                             file_names['input_params_files'][vocoder_param_name][file_index])
                    dataset_file_path = dataset.input_param_file_names_list[vocoder_param_name][file_index]
                    assert(dataset_file_path == file_path)


def test_fixed_lag_frame_alignment():
            # Test the offset between the freature sequence and the target sequence for
            # fixed_lag_frames < 0.
            is_load_file_path_from_list = True
            all_vocoder_params = fixtures.return_all_vocoder_parameter_and_dims()
            vocoder_param_names = [list(all_vocoder_params.keys())[0]] # any is ok, since only the sequence length is relevant, which is the same for all of them.
            fixed_lag_frames = -5

            non_shifted_dataset = rs_synthesis_dataset.RsSynthesisDataset(data_file_path,
                                                                          is_load_file_path_from_list=is_load_file_path_from_list,
                                                                          input_parameter_list=vocoder_param_names,
                                                                          fixed_lag_frames=0)

            shifted_dataset = rs_synthesis_dataset.RsSynthesisDataset(data_file_path,
                                                                    is_load_file_path_from_list=is_load_file_path_from_list,
                                                                    input_parameter_list=vocoder_param_names,
                                                                    fixed_lag_frames=fixed_lag_frames)

            # Get the original sequence length for each training item.
            sequence_lengths = []
            feature_sequences = []
            for item_index, data in enumerate(non_shifted_dataset):
                feature_sequence, vocoder_params = data
                feature_sequences.append(feature_sequence)
                sequence_lengths.append(feature_sequence.shape[0])

            # Get the sequence lengths of the shifted sequences. These need to be
            # |fixed_lag_frames| shorter.
            for item_index, data in enumerate(shifted_dataset):
                shifted_feature_sequence, shifted_vocoder_params = data
                shifted_sequence_length = shifted_feature_sequence.shape[0]

                # Check for length.
                assert(shifted_sequence_length == sequence_lengths[item_index]+fixed_lag_frames)
                # Check for correct shift direction. Shift has to be to the left for the feature sequence.

                for frame_index in range(shifted_sequence_length):
                    # Sum over the frame should be equal.
                    assert sum(shifted_feature_sequence[frame_index,:]) == sum(feature_sequences[item_index][frame_index-fixed_lag_frames,:]) 


def test_batch_sequences():

    test_sequence_length = torch.tensor([10,50,66,124,169]) # TODO: raise exception with < 10
    input_size = 67
    output_size = 35
    batch_size = 32
    num_channels = 3
    time_context_frames = 10
    batch_sizes = [torch.tensor([1]),
                   torch.tensor([32,9]),
                   torch.tensor([32,25]),
                   torch.tensor([32,32,32,19]),
                   torch.tensor([32,32,32,32,32])]

    # Create a few test sequences.
    for seq_index, test_length in enumerate(test_sequence_length):
        input_sequence = torch.randn((1, num_channels, test_length, input_size), dtype=torch.float32)
        output_sequence = torch.empty((1, test_length, input_size), dtype=torch.float32)
        for n in range(test_length):
            output_sequence[:,n,:] = torch.ones((1,1,input_size), dtype=torch.float32)*n
        
        batched_data = rs_synthesis_dataset.BatchData(input_sequence=input_sequence,
                                                      target_sequence=output_sequence,
                                                      batch_size=batch_size,
                                                      time_context_frames=time_context_frames)
        
        # Test the batching which should return the input/target pairs.
        # Example for time_context_frames=5 : 0:4 -> 4, 1:5 -> 5, 2:6 -> 6 etc.
        target_frame_index = time_context_frames - 1 # start index for the target frame.
        for batch_index, data in enumerate(batched_data):
            X,Y = data
            # Check the correct batch and residual batch sizes.
            assert(X.shape[0] == batch_sizes[seq_index][batch_index])

            # Check if the targets are correctly aligned with the corresp. sequences in the batch.
            # X has shape [batch_size, time_context_frames, output_size]
            # Y has shape [batch_size, 1, output_size]
            for n in range(X.shape[0]):
                y = Y[0,n,:]
                target = output_sequence[0,target_frame_index,:] # -1: index 9 is the 10th frame etc.
                target_frame_index += 1
                assert sum(target) == sum(y)

        
def test_transform_pipeline_function_selection():

    # Test with ltas subtraction.
    use_ltas_subtraction = True
    transform_keys = ['mag','mag_db','phase','mag_delta','mag_db_delta','phase_delta']
    transform_fcn_names = ['calc_magnitude','calc_magnitude_db','calc_phase_angle', \
                           'calc_delta_magnitude','calc_delta_magnitude_db','calc_delta_phase']
    spectrum_keys = ['S12','S32','S32','S12','S32','S32']
    freq_interval = [0,40]
    num_features_per_transform = freq_interval[1] - freq_interval[0]
    norm_intervals = [[0,1],[0,1],[-1,1],[-1,1],[0,1],[0,1]]
    norm_type = 'sequence'
    # Create an artificial ltas.
    ltas_s12 = torch.abs(torch.randn((1,128),dtype=torch.float32))
    ltas_s32 = torch.abs(torch.randn((1,128),dtype=torch.float32))
    ltas_list = [ltas_s12, ltas_s32]

    transform_obj = transforms.ParameterizedTransformFunction(spectrum_keys=spectrum_keys,
                                                            transform_keys=transform_keys,
                                                            num_features_per_transform=num_features_per_transform,
                                                            norm_intervals=norm_intervals,
                                                            norm_type=norm_type,
                                                            ltas_list=ltas_list,
                                                            use_ltas_subtraction=use_ltas_subtraction)
    transform_pipeline = transform_obj.transform_pipeline
    # Create the target function pipeline(s) for each transform. (Ltas subtr.), crop sequence and normalization
    # is done for almost every sequence, as such, they are appended to each function name target list.
    target_fcn_names = []
    target_fcn_names.append(["calc_magnitude", "subtract_ltas", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_magnitude_db", "subtract_ltas", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_phase_angle", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_magnitude", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_magnitude_db", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_phase", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_impulse_response", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_impulse_response_delta", "crop_sequence", "normalize_to_sequence"])

    for (spectrum_index, transform) in enumerate(transform_pipeline):
        for (transform_index, t) in enumerate(transform):
            fcn_name = t.__name__
            target_fcn_name = target_fcn_names[spectrum_index][transform_index]
            assert(fcn_name == target_fcn_name)

    # test without ltas subtraction.
    use_ltas_subtraction = False
    transform_keys = ['mag','mag_db','phase','mag_delta','mag_db_delta','phase_delta']
    transform_fcn_names = ['calc_magnitude','calc_magnitude_db','calc_phase_angle', \
                           'calc_delta_magnitude','calc_delta_magnitude_db','calc_delta_phase']
    spectrum_keys = ['S12','S32','S32','S12','S32','S32']
    freq_interval = [0,30]
    norm_intervals = [[0,1],[0,1],[-1,1],[-1,1],[0,1],[0,1]]
    norm_type = 'sequence'
    ltas_list = [None, None]

    transform_obj = transforms.ParameterizedTransformFunction(spectrum_keys=spectrum_keys,
                                                            transform_keys=transform_keys,
                                                            num_features_per_transform=num_features_per_transform,
                                                            norm_intervals=norm_intervals,
                                                            norm_type=norm_type,
                                                            ltas_list=ltas_list,
                                                            use_ltas_subtraction=use_ltas_subtraction)
    transform_pipeline = transform_obj.transform_pipeline
    target_fcn_names = []
    target_fcn_names.append(["calc_magnitude", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_magnitude_db", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_phase_angle", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_magnitude", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_magnitude_db", "crop_sequence", "normalize_to_sequence"])
    target_fcn_names.append(["calc_delta_phase", "crop_sequence", "normalize_to_sequence"])

    for (spectrum_index, transform) in enumerate(transform_pipeline):
        for (transform_index, t) in enumerate(transform):
            fcn_name = t.__name__
            target_fcn_name = target_fcn_names[spectrum_index][transform_index]
            assert(fcn_name == target_fcn_name)


def test_impulse_response_function():
    ''' Test the impulse response function separately.
    '''

    seq_length = 300
    num_freqs = 128
    imp_resp_win_lens = torch.linspace(0,num_freqs-1,1,dtype=torch.long)
    test_sequence = torch.randn((seq_length, num_freqs),dtype=torch.float32)

    # Test the acceptable range.
    for imp_resp_win_len in imp_resp_win_lens:

        resp_len = imp_resp_win_len.item()
        if(resp_len % 2 != 0):
            with pytest.raises(ValueError):
                calc_impulse_response_fcn = transforms._get_impulse_response_function(imp_resp_win_len.item())
        else:
            calc_impulse_response_fcn = transforms._get_impulse_response_function(imp_resp_win_len.item())
            imp_response = calc_impulse_response_fcn(test_sequence)
            assert(imp_response.shape[1] == imp_resp_win_len)
    
    # Test out of range impulse response lengths.
    imp_resp_win_len = -1 
    # Correctly catch Value error for lens < 0.
    with pytest.raises(ValueError):
        calc_impulse_response_fcn = transforms._get_impulse_response_function(imp_resp_win_len)

    # Correctly catch Value errors for too large even impulse response lens.
    imp_resp_win_lens = [num_freqs,num_freqs+2,num_freqs+4,num_freqs+6]

    for imp_resp_win_len in imp_resp_win_lens:
        calc_impulse_response_fcn = transforms._get_impulse_response_function(imp_resp_win_len)

        with pytest.raises(ValueError):
            calc_impulse_response_fcn = calc_impulse_response_fcn(test_sequence)



def test_feature_sequence_size():
    # Test all transforms to check that they output the correct sequence shapes.

    use_ltas_subtraction = True
    transform_keys = ['mag','mag_db','phase','mag_delta','mag_db_delta','phase_delta','impulse_response','impulse_response_delta']
    spectrum_keys = ['S12','S32','S32','S12','S32','S32','S12','S32']

    num_steps = 128 # number of measured frequency points in the spectrum.
    freq_interval = [0,30]
    norm_intervals = [[0,1],[0,1],[-1,1],[-1,1],[0,1],[0,1],[0,1],[-1,1]]
    test_seq_len = 10
    norm_type = 'sequence'

    # Test each transform for its correct input/output shape.

    for n,_ in enumerate(transform_keys):

        num_channels = 1 # test each transform one at a time.

        # Create an artificial ltas.
        num_features_per_transform = freq_interval[1] - freq_interval[0]
        ltas_s12 = torch.abs(torch.randn((num_steps),dtype=torch.float32))
        ltas_s32 = torch.abs(torch.randn((num_steps),dtype=torch.float32))
        ltas_list = [ltas_s12, ltas_s32]

        transform_obj = transforms.ParameterizedTransformFunction(spectrum_keys=[spectrum_keys[n]], # argument expects list
                                                                transform_keys=[transform_keys[n]], # argument expects list
                                                                num_features_per_transform=num_features_per_transform,
                                                                norm_intervals=[norm_intervals[n]],
                                                                norm_type=norm_type,
                                                                ltas_list=ltas_list,
                                                                use_ltas_subtraction=use_ltas_subtraction)

        test_sequences = [torch.randn((test_seq_len,num_steps),dtype=torch.float32),    
                          torch.randn((test_seq_len,num_steps),dtype=torch.float32)]

        target_num_features_per_transform = (freq_interval[1] - freq_interval[0])

        transformed_sequence = transform_obj.transform(test_sequences)
        assert(transformed_sequence.shape[0] == num_channels)
        assert(transformed_sequence.shape[1] == test_seq_len)
        assert(transformed_sequence.shape[2] == target_num_features_per_transform)



def test_cnn_output_shape():
    
    # Initial test with the dimensions from the paper.
    in_channels = [4,16,32,64,128]
    out_channels = in_channels[1:] + [256]
    batch_size = 1
    seq_len = 500
    num_features_per_transform = 161
    hyperparameters = {'in_channels' : in_channels,
                       'out_channels' : out_channels}

    cnn_model = CnnRnnModel.CNN(num_features_per_transform,
                                hyperparameters['in_channels'],
                                hyperparameters['out_channels'])

    x = torch.randn((batch_size, in_channels[0],seq_len,num_features_per_transform))
    out_shape = cnn_model.calc_cnn_output_shape(in_channels[0],seq_len,num_features_per_transform)
    assert(out_shape[0] == 1)
    assert(out_shape[1] == out_channels[-1])
    assert(out_shape[2] == seq_len)
    assert(out_shape[3] == 4)

    # Test arbitrary in_ch/out_ch combination.
    in_channels = [4,11,22,45,112]
    out_channels = in_channels[1:] + [233]
    num_features_per_transform = 255

    hyperparameters = {'in_channels' : in_channels,
                       'out_channels' : out_channels}

    cnn_model = CnnRnnModel.CNN(num_features_per_transform,
                                hyperparameters['in_channels'],
                                hyperparameters['out_channels'])

    x = torch.randn((batch_size, in_channels[0],seq_len,num_features_per_transform))
    out_shape = cnn_model.calc_cnn_output_shape(in_channels[0],seq_len,num_features_per_transform)
    assert(out_shape[0] == 1)
    assert(out_shape[1] == out_channels[-1])
    assert(out_shape[2] == seq_len)
    assert(out_shape[3] == 7)


def test_cnn_lstm_output_shape():

    in_channels = [4,16,32,64,128]
    out_channels = in_channels[1:] + [256]
    batch_size = 16
    seq_len = 500
    num_features_per_transform = 255
    lstm_hidden_size = 50
    output_size = 80
    hyperparameters = {'num_features_per_transform' : num_features_per_transform,
                       'in_channels' : in_channels,
                       'out_channels' : out_channels,
                       'lstm_num_layers' : 1,
                       'lstm_hidden_size' : lstm_hidden_size,
                       'output_size' : output_size,
                       'batch_size' : batch_size}

    model = CnnRnnModel.CRN(hyperparameters)
    x = torch.randn((batch_size, in_channels[0],seq_len,num_features_per_transform))
    out = model(x)
    print(out.shape)
    assert(out.shape[0] == batch_size)
    assert(out.shape[1] == output_size)


def test_reshape_feature_sequences():
    batch_size = 4
    in_channels = 3
    seq_len = 10
    num_features_per_transform = 5

    input_sequences = torch.empty(in_channels, seq_len, num_features_per_transform)
    for n in range(in_channels):
        input_sequences[n] = torch.randn((1, seq_len, num_features_per_transform))




# Run tests.
if __name__ == "__main__":
    test_create_data_set()
    test_batch_sequences()
    test_fixed_lag_frame_alignment()
    test_feature_sequence_size()
    test_transform_pipeline_function_selection()
    test_impulse_response_function()
    test_feature_sequence_size()
    test_cnn_output_shape()
    test_cnn_lstm_output_shape()