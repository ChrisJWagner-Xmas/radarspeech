import torch
import pandas as pd
import os
import sys
import glob
import numpy as np
from src.fileio.import_rs_data import import_rs_binary_file
import matplotlib.pyplot as plt

class RsSynthesisDataset(torch.utils.data.Dataset):
    '''
    Dataset class for the radarspeech synthesis corpus.
    Inherits from the torch Dataset class.
   '''

    def __init__(self, 
                 absolute_data_files_path, 
                 is_load_file_path_from_list,
                 input_parameter_list,
                 fixed_lag_frames,
                 transform_fn=None):
        ''' 
        Constructor.

        Args:
            absolute_data_files_path (str): path where the radar, audio and target seq. files are found.
                                         The folder names need to be named "radar_files", "audio_files" and "input_params_files".
            is_load_file_path_from_list (bool): option to load the file names from a predefined .csv list (if sorted() does not work e.g.).
            input_parameter list(str): list of strings (can be a single entry) of which input params to load.
                                         Must match the folder name of the corresponding input parameter.
            fixed_lag_frames (int): specify the frame shift by which the radar frames lead or lag the predicted input parameters.
                               Negative values cause the radar frames to lead ("look into future"). Only negative values are allowed.
            transform_fn (function): function to transform each dataset item before returning it.

        Returns:
            None.

        '''

        # File path list for the radar, audio and input sequences.
        if not os.path.isdir(absolute_data_files_path):
            raise IOError(f"The provided path {absolute_data_files_path} does not exist.")

        if(fixed_lag_frames > 0):
            raise ValueError("Error: fixed_lag_frames has to be <= 0, only future look-ahead is allowed.")

        self.absolute_data_files_path = absolute_data_files_path
        self.radar_sequence_file_names_list = []
        self.radar_sequence_full_file_names_list = []
        self.input_parameter_list = input_parameter_list
        self.input_param_file_names_list = {} # dict() of the file locations for the input parameters stored in this dataset.
        self.transform_fn = transform_fn
        self.fixed_lag_frames = fixed_lag_frames

        # Load the radar sequences ++++++++++++++++++++++++++++++++++
        # Using glob:
        if(is_load_file_path_from_list):
            # Load the file names from the .csv list.
            full_file_path_stump = os.path.join(self.absolute_data_files_path, "radar_files", "radar*")
            full_file_path = glob.glob(full_file_path_stump)
            if not full_file_path:
                raise IOError(f"No radar files list found at {full_file_path_stump}")

            full_file_path = full_file_path[0]
            df = pd.read_csv(full_file_path, header=None)
            self.radar_sequence_file_names_list = df.iloc[:,0].values.tolist() # get the first column. Save the file names
            self.radar_sequence_full_file_names_list = df.iloc[:,0].values.tolist() # And also save the full file names (with path)
            # Add the full file path back to each file name.
            for file_index in range(len(self.radar_sequence_full_file_names_list)):
                self.radar_sequence_full_file_names_list[file_index] = os.path.join(self.absolute_data_files_path,
                                                                               "radar_files",
                                                                               self.radar_sequence_full_file_names_list[file_index])
        else:
            # Load the file names by looking up the directory content.
            full_file_path = os.path.join(self.absolute_data_files_path, "radar_files", "S0*")
            self.radar_sequence_full_file_names_list = sorted(glob.glob(full_file_path), key=len)

        if not self.radar_sequence_full_file_names_list:
            raise IOError(f"Could not load radar files from {full_file_path}")
        
        # Load the LTAS from the same path as the radar data.
        ltas_S12_file_path = os.path.join(self.absolute_data_files_path, "radar_files", "ltas_S12.csv")
        ltas_S32_file_path = os.path.join(self.absolute_data_files_path, "radar_files", "ltas_S32.csv")
        try:
            ltas_s12 = pd.read_csv(ltas_S12_file_path, 
                                    header=None, 
                                    dtype=np.float32).values.squeeze(axis=1) 
            if not torch.is_tensor(ltas_s12):
                ltas_s12 = torch.from_numpy(ltas_s12)

        except OSError:
            print(f"Warning: no S32 LTAS found at file path {self.absolute_data_files_path}. Will be assigned None")
            ltas_s12 = None

        try:
            ltas_s32 = pd.read_csv(ltas_S32_file_path, 
                                   header=None, 
                                   dtype=np.float32).values.squeeze(axis=1) 
            if not torch.is_tensor(ltas_s32):
                ltas_s32 = torch.from_numpy(ltas_s32)

        except OSError:
            print(f"Warning: no S32 LTAS found at file path {self.absolute_data_files_path}. Will be assigned None")
            ltas_s32 = None

        self.ltas_list = [ltas_s12, ltas_s32]

        # Load the input parameters. +++++++++++++++++++++++++++++++++++++++++
        # Store the different input parameter lists in a dictionary of lists.
        input_param_file_list_lens = []

        for input_param in input_parameter_list:
            # Using glob:
            if(is_load_file_path_from_list):
                # Load the file names from the .csv list.
                full_file_path_stump = os.path.join(self.absolute_data_files_path, "input_params_files", input_param, input_param + "*")
                full_file_path = glob.glob(full_file_path_stump)
                if not full_file_path:
                    raise IOError(f"No files list found at {full_file_path} for parameter {input_param}")

                full_file_path = glob.glob(full_file_path_stump)[0]
                df = pd.read_csv(full_file_path, header=None)
                self.input_param_file_names_list[input_param] = df.iloc[:,0].values.tolist()  # get the first column.
                # Add the full file path back to each file name.
                for file_index in range(len(self.radar_sequence_full_file_names_list)):
                    self.input_param_file_names_list[input_param][file_index] = os.path.join(self.absolute_data_files_path,
                                                                                   "input_params_files",
                                                                                   input_param,
                                                                                   self.input_param_file_names_list[input_param][file_index])
            else:
                full_file_path = os.path.join(self.absolute_data_files_path, "input_params_files", input_param, "S0*")
                self.input_param_file_names_list[input_param] = sorted(glob.glob(full_file_path), key=len)

            # save the list length's for later error checking.
            input_param_file_list_lens.append(len(self.input_param_file_names_list[input_param])) 
            if not self.input_param_file_names_list[input_param]:
                raise IOError("Error: could not locate the necessary input param %s files at path %s" %(input_param, full_file_path))
        
        # Check, if all input parameter have the same number of files.
        input_param_file_list_lens = torch.LongTensor(input_param_file_list_lens)
        num_input_param_files = torch.unique(input_param_file_list_lens)
        if(num_input_param_files.shape[0] != 1):
            raise ValueError("Error: not all input parameter file lists have the same length. Are there missing files?")

        # Also make sure all three lists (radar, audio, input parameters) have the same number of files.
        file_list_lens = torch.tensor([len(self.radar_sequence_full_file_names_list),
                          num_input_param_files.item()])
     
        unique_lens = torch.unique(file_list_lens)
        if(unique_lens.shape[0] != 1):
            raise ValueError("Error: Radar sequence and input file lists are of unequal length." \
            + "All three need to have identical length. Please check the data folder.")


        # Check the sorting of the files by comparing the sentence number.
        num_files = unique_lens[0].item()

        for file_index in range(num_files):
            radar_file_name = self.radar_sequence_full_file_names_list[file_index]
            radar_file_segments = radar_file_name.split(os.sep)[-1]
            radar_file_segments = radar_file_segments.split("_")
            radar_file_sentence_num = radar_file_segments[-1].split('.')[0]

            for input_param in self.input_parameter_list:
                input_param_file_name = self.input_param_file_names_list[input_param][file_index]
                input_param_file_segments = input_param_file_name.split(os.sep)[-1]
                input_param_file_segments = input_param_file_segments.split("_")
                input_param_file_sentence_num = input_param_file_segments[2].split('.')[0]
                    
                if(radar_file_sentence_num != input_param_file_sentence_num):
                    raise ValueError("Error: radar file %s ends on %s, but %s input file %s ends on %s" % (radar_file_name,
                                                                                                     radar_file_sentence_num, 
                                                                                                     input_param, 
                                                                                                     input_param_file_name,
                                                                                                     input_param_file_sentence_num))

    def __len__(self):
        ''' Overridden function that returns the length of the {audio, target} set.

        Args:
            None.

        Returns:
            length (int): number of sequences in the set.
        '''

        return len(self.radar_sequence_full_file_names_list)


    def __getitem__(self, index):
        ''' Overridden function to pick a sample from the data set.
            
        Args:
            index (int): sample index from the dataset.

        Returns:
            radar_sequence (2d-tensor, float32): the radar sequence of the sentence of phrase at the file index.
            audio_samples (1d-tensor, float32): audio samples of the aligned audio signal for the sentence or phrase.
            input_params (2d-tensor, float32): input params of the aligned input params for the sentence or phrase.

        '''

        if(torch.is_tensor(index)):
            index = index.to_list()

        # Load the indexed radar sequence file into a torch tensor.
        rs_data = import_rs_binary_file(self.radar_sequence_full_file_names_list[index])
        S12 = torch.from_numpy(rs_data.radargrams[1]) # cheek-chin path
        S32 = torch.from_numpy(rs_data.radargrams[7]) # cheek-cheek path

        # Read the input param values into a torch tensor.
        input_param_name = self.input_parameter_list[0]
        input_params = pd.read_csv(self.input_param_file_names_list[input_param_name][index], 
                                         header=None, 
                                         dtype=np.float32).values # Shape needs to be [input_params, num_frames]
        input_params = torch.from_numpy(input_params)
        
        # If more than one parameter should be loaded, concatenate them along the first (row) dimension.
        for input_param_index in range(1, len(self.input_parameter_list)):
            input_param_name = self.input_parameter_list[input_param_index]
            input_params_temp = pd.read_csv(self.input_param_file_names_list[input_param_name][index], 
                                           header=None, 
                                         dtype=np.float32).values
            input_params_temp = torch.from_numpy(input_params_temp)
            input_params = torch.cat((input_params, 
                                        input_params_temp), 
                                       dim=1)

        # Transform the radar-spectrograms if needed and concatenate them along the feature dimension.
        if self.transform_fn is not None:
            feature_sequence = self.transform_fn([S12,S32])
        else:
            feature_sequence = torch.cat((S12,S32),dim=1)

        # Add a fixed lead or lag to the radar frame sequence if specified. The shift is applied
        # to the first dimension/rows (time axis) of the 2-d sequence.
        if(self.fixed_lag_frames < 0):
            feature_sequence = torch.roll(feature_sequence, 
                                            shifts=self.fixed_lag_frames, 
                                            dims=1) # along the frame/time axis
            feature_sequence = feature_sequence[:,:self.fixed_lag_frames,:] # Crop the tail of the "looked-ahead" sequence
            input_params = input_params[:self.fixed_lag_frames,:]     # as it is not meaningful anymore.

        return (feature_sequence, input_params) # input_params are the target vectors (mel spectrograms e.g.)


def pad_seqs_and_collate(batch):
    ''' custom collate (="zuordnen") function for the data_loader object. This function can be overridden
        when passed to the collate_fn argument when creating the data_loader object.
        The data loader fetches a list of samples, which can then be used to access the individual samples
        for their, e.g., (sequence, label) pairs and further processed (padded etc.).
        
    Args:
        batch (tuple): in this case a [batch_size, 2] tuple containing the (sequence, label) tuple for each 
                entry in the batch. Passed in by the data loader object. Content is:
                input sequence: [2-d tensor (float64/float32)] of dimension [num_frames, input_size] 
                target sequence (input params): [2-d tensor (int64/long)] of dim. [num_frames, output_size]

    Returns:
        padded_input_sequence_stack (3-d tensor (float32)): The padded sequences per batch 
                                        with dimensions [batch_size, max_seq_length, input_size]
        padded_output_sequence_stack (3-d tensor (float32)): Target sequence for the regression task with
                                        dimensions [batch_size, max_seq_length, output_size] 
        unpadded_sequence_lengths (1-d tensor (int64/long)): The actual sequence lengths for every sequence
                                        in the batch (scalars). Has length [batch_size].
    ''' 

    input_sequences, target_sequences = zip(*batch) # unzip the batch-tuple into sequences and their target values.
    
    # convert tuple to list
    input_sequences = list(input_sequences) 
    target_sequences = list(target_sequences) 
   
    # Pad the sequences out to a 3d-stack using the built-in pytorch functionality.
    # Note: currently has no effect, because all sequences have the same length.
    padded_input_sequence_stack = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first=True) # actually only adds the batch_size dim, which could be done with .squeeze as well.
    padded_target_sequence_stack = torch.nn.utils.rnn.pad_sequence(target_sequences, batch_first=True)
    
    return padded_input_sequence_stack, padded_target_sequence_stack


class BatchData(object):

    def __init__(self, input_sequence, target_sequence, batch_size, time_context_frames):
        '''
           Iterable object to create the batched input/output sequence pairs.
           Args:
                input_sequence (4-d torch tensor): input sequence with dimension
                                                   [1, num_channels, sequence_length, input_size]                  
                output_sequence (3-d torch tensor): output sequence with dimension
                                                   [1, sequence_length, output_size]  
                batch_size (int): batch size that will be returned when iterated over.
                time_context_frames (int): length of each of the batched sequences, created from
                                            the single long input/target sequence.

            Returns:
                _batched_input_sequence (4-d torch tensor): batched input sequences for training with dim
                                                        [batch_size, num_channels, time_context_frames, num_features_per_transform]
                _batched_targets (3-d torch tensor): batched target frames for each sequence with dim
                                                     [batch_size, time_context_frames, num_features_per_transform]
            ToDo:
                Add fixed-lag here by adjusting the target frame in _batched_targets    

        '''
        self._batch_size = batch_size
        if(batch_size < 1):
            raise ValueError("Batch size needs to be > 0")

        self._time_context_frames = time_context_frames
        self._start_index = 0
        self._stop_index = 0
        self._num_input_sequences = input_sequence.shape[0] 
        if(self._num_input_sequences > 1):
            raise ValueError("Error: the number of sequences for batching is restricted to 1 for now, as"
                             + "concatenating sequences (i.e., sentences) could lead to context jumps across sentence"
                             + "boundaries.")

        self._seq_batch_size, self._num_channels, self._sequence_length, self._input_size = input_sequence.shape
        self._stop_iter_on_next = False

        # Divide the input sequence into as many fixed-size sequences as possible.
        self._num_stacked_sequences = self._sequence_length - self._time_context_frames + 1
                
        if(self._num_stacked_sequences < 1):
             raise ValueError("Error: input sequence of length %d cannot be divided further" % (self._sequence_length) + 
                              "with a time context of %d samples." % (self._time_context_frames))

        # Allocate tensors.
        with torch.no_grad():
            self._batched_input_sequence = torch.empty((self._num_stacked_sequences, 
                                                        self._num_channels,
                                                        self._time_context_frames, 
                                                        self._input_size), dtype=torch.float32).to(input_sequence.device)

            # Assign the sequences one by one.
            for seq_index in range(self._num_stacked_sequences):
                seq = input_sequence[0, :, seq_index:seq_index+self._time_context_frames, :]
                self._batched_input_sequence[seq_index,:,:,:] = seq
        
            # Assign the targets. The first target frame aligns with the end of the first sequence after 
            # "time_context_frames" samples.
            self._batched_targets = target_sequence[:, self._time_context_frames-1:, :]

    def __iter__(self):
        return self

    def __next__(self):
        # Check if the iteration is completed.
        if(self._stop_iter_on_next):
            self._start_index = 0
            self._stop_index = 0
            raise StopIteration
        else:
            # If not, increment the batch indices.
            self._start_index = self._stop_index 
            self._stop_index = self._start_index + self._batch_size

            # Check, if the end of the array will be reached this iteration.
            if(self._stop_index >= self._num_stacked_sequences):
                self._stop_iter_on_next = True
                self._stop_index = self._num_stacked_sequences

            # Return the respective slice, including any remaining part at 
            # the end of the iteration.
            # _batched_input_sequence = [num_stacked_sequences, _time_context_frames, input_size]
            # _batched_targets = [num_stacked_sequences, input_size]
            return (self._batched_input_sequence[self._start_index:self._stop_index, :, :, :],
                    self._batched_targets[:, self._start_index:self._stop_index, :])




            
