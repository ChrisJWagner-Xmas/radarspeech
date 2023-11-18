from dataclasses import dataclass
from typing import List

@dataclass
class Dataset:
    subject_id : str
    full_hpc_corpus_path : str
    full_hpc_test_corpus_path : str
    full_local_corpus_path : str
    full_local_test_corpus_path : str

@dataclass
class Transforms:
    spectrum_keys : List[str]
    transform_keys : List[str]
    norm_intervals : List[int]
    num_freqs_used : int # Always starts at freq. index 0 (1 GHz currently).
    use_ltas_subtraction : bool


@dataclass
class Run:
    full_results_output_folder : str
    full_checkpoint_folder_name : str
    full_log_file_name : str
    is_hpc : bool
    is_test_mode : bool
    num_hp_eval_runs : int


@dataclass
class LstmHyperparams:
    input_size : int # Will get adjusted when constucting the lstm.
    output_size : int # Number of mels.
    learning_rate : float
    lr_reduction_factor : float # Currently not used.
    hidden_size : int
    num_layers : int
    dropout_prob : float
    is_bidirectional : bool
    num_epochs : int
    batch_size : int 
    patience : int
    time_context_frames : int # time context the lstm sees.
    fixed_lag_frames : int # future context if negative.

@dataclass
class CrnHyperparams:
    in_channels : List[int]
    out_channels : List[int]
    output_size : int
    lstm_hidden_size : int
    lstm_num_layers : int
    batch_size : int
    num_epochs : int
    time_context_frames : int
    fixed_lag_frames : int
    learning_rate : float
    patience : int

@dataclass
class LstmRunConfig:
    dataset : Dataset
    transforms : Transforms
    run : Run
    lstm_hyperparams : LstmHyperparams

@dataclass
class CrnRunConfig:
    dataset : Dataset
    transforms : Transforms
    run : Run
    crn_hyperparams : CrnHyperparams