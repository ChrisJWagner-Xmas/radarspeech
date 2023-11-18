
'''
    Main script for training a network with all available training data on optimized hyperparameters.
'''

import matplotlib.pyplot as plt
import torch
import numpy as np
import sys, os

from datasets import rs_corpus
from datasets import rs_dataset
from train_model import train_and_evaluate
from models import RsLstmModel
from utils.file_io_functions import save_results_to_file, log_message
from utils.plotting_functions import plot_sequence_stack
import datetime
import sklearn.model_selection


if __name__ == "__main__":

    # +++++++++++++++++ Evaluation input selection ++++++++++++++++++++++++++++++++++++

    subject_index = 0 # from [0,1]
    
    spectra_list = ['S12','S12','S12'] # Careful, S12 and S32 might be switched
    transforms_list = ['mag','mag_delta','phase_delta']
    norm_interval = [[0,1],[0,1],[-1,1]] 
    freq_start_index = 0
    freq_stop_index = 85

    checkpoint_file = "lstm_optim_model_18_22_12.pt" # NYI

    num_hidden_units = 100
    num_epochs = 30
    learning_rate = 0.001
    batch_size = 8
    dropout_prob = 0
    num_lstm_layers = 1 

    num_workers = 4
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    dt = datetime.datetime.now()
    timestamp = "%d_%d_%d" % (dt.hour, dt.minute, dt.second)
    log_file_name = "full_training_intra_session_run_log_file_" + timestamp + ".txt"

    num_classes = 50
    num_features = len(spectra_list)*(freq_stop_index - freq_start_index)
    ptf = train_and_evaluate.ParameterizedTransformFunction(spectra=spectra_list,
                                                            transforms=transforms_list,
                                                            freq_start_index=freq_start_index,
                                                            freq_stop_index=freq_stop_index,
                                                            norm_interval=norm_interval,
                                                            ltas_mag = [None,None])

    transform_function = ptf.transform_sequences

    # Try to load the corpus from either the local corpus directory of the HPC corpus directory.
    paths = rs_corpus.get_corpus_full_file_paths()

    # Try to load the corpus with one of the provided paths.
    for path_name in paths:
        path = paths[path_name]
        training_corpus = rs_corpus.load_corpus_from_file("%s\\processed_training_corpus.pkl" % (path))
        if training_corpus is not None:
            log_message("Loaded Corpus.", log_file_name)
            break

    # If training_corpus is still unloaded, exit with error.
    if training_corpus is None:    
        sys.exit("Error: training_corpus is None. Check the path.")

    full_dataset = rs_dataset.RsDataSet(training_corpus, 
                            subject_index=subject_index, 
                            session_indices=[0,1,2], 
                            transform_fn=transform_function)
   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    log_message("Current Subject: S00%d" % (subject_index+1), log_file_name)
    log_message("Spectra: %s" % (spectra_list), log_file_name)
    log_message("Transforms: %s" % (transforms_list), log_file_name)
    log_message("number of features: %d" % (num_features), log_file_name)
    log_message("Used device: %s" %(device), log_file_name)

    # Initialize the hyperparameter dict with the fixed values and default values for variable ones.
    hyperparameters = {'input_size' : num_features,
                        'output_size' : num_classes,
                        'num_hidden_units' : num_hidden_units,
                        'num_lstm_layers' : num_lstm_layers,
                        'dropout_prob' : dropout_prob,
                        'is_bidirectional' : True,
                        'batch_size': batch_size,
                        'learning_rate' : learning_rate,
                        'num_epochs' : num_epochs}

    # Create the train/test sampler and data loader.
    train_indices = np.linspace(0,full_dataset.__len__()-1, full_dataset.__len__(),dtype='int32') # all radargram sequence indices.
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices) # Random sampling helps with faster convergence.

    train_data_loader = torch.utils.data.DataLoader(full_dataset, 
                                                   batch_size=hyperparameters['batch_size'], 
                                                   shuffle=False, 
                                                   num_workers=num_workers,
                                                   collate_fn=rs_dataset.pad_seqs_and_collate,
                                                   sampler=train_sampler)

    log_message("Instantiating model with parameters:", log_file_name)
    for parameter in hyperparameters:
        log_message("%s %s" % (parameter, hyperparameters[parameter]), log_file_name)
     
    # Instantiate new model & optimizer.
    rs_lstm_model = RsLstmModel(hyperparameters, device=device, debug_mode=False).to(device)
    optimizer = torch.optim.Adam(rs_lstm_model.parameters(), 
                                            lr=hyperparameters['learning_rate'])
    trainer = train_and_evaluate.Trainer(num_epochs=hyperparameters['num_epochs'],
                                            device=device,
                                            is_verbose=True,
                                            log_fn=log_message)

    trainer.full_log_file_name = log_file_name # use same logging function for training outputs.

    # Train the model.
    log_message("Started training.", log_file_name)

    trainer.fit_model(model=rs_lstm_model, 
                        optimizer=optimizer, 
                        train_data_loader=train_data_loader, 
                        validation_data_loader=None, 
                        patience=None)

    hyperparameters['num_epochs'] = trainer.training_results.num_epochs_to_train # override initial value for saving.

    training_results = {"batch_losses" : trainer.training_results.batch_loss_history,
                        "validation_metric" : trainer.training_results.validation_accuracies_history,
                        "train_metric" : trainer.training_results.train_accuracies_history,
                        "validation_accuracy" : trainer.training_results.max_validation_accuracy}

    train_results_file_name = "lstm_optimization_" + timestamp + "_" + "_train.txt"

    save_results_to_file(hyperparameters=hyperparameters, 
                            history=training_results,
                            file_name=train_results_file_name,
                            append=True)

    # Save the model.
    model_file_name = "lstm_optim_model_" + timestamp + ".pt"
    torch.save(rs_lstm_model, model_file_name)

    # Also save the trace for C++ import.
    trace_file_name = "lstm_optim_trace_" + timestamp + ".pt"
    example_length = torch.tensor([100],dtype=torch.int64)
    example_input = torch.rand((1,100,num_features))
    output = rs_lstm_model(example_input, example_length)
    traced_script_module = torch.jit.trace(rs_lstm_model, (example_input, example_length))
    traced_script_module.save(trace_file_name)




