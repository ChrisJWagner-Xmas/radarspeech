from src.datasets import rs_synthesis_dataset
from src.models import RsVocoderSynthesisLstmModel, RsVocoderSynthesisGruModel, CnnRnnModel
from src.training import train_and_evaluate
from src.training import transforms
from src.fileio import file_io_functions
import torch
import os
import glob
import numpy as np
import sklearn.model_selection
import datetime
import logging
import pandas as pd
import json
import hydra
from hydra.core.config_store import ConfigStore
from config import CrnRunConfig

# Create a configStore object to "link" the dataclasses in config.py to the config.yaml params.
# (not mandatory, but this assigns actual data types to all fields for easier error checking)s
cs = ConfigStore.instance()
cs.store(name='crn_run_params',node=CrnRunConfig)

@hydra.main(version_base=None,config_path="src/conf",config_name="config")
def main(cfg: CrnRunConfig):

    if(cfg.run.is_hpc):
        num_workers = 8
    else:
        num_workers = 0

    # Currently fixed parameters.
    criterion = torch.nn.L1Loss() 
    norm_type = 'sequence' 
    input_parameter_list = ["tacotron_mel_power_spectrogram"]

    subject_id = cfg.dataset.subject_id

    os.makedirs(cfg.run.full_results_output_folder,exist_ok=True)
    os.makedirs(cfg.run.full_checkpoint_folder_name,exist_ok=True)

    dt = datetime.datetime.now()
    timestamp = f"{dt.hour}_{dt.minute}_{dt.second}"
    logger = logging.getLogger(f"logger_{timestamp}_{subject_id}")
    print(logger.name)
    # logger = file_io_functions.setup_logger(f"logger_{timestamp}", f"log_file_{'_'.join(cfg.transforms.transform_keys)}_{timestamp}")

    if(cfg.run.is_hpc):
        if(cfg.run.is_test_mode):
            full_corpus_path = os.path.join(cfg.dataset.full_hpc_test_corpus_path,subject_id)           
            print("====== IN TESTING MODE =======")
        else:
            full_corpus_path = os.path.join(cfg.dataset.full_hpc_corpus_path,subject_id)
    else:
        if(cfg.run.is_test_mode):
            full_corpus_path = os.path.join(cfg.dataset.full_local_test_corpus_path,subject_id)
            print("====== IN TESTING MODE =======")
        else:
            full_corpus_path = os.path.join(cfg.dataset.full_local_corpus_path,subject_id)
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("using_device: %s" %(device))

    ########################################################################
    ########## Setup the data set and loaders ##############################
    ########################################################################

    # Note: the datasets store the location of the files. Files are loaded into
    # memory in the data_loader during training.

    dataset = rs_synthesis_dataset.RsSynthesisDataset(full_corpus_path,
                                                      is_load_file_path_from_list=True,
                                                      input_parameter_list=input_parameter_list,
                                                      fixed_lag_frames=cfg.crn_hyperparams.fixed_lag_frames)

    # Create the object that contains the parameterized transform function for the dataset.
    transform_obj = transforms.ParameterizedTransformFunction(spectrum_keys=cfg.transforms.spectrum_keys,
                                                            transform_keys=cfg.transforms.transform_keys,
                                                            num_features_per_transform=cfg.transforms.num_freqs_used,
                                                            norm_intervals=cfg.transforms.norm_intervals,
                                                            norm_type=norm_type,
                                                            ltas_list=dataset.ltas_list,
                                                            use_ltas_subtraction=cfg.transforms.use_ltas_subtraction)

    dataset.transform_fn = transform_obj.transform
    print("Built dataset.")

    ### Split the dataset into {train, val, test}.
    X = np.linspace(start=0, stop=dataset.__len__()-1, num=dataset.__len__(), dtype='int32')
    train_fraction = 0.8
    val_test_fraction = 0.2

    # Split all sequences into a training set and a remaining set for testing and validation.
    # The ratio is 80/10/10 %.
    train_indices, val_test_indices = sklearn.model_selection.train_test_split(X, 
                                                                                train_size=train_fraction,
                                                                                test_size=val_test_fraction,
                                                                                random_state=1337)

    # Split the remaining sequences further into equal validation and test sets.
    validation_indices, test_indices = sklearn.model_selection.train_test_split(val_test_indices, 
                                                                                train_size=0.5,
                                                                                test_size=0.5,
                                                                                random_state=1337)

    # Double check the indices.
    all_indices = np.sort(np.concatenate((train_indices, validation_indices, test_indices), axis=0))
    if(np.sum(X == all_indices) != len(X)):
        raise ValueError("Error: Some indices are not part of any set. Check the indexing.")

    ### Instantiate the dataLoader.
    train_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, # fixed to 1 as the batching is done on the sequence itself.
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    collate_fn=rs_synthesis_dataset.pad_seqs_and_collate,
                                                    sampler=train_indices)

    valid_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, # fixed to 1 as the batching is done on the sequence itself.
                                                    shuffle=False,
                                                    num_workers=0,
                                                    collate_fn=rs_synthesis_dataset.pad_seqs_and_collate,
                                                    sampler=validation_indices)

    test_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, # fixed to 1 as the batching is done on the sequence itself.
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    collate_fn=rs_synthesis_dataset.pad_seqs_and_collate,
                                                    sampler=test_indices)

    print("Built data loaders.")

    ########################################################################
    ########## Train the model. ############################################
    ########################################################################

    run_folder_name = f"run_{timestamp}_{subject_id}"
    full_run_output_folder_name = os.path.join(cfg.run.full_results_output_folder, run_folder_name)
    os.mkdir(full_run_output_folder_name)

    logger.info(f"Files used: {full_corpus_path}")

    # Check parameters that need to match.
    if(cfg.crn_hyperparams.in_channels[0] != len(cfg.transforms.spectrum_keys)):
        raise ValueError(f"Number of input channels ({cfg.crn_hyperparams.in_channels[0]})" \
           + "does not match number of features used ({len(cfg.transforms.spectrum_keys)})")

    if(cfg.crn_hyperparams.in_channels[0] != len(cfg.transforms.spectrum_keys)):
        raise ValueError(f"Number of frequencies used ({cfg.transforms.num_freqs_used})" \
           + "does not match number features per transform ({cfg.transforms.num_features_per_transform})")

    crn = CnnRnnModel.CRN(hyperparameters=cfg.crn_hyperparams,
                            device=device).to(device)

    optimizer = torch.optim.Adam(crn.parameters(), lr=cfg.crn_hyperparams.learning_rate)
        
    if(cfg.run.is_test_mode):
        num_epochs = 2

    trainer = train_and_evaluate.Trainer(num_epochs=num_epochs,
                                            device=device,
                                            eval_mode='batch',
                                            logger=logger)

    # Also save the validation and test indices.
    full_val_indices_file_name = os.path.join(full_run_output_folder_name,"val_indices.txt")
    full_test_indices_file_name = os.path.join(full_run_output_folder_name,"test_indices.txt")
    np.savetxt(full_val_indices_file_name, validation_indices.astype(int), fmt='%i', delimiter=',')
    np.savetxt(full_test_indices_file_name, test_indices.astype(int), fmt='%i', delimiter=',')

    print("Started training."); logger.info("started_training")

    trainer.fit_model(model=crn, 
                        optimizer=optimizer, 
                        scheduler=None,
                        train_data_loader=train_data_loader, 
                        validation_data_loader=valid_data_loader, 
                        criterion=criterion,
                        patience=cfg.crn_hyperparams.patience,
                        batch_size=cfg.crn_hyperparams.batch_size,
                        time_context_frames=cfg.crn_hyperparams.time_context_frames)

    # Save the model checkpoint.
    checkpoint_file_name = f"crn_model_checkpoint_run_{timestamp}_{subject_id}.pt"
    full_checkpoint_file_name = os.path.join(cfg.run.full_checkpoint_folder_name, 
                                             checkpoint_file_name)

    checkpoint = {"model" : crn,
                    "optimizer_state_dict" : optimizer.state_dict(),
                    "num_epochs" : trainer.training_results.num_epochs_to_train}

    torch.save(checkpoint, full_checkpoint_file_name)

    ########################################################################
    ##### Predict frames using the trained model and the test set. #########
    ########################################################################

    print("started_testing"); logger.info("started_testing")

    # Create a folder to store the output parameter files in.
    output_params_file_name = f"{'_'.join(input_parameter_list)}_predicted_{timestamp}_{subject_id}"
    full_output_params_folder_name = os.path.join(cfg.run.full_results_output_folder, 
                                                    full_run_output_folder_name, 
                                                    output_params_file_name)

    if not os.path.isdir(full_output_params_folder_name):
        print(f"Creating directory {full_output_params_folder_name}")
        os.mkdir(full_output_params_folder_name)

    # Reload the saved model from file (to check if the loading is ok).
    loaded_checkpoint = torch.load(full_checkpoint_file_name)
    test_model = loaded_checkpoint['model']

    # Note: currently a workaround because there is a weird memory error
    # on the hpc when the test_loss is calculated which i did not catch yet.
    try:
        test_loss = trainer.eval_function(model=test_model, 
                                            data_loader=test_data_loader, 
                                            criterion=criterion,
                                            time_context_frames=cfg.crn_hyperparams.time_context_frames)
        logger.info(f"Test error: {test_loss}")
    except:
        mem_err = torch.cuda.memory_summary(device=device)
        print(mem_err)
        logger.info("Test error: n.c.")
        
    # Test the model on the test set.
    with torch.no_grad():
        for file_index, data in enumerate(test_data_loader):

            input_sequence, target_sequence = data
            input_sequence = input_sequence.to(device)
            target_sequence = target_sequence.to(device)

            sequence_length = target_sequence.shape[1]-cfg.crn_hyperparams.fixed_lag_frames # sequences are |fixed_lag_frames| short
            output_size = target_sequence.shape[-1]
            test_batch_size = input_sequence.shape[2] - cfg.crn_hyperparams.time_context_frames + 1

            batched_data = rs_synthesis_dataset.BatchData(input_sequence=input_sequence, 
                                                            target_sequence=target_sequence, 
                                                            batch_size=test_batch_size,
                                                            time_context_frames=cfg.crn_hyperparams.time_context_frames)

            for X,Y in batched_data:
                predictions = test_model(X)

            # Note: the lag is introduced in the dataloader and the last fixed_lag_frames excluded
            # from the sequence and thus from training. these need to be padded here to have them
            # have the same length as the ground truth data and avoid dim hassle.
            for n in range(-cfg.crn_hyperparams.fixed_lag_frames):
                predictions = torch.cat((predictions,predictions[-1,:].unsqueeze(dim=0)), dim=0)

            save_file_name = dataset.radar_sequence_file_names_list[test_indices[file_index]]
            save_file_name = save_file_name.split('.')[0] # remove .bin
  
            file_name = os.path.join(full_output_params_folder_name, f"{save_file_name}_{input_parameter_list[0]}.csv")
            np.savetxt(file_name, predictions.detach().cpu().numpy(), delimiter=',')

            # Also save as .numpy file for the HIFI-GAN Vocoder.
            full_numpy_folder_name = os.path.join(full_output_params_folder_name, "npy")
            if not os.path.isdir(full_numpy_folder_name):
                os.mkdir(full_numpy_folder_name)

            np_file_name = os.path.join(full_numpy_folder_name, f"{save_file_name}_{input_parameter_list[0]}.npy")
            np.save(np_file_name, predictions.detach().cpu().numpy())

    logger.info("Run finished successfully.")

if __name__ == "__main__":
    main()                


