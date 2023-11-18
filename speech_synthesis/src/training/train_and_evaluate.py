''' This module contains functions to evaluate the model with, as well as the
    Trainer class for training the model.
'''
import torch
import numpy as np
import collections
import copy
import matplotlib.pyplot as plt
import os
import math
from src.datasets import rs_synthesis_dataset
import logging
import src.training.transforms
from tqdm import tqdm 


training_results = collections.namedtuple('training_results', ['validation_loss_history',
                                                                'min_validation_loss'
                                                                'train_loss_history',
                                                                'num_epochs_to_train',
                                                                'best_model'])


def evaluate_model_on_sequence(model, data_loader, criterion, time_context_frames):
    ''' 
        Function to evaluate the model with.

    Args:
        model (torch.nn.Module): model to evaluate.
        data_loader (torch.utils.data.DataLoader): data loader to provide the {data, label} pairs.
        criterion (torch.nn.XXX): loss function. 

    Returns:
        loss (float32): scalar summed loss, according to the criterion.
    '''
    model.eval()

    loss = torch.tensor(0.0, dtype=torch.float32).to(model.device)
    total_num_frames = 0

    for input_sequence, target_sequence in tqdm(data_loader):

        input_sequence = input_sequence.to(model.device)
        target_sequence = target_sequence.to(model.device)

        # sequence_length is |fixed_lag_frames| short.
        # The alignment btwn input and target sequence is still offset by |fixed_lag_frames|, so 
        # the predictions can be directly compared to the targets, i.e.,
        # The input sequence starts at frame number |fixed_lag_frames| whereas the
        # target sequence starts at frame number 0. E.g., lag=-3: 3->0, 4->1, 5->2 etc.
        Y_pred = model(input_sequence, return_sequence=True)
        loss += criterion(Y_pred[time_context_frames:,:], target_sequence[0,time_context_frames:,:])
        total_num_frames += Y_pred[time_context_frames:,:].shape[0]

        # Leave out the first N frames (corr. to time context), because the network
        # needs to "settle" first.
    
    return loss/total_num_frames # Return the frame-averaged loss


def evaluate_model_batched(model, data_loader, criterion, time_context_frames):
    ''' 
        Function to evaluate the model on batched sequences with.

    Args:
        model (torch.nn.Module): model to evaluate.
        data_loader (torch.utils.data.DataLoader): data loader to provide the {data, label} pairs.
        criterion (torch.nn.XXX): loss function. 

    Returns:
        loss (float32): scalar summed loss, according to the criterion.
    '''
    model.eval()

    loss = torch.tensor(0.0, dtype=torch.float32).to(model.device)
    total_num_frames = 0

    for input_sequence, target_sequence in tqdm(data_loader):

        input_sequence = input_sequence.to(model.device)
        target_sequence = target_sequence.to(model.device)

        batch_size = input_sequence.shape[2] - time_context_frames + 1
        batched_data = rs_synthesis_dataset.BatchData(input_sequence=input_sequence, 
                                                      target_sequence=target_sequence, 
                                                      batch_size=batch_size,
                                                      time_context_frames=time_context_frames)

        for X,Y in batched_data:
            Y_pred = model(X)
            Y = Y.view(-1, Y_pred.shape[-1]) 
            loss += criterion(Y_pred, Y)
            total_num_frames += Y_pred.shape[0]
   
    return loss/total_num_frames # Return the frame-averaged loss


def _get_model_evaluation_function(mode):
    if(mode == 'sequence'):
        return evaluate_model_on_sequence
    elif(mode == 'batch'):
        return evaluate_model_batched
    else:
        raise ValueError("\'mode\' needs to be either \'sequence\' or \'batch\'")


class Trainer():
    '''
        Trainer class to train a given lstm model.
    '''

    def __init__(self, num_epochs=200, device='cpu', eval_mode='sequence', logger=None):
        ''' 
        Args:
            num_epochs (int): maximal number of epochs to train the model with.
            device (str): 'cpu' or 'cudaX' (X = device number) to train on cpu or gpu.
            eval_mode (str): either 'sequence' to evaluate the model on a 2-d sequence (for the pure rnn)
                             or 'batch' to evaluate it on successive [num_channels, time_context_frames, num_features_per_transform] stacks
                             (for the crn).
            logger (Logger): logger to print/save the progress status during training.

        Returns:
            None.
        '''
        
        if logger is not None:
            if not isinstance(logger, logging.Logger):
                raise TypeError("Error: logger needs to be an instance of logging.Logger, but is {:s}".format(type(logger)))

        self.num_epochs = num_epochs
        self.device = device
        self.eval_function = _get_model_evaluation_function(eval_mode)
        self.logger = logger
        self.training_results = training_results
        print("Run output will be saved to file {:s}".format(logger.name))


    def fit_model(self, model, optimizer, scheduler, train_data_loader, validation_data_loader, 
                      criterion, patience, batch_size, time_context_frames):
            '''
                Model fit function.
            Args:
                model (torch.nn.Module): lstm model being trained.
                optimizer (torch.optim): selected optimizer for training.
                scheduler: scheduler for learning rate adaptation.
                train_data_loader (torch.utils.data.DataLoader): data loader for the training data.
                validation_data_loader (torch.utils.data.DataLoader): data loader for the validation data.
                criterion: loss function.
                patience (int): allowed number of validation set evaluations below the current best metric 
                    until training is stopped.
                batch_size (int): batch size.
                time_context_frames (int): time context the crn is exposed to during each time step.
            Returns:
                None.
            '''

            # Init the training results struct.
            fail_count = 0
            self.training_results.validation_loss_history = []
            self.training_results.min_validation_loss = float("inf")
            self.training_results.train_loss_history = []
            self.training_results.num_epochs_to_train = self.num_epochs
            self.batch_size = batch_size
            self.patience = patience
            self.time_context_frames = time_context_frames

            self.criterion = criterion
            device = model.device
            learning_rate = optimizer.param_groups[0]['lr']

            for epoch in range(self.num_epochs):

                model.train() # Tells the model that it is in training mode

                for seq_index, data in tqdm(enumerate(train_data_loader)):

                    if((seq_index % 100) == 0):
                        print(" Processed %d sentences." % (seq_index))

                    input_sequence, target_sequence = data # both [batch_size, seq_len, num_features_per_transform]
                    input_sequence = input_sequence.to(self.device)
                    target_sequence = target_sequence.to(self.device)

                    _, num_channels, seq_len, num_features_per_transform = input_sequence.shape

                    # Note: the lstm model reshapes this 4-d tensor in its forward() function, no
                    # further action needed here.

                    '''
                        fig, axs = plt.subplots(1,num_channels)
                        for plot_index in range(num_channels):
                            # axs[plot_index].imshow(input_sequence[0,plot_index,:,:])
                            axs[plot_index].plot(input_sequence[0,plot_index,:,:].T)

                        plt.show()                   
                    '''

                    # Create iterable object to iterate over the batches for a single input/target sequence pair.
                    # Overwrite the batch_size (see below).
                    self.batch_size = seq_len - self.time_context_frames + 1
                    batched_data = rs_synthesis_dataset.BatchData(input_sequence=input_sequence, 
                                                                target_sequence=target_sequence, 
                                                                batch_size=self.batch_size,
                                                                time_context_frames=self.time_context_frames)

                    # Train with the fixed-size sequences in batches.
                    # Every small sequence in the batch is analyzed by the cnn frontend
                    # and fed into the lstm afterwards, to produce a single output frame.
                    # As such, batch_size in batched_data corresponds to the time context given
                    # to the lstm (filtered through the cnn) and time_context_frames is the width 
                    # of the image stack that the CNN sees.
                    for X, Y in batched_data:

                        # Clear the current parameter's gradients.
                        optimizer.zero_grad(set_to_none=True) # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

                        # Forward the model. 
                        Y_pred = model(X) # [batch_size, output_size]
                                              
                        Y = Y.view(-1, Y_pred.shape[-1]) # Last dim. is output_size

                        # Calculate the loss on the last frames of the sequences in the batch.  
                        loss = self.criterion(Y, Y_pred)

                        # Backward pass.     
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2) 

                        # Update parameters.
                        optimizer.step()

                print("Finished epoch %d" % (epoch))

                # +++++++++++++ Finished epoch ++++++++++++++

                with torch.no_grad():
                    print("Evaluating the model on the training set...")
                    train_loss = self.eval_function(model=model, 
                                                    data_loader=train_data_loader,
                                                    criterion=criterion,
                                                    time_context_frames=time_context_frames) 

                    # Check validation accuracy if a data loader is provided.
                    validation_loss = float("inf")
                    if(validation_data_loader is not None):
                        print("Evaluating the model on the validation set...")
                        validation_loss = self.eval_function(model=model, 
                                                             data_loader=validation_data_loader, 
                                                             criterion=criterion,
                                                             time_context_frames=time_context_frames)

                        self.training_results.validation_loss_history.append(validation_loss.item())

                        # Save the best-performing model (w.r.t. the validation accuracy).
                        if(validation_loss < self.training_results.min_validation_loss):
                            fail_count = 0
                            # update minimum and reset fail count if a new minimum was reached within the count limit.
                            self.training_results.min_validation_loss = validation_loss
                            self.training_results.num_epochs_to_train = epoch + 1 # +1 b.c. epoch index starts at 0.
                            self.training_results.best_model = copy.deepcopy(model) # Save a copy of the best-performing model. 
                        else:
                            fail_count += 1

                        # Adjust the learning rate if required.
                        if scheduler is not None:
                            scheduler.step(validation_loss)
                            learning_rate = optimizer.param_groups[0]['lr']

                        # Use early stopping if a patience value is provided.
                        if(self.patience is not None):
                            if(fail_count >= self.patience):
                                print("Early stopping asserted. training_finished.")
                                break # Stop at the current epoch and finish training.
                    else:
                        # If no validation data loader is provided, always overwrite the previous model with the new one.
                        self.training_results.min_validation_loss = validation_loss
                        self.training_results.num_epochs_to_train = epoch + 1 # +1 b.c. epoch index starts at 0.
                        self.training_results.best_model = copy.deepcopy(model) # Save a copy of the best-performing model. 

        
                if self.logger is not None:
                    self.logger.info("Epoch {:d}, Validation error: {:1.8f}, Train error: {:1.8f}, fail count: {:d}, lr={:1.6f}" 
                                .format(epoch, validation_loss, train_loss, fail_count, learning_rate)) 
                else:
                    print("Epoch %d, Validation error: %1.8f, Train error: %1.8f, fail count: %d lr=%1.6f" 
                          % (epoch, validation_loss, train_loss, fail_count, learning_rate))    