import torch
import matplotlib.pyplot as plt

class RsVocoderSynthesisLstmModel(torch.nn.Module):
    ''' LSTM Model for the mel spectrogram prediction. 
    Args:
        hyperparameters (dict): dictionary containing the hyperparameters.
        device (str): computing device, cpu or cudaX (X = GPU number).
    '''

    def __init__(self, hyperparameters, device="cpu"):

        super(RsVocoderSynthesisLstmModel, self).__init__()

        # Model's hyperparameters.
        self.input_size = hyperparameters['input_size']   # number of features (frequencies)
        self.output_size = hyperparameters['output_size'] # number of vocoder parameters
        self.hidden_size = hyperparameters['hidden_size'] 
        self.num_layers = hyperparameters['num_layers']
        self.is_bidirectional = hyperparameters['is_bidirectional']
        self.num_directions = 2 if self.is_bidirectional else 1
        self.dropout_prob = hyperparameters['dropout_prob']
        self.num_epochs = hyperparameters['num_epochs']
        self.device = device

        # Initialze the hidden and cell states of the lstm.
        self.c = []
        self.h = []
        self.reset_hidden_states(batch_size=1)

        # Define the model structure.
        if(self.num_layers > 1):
            self.lstm = torch.nn.LSTM(input_size=self.input_size, 
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      dropout=self.dropout_prob,
                                      bidirectional=self.is_bidirectional,
                                      batch_first=True)
        else:
            self.lstm = torch.nn.LSTM(input_size=self.input_size, 
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers,
                                      dropout=0,
                                      bidirectional=self.is_bidirectional,
                                      batch_first=True)

        self.dropout = torch.nn.Dropout(p=self.dropout_prob)

        self.linear = torch.nn.Linear(in_features=self.hidden_size*self.num_directions,
                                      out_features=self.output_size)


    def reset_hidden_states(self, batch_size):
        self.h = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.c = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, requires_grad=True).to(self.device)


    def forward(self, X_batch, return_sequence=False):
        '''
            Necessary function to override to forward the input through the model.
            Args:
                X_batch (3-d tensor, float32): containing the padded radar sequences [batch_size, max_seq_length, input_size]
                Note: currently, only a single sequence is allowed, so max_seq_length = sequence_length (no batching here).
            Returns:
                output (2-d tensor, float32): Predicted vocoder parameters. Has dimensions of
                                            [batch_size*sequence_length, output_size] or
                                            [batch_size, output_size] (if return_sequence=False).
        '''

        self.lstm.flatten_parameters() # Prevent excessive memory use by scattered parameters in memory.
        batch_size, num_channels, sequence_length, num_features = X_batch.shape

        # Flatten the 3-d stack of transforms to a stack of 2-d sequences, concatenated along the feature dimension.
        X_batch = X_batch.permute(0,2,1,3).reshape(batch_size,sequence_length,-1)

        '''
            fig, axs = plt.subplots(3,3)
            for idx in range(axs.size):
                row = idx//3
                col = idx%3
                axs[row][col].imshow(X_batch[idx,:,:])
            plt.show()
        '''

        # Initialize the hidden and the cell states to 0. Also detaches the comp. graph.
        self.reset_hidden_states(batch_size)
        X_batch = X_batch.to(self.device)

        # Calculate the lstm output.
        output, (self.h, self.c) = self.lstm(X_batch, (self.h, self.c)) 

        # Reshape the output to [batch_size*seq_length, hidden_size*num_dimensions]
        # if the full sequence should be processed further. Sequences are concatenated along dim=0.
        if(return_sequence):
            output = torch.reshape(output, (-1,self.num_directions*self.hidden_size)).contiguous() 
        else: # If not, just return the last hidden state. 
            output = output[:,-1,:].contiguous() # [batch_size, hidden_size*num_dimensions]
        
        # Apply dropout.
        output = self.dropout(output) 

        # Apply the fully connected linear layer.
        output = self.linear(output)

        return output


    def predict_stepwise(self, X_batch):
        '''
            Function to predict one step at a time without resetting the internal state (equivalent to 
            using forward() with return_sequence=True, but with manual advance step by step).

            Args:
                X_batch (3-d tensor, float32): containing the padded radar sequences [batch_size, max_seq_length, input_size]
                Note: currently, only a single sequence is allowed, so max_seq_length = sequence_length (no batching here).

            Returns: output (2-d tensor): Predicted vocoder parameters. Has dimensions of
                                        [1, output_size] 

            Note: the hidden states need to be reset prior to calling this functions consecutively by calling
                  reset_hidden_states(batch_size=1).
        '''

        self.lstm.flatten_parameters() # Prevent excessive memory use by scattered parameters in memory.

        X_batch = X_batch.to(self.device)

        # Calculate the lstm output.
        output, (self.h, self.c) = self.lstm(X_batch, (self.h, self.c)) 

        output = output.squeeze(dim=1).contiguous() # [1, hidden_size*num_directions] remove the frame/time dimension (must be 1)
        
        # Apply dropout.
        output = self.dropout(output) 

        # Apply the fully connected linear layer.
        output = self.linear(output) # [1, output_size]

        return output
   