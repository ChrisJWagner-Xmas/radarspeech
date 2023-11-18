import torch
import matplotlib.pyplot as plt

class RsVocoderSynthesisGruModel(torch.nn.Module):
    """description of class"""

    def __init__(self, hyperparameters, device="cpu", debug_mode=False):
        '''
            LSTM Model for the vocoder parameter prediction. 
        
        Args:
            hyperparameters (dict): dictionary containing the hyperparameters.
            device (str): computing device, cpu or cudaX (X = GPU number).
            debug_mode (bool): Output debug information (True) or not (False).
        '''
        super(RsVocoderSynthesisGruModel, self).__init__()

        # Model's hyperparameters.
        self.input_size = hyperparameters['input_size']   # number of features (frequencies)
        self.output_size = hyperparameters['output_size'] # number of vocoder parameters
        self.hidden_size = hyperparameters['hidden_size'] 
        self.num_layers = hyperparameters['num_layers']
        # self.batch_size = hyperparameters['batch_size'] # Infer from input b.c. of last batch which might be smaller
        self.dropout_prob = hyperparameters['dropout_prob']
        self.num_epochs = hyperparameters['num_epochs']
        self.debug_mode = debug_mode
        self.device = device

        # Define the model structure.
        self.gru = torch.nn.GRU(input_size=self.input_size, 
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout=self.dropout_prob,
                                  bidirectional=False,
                                  batch_first=True)

        self.dropout = torch.nn.Dropout(p=self.dropout_prob)

        self.linear = torch.nn.Linear(in_features=self.hidden_size,
                                      out_features=self.output_size)



    def forward(self, X_batch):
        '''
            Necessary function to override to forward the input through the model.
            Args:
                X_batch (3-d tensor, float32): containing the (padded) radar sequences [batch_size, max_sequence_length, input_size]
                Note: currently, only a single sequence is allowed, so max_seq_length = sequence_length (no batching here).
                return h_T (2-d tensor, float32): Predicted vocoder parameters [batch_size, max_sequence_length, output_size]
                            Note: output_size is the number of vocoder output features.
        '''

        # self.lstm.flatten_parameters() TODO: NECESSARY?

        batch_size, sequence_length, _ = X_batch.shape

        # Initialize the hidden and the cell states to 0.
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device)

        X_batch = X_batch.to(self.device)

        # Calculate the lstm output.
        _, h_T = self.gru(X_batch, h_0) # No detach() here (TODO: OK?) 

        # Since only the last output frame will be used, h_T is used from here.
        h_T = h_T.contiguous() 
        
        # Use only the last layer's hidden output for num_layers > 1.
        h_T = h_T[-1,:,:] # has dimensions [batch_size, hidden_size]

        # Apply dropout.
        h_T = self.dropout(h_T) 

        # Apply the fully connected linear layer.
        h_T = self.linear(h_T)

        return h_T # has dimensions [batch_size, output_size]



    



