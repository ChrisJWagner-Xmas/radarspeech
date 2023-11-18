import torch

class CausalConvBlock(torch.nn.Module):
    # Taken from https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement/blob/master/model/crn.py
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2,3), # (time, freq) TODO: make kernel size a hyperparameter
            stride=(1,2),      # (-"-)
            padding=(1,0)      # (-"-), Note: (1,0) pads one frame to the left and right of the sequence and 0 frames to the upper and lower border.
                               # AFAIK: Padding needs to be (kernel_size[0]-1, 0) (see docu) # TODO: adjust if kernel_size is a HP
        )
        self.norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.ELU()
        self.chomp = self.conv.padding[0]

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x (4-d tensor): [batch_size, channels_in, time_steps, freqs]
        Returns:
            x (4-d tensor): batch_size, channels_in, time_steps, freqs]
        """
        x = self.conv(x)
        x = x[:, :, :-self.chomp, :]  # chomp away the right-most activation in the time dimension due to padding.
        x = self.norm(x)
        x = self.activation(x)
        return x


class CNN(torch.nn.Module):
    def __init__(self, num_features, in_channels, out_channels, device='cpu'):
        super(CNN, self).__init__()

        self.conv_blocks = torch.nn.ModuleList()
        self.num_conv_blocks = len(out_channels)

        for layer_index in range(self.num_conv_blocks):
            self.conv_blocks.append(CausalConvBlock(in_channels[layer_index],
                                                    out_channels[layer_index]))

    def calc_cnn_output_shape(self,in_channels,seq_len,num_features):
        ''' Automatically determine the shape of the output feature maps 
            stack produced by the CNN frontend to calculate the correct
            lstm input_size.
        '''
        with torch.no_grad():
            x = torch.randn((1,in_channels,seq_len,num_features)) # create dummy input
            x = self.forward(x)
            return x.shape

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x


class LSTM(torch.nn.Module):
    def __init__(self, input_size, num_lstm_layer, hidden_size, output_size, batch_size, device='cpu'):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_lstm_layer = num_lstm_layer
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.lstm_layer = torch.nn.LSTM(input_size=self.input_size,
                                        hidden_size=self.hidden_size,
                                        num_layers=self.num_lstm_layer,
                                        batch_first=True)

        self.linear = torch.nn.Linear(in_features=self.hidden_size,
                                out_features=self.output_size)

        self.h = []
        self.c = []
        self.reset_hidden_states(batch_size=self.batch_size)

    def reset_hidden_states(self, batch_size):
        self.h = torch.zeros(self.num_lstm_layer, batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.c = torch.zeros(self.num_lstm_layer, batch_size, self.hidden_size, requires_grad=True).to(self.device)

    def forward(self,x):
        '''
        Args:
            x (4-d tensor): [batch_size, ch_out, seq_len, width], i.e., ch_out feature maps of shape [seq_len, width]
                            batch_size corresponds to the number of frames the rnn sees during one forward pass.
        Returns:
            h (2-d tensor): [batch_size, output_size]
        '''
        self.lstm_layer.flatten_parameters()
        batch_size, ch_out, seq_len, width = x.shape
        # Reset the hidden states before feeding in one full batched sequence.
        self.reset_hidden_states(batch_size)
        # Reshape CNN output to [batch_size, seq_len, out_channels*width]
        x = x.permute((0,2,1,3)).reshape((batch_size, seq_len, -1))
        x, (self.h,self.c) = self.lstm_layer(x, (self.h, self.c)) # lstm input: [batch_size, seq_length, input_size] (with batch_first=True)
        x = x[:,-1,:].contiguous() # only use hast hidden frame
        x = self.linear(x)
        return x # [batch_size, output_size]


class CRN(torch.nn.Module):
    def __init__(self, hyperparameters, device="cpu"):
        '''
        Args:
            hyperparameters (dict): dictionary with all hyperparameters.
            device (str): compute device.
        '''
        super(CRN, self).__init__()
        self.device = device
        # CNN hyperparameter.
        self.num_features_per_transform = hyperparameters['num_features_per_transform']
        self.in_channels = hyperparameters['in_channels']
        self.out_channels = hyperparameters['out_channels']
        # LSTM hyperparameter.
        self.lstm_num_layers = hyperparameters['lstm_num_layers']
        self.lstm_hidden_size = hyperparameters['lstm_hidden_size']
        self.output_size = hyperparameters['output_size']
        self.batch_size = hyperparameters['batch_size']

        self.cnn = CNN(self.num_features_per_transform,
                       self.in_channels,
                       self.out_channels,
                       device=self.device)

        out_shape = self.cnn.calc_cnn_output_shape(self.in_channels[0],
                                                   500,
                                                   self.num_features_per_transform)

        lstm_input_size = self.out_channels[-1]*out_shape[3] # needs to be out_channels*width ["W" in pytorch docu]

        self.lstm = LSTM(lstm_input_size,
                         self.lstm_num_layers,
                         self.lstm_hidden_size,
                         self.output_size,
                         self.batch_size,
                         device=self.device)

    def forward(self, x, return_sequence=True):
        '''
        Args:
            x (4-d tensor): stack of batch of small input sequences [batch_size, ch_in, seq_len, freqs]
            return_sequence (bool): argument added so that the crn forward() function has the same signature.
                            Technically, the crn always returns as many output frames as specified by batch_size.
        Returns:
            h (2-d tensor): [batch_size, output_size]
        '''
        x = self.cnn(x)
        x = self.lstm(x)
        return x