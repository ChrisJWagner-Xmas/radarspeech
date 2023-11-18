import torch
import matplotlib.pyplot as plt

# Source: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py

class Chomp1d(torch.nn.Module):
    ''' TODO: WHAT EXACTLY DOES THIS FUNCTION DO?
    '''

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    ''' Main building block of a TCN, see "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
        for Sequence Modeling, 2018" (https://arxiv.org/abs/1803.01271) page 3.
    Args:
        input_size (int): input feature size for this temporal block.
        output_size (int): output feature size for this temporal block.
        kernel_size (int): Kernel size along the time axis.
        stride (int): stride along the time axis.
        dilation (int): dilation factor. Depends on the level of the current temporal block.
                        Usually increases quadratically, i.e., 1,2,4,8,... . Also influences the receptive field.
        padding (int): Padding at the end of the sequence. 
        dropout_prob (float32): dropout probability for this temporal block.

    Returns:
        feature maps (3-d torch tensor) of size [batch_size, num_channels[layer_level], seq_length] 

    '''

    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout_prob=0.2):

        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(input_size, output_size, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.ReLU()
        self.dropout_prob1 = torch.nn.Dropout(dropout_prob)

        self.conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(output_size, output_size, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = torch.nn.ReLU()
        self.dropout_prob2 = torch.nn.Dropout(dropout_prob)

        self.net = torch.nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout_prob1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout_prob2)
        self.downsample = torch.nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    ''' Full TCN.
        Args:
            num_inputs (int): number of input features.
            num_channels (list(int)): List of channel sizes after the input channel. Its length equals the number of layers
                                      between the input and output sequence.
                                      Equals the "y-direction" for a 2D sequence long the x/t direction.
            kernel_size (int): kernel size in the time direction. The Kernel size in the "y-direction"
                               is fixed to its length, resulting in "num_inputs" filter kernels along the time axis. (TODO: CHECK THIS CLAIM!)
                               Also influences the receptive field, together with the dilation_size and the number of layers:
                               receptive field size: 2**(len(channel_sizes))*(kernel_size-1)-1 (TODO: CHECK!)

        Returns:
            output sequence (3-d torch tensor) of size [batch_size, output_size, seq_length]        
    '''

    def __init__(self, hyperparameters, device="cpu"):

        super(TemporalConvNet, self).__init__()

        self.device = device
        self.input_size = hyperparameters['input_size']
        self.num_channels = hyperparameters['channel_sizes']
        self.kernel_size = hyperparameters['kernel_size']
        self.dropout_prob = hyperparameters['dropout_prob']

        layers = []
        num_levels = len(self.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1] # input features or subsequent num of feature maps at the residual block input.
            out_channels = self.num_channels[i] # num of feature maps after the passing through the residual block.
            layers += [TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(self.kernel_size-1) * dilation_size, dropout_prob=self.dropout_prob)]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

        