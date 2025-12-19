# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 16:09:53 2022

@author: a.demou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities3 import *
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from Adam import Adam

layers = 8

################################################################
# fourier layer
################################################################

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        # weights are initialised and are now considered as learnable parameters
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d_corrections(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d_corrections, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps and the grid (a(x), x)
        input shape: (batchsize, x=s, c=11)
        output: the solution of the next timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        # Applies a linear transformation to the incoming data
        # nn.Linear(in_features, out_features, ...)
        # input channel is 2
        self.fc0 = nn.Linear(3, self.width)
        # Apply a 1D convolution in the spectral case (from class SpectralConv1d)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        if layers > 1:
            self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        if layers > 2:
            self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        if layers > 3:
            self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        if layers == 8:
            self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv5 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv6 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv7 = SpectralConv1d(self.width, self.width, self.modes1)
        # Apply a 1D convolution over an input signal
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size,...)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        if layers > 1:
            self.w1 = nn.Conv1d(self.width, self.width, 1)
        if layers > 2:
            self.w2 = nn.Conv1d(self.width, self.width, 1)
        if layers > 3:
            self.w3 = nn.Conv1d(self.width, self.width, 1)
        if layers == 8:
            self.w4 = nn.Conv1d(self.width, self.width, 1)
            self.w5 = nn.Conv1d(self.width, self.width, 1)
            self.w6 = nn.Conv1d(self.width, self.width, 1)
            self.w7 = nn.Conv1d(self.width, self.width, 1)
 
       # This is the step before the last application of the non-linearity
        self.fc1 = nn.Linear(self.width, 128)
        # This is the final step, going from 128 to 3 channels (i.e. the solution u(x))
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # first step, raise the input to a higher-dimension space
        # from 2 to self.width channels
        # i.e. from [batch_size,s,2] to [batch_size,s,self.width]
        x = self.fc0(x)
        # change the ordering of the columns of x
        # from [batch_size,s,self.width] to [batch_size,self.width,s]
        x = x.permute(0, 2, 1)
        
        # Fourier layer (1)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        if layers > 1:
            x = F.gelu(x)

            # Fourier layer (2)
            x1 = self.conv1(x)
            x2 = self.w1(x)
            x = x1 + x2

        if layers>2:
            x = F.gelu(x)

            # Fourier layer (3)
            x1 = self.conv2(x)
            x2 = self.w2(x)
            x = x1 + x2
        
        if layers>3:
            x = F.gelu(x)
        
            # Fourier layer (4)
            x1 = self.conv3(x)
            x2 = self.w3(x)
            x = x1 + x2

        if layers == 8:
            x = F.gelu(x)

            # Fourier layer (5)
            x1 = self.conv4(x)
            x2 = self.w4(x)
            x = x1 + x2
            x = F.gelu(x)

            # Fourier layer (6)
            x1 = self.conv5(x)
            x2 = self.w5(x)
            x = x1 + x2
            x = F.gelu(x)
        
            # Fourier layer (7)
            x1 = self.conv6(x)
            x2 = self.w6(x)
            x = x1 + x2
            x = F.gelu(x)

            # Fourier layer (8)
            x1 = self.conv7(x)
            x2 = self.w7(x)
            x = x1 + x2

        # change the ordering of the columns of x
        # from [batch_size,self.width,s] to [batch_size,s,self.width]
        x = x.permute(0, 2, 1)
        # raise the solution even further, from [batch_size,s,self.width] to [batch_size,s,128]
        x = self.fc1(x)
        # apply the final non-linearity
        x = F.gelu(x)
        # bring the solution back to [batch_size,s,1]
        x = self.fc2(x)
        return x
  
   
