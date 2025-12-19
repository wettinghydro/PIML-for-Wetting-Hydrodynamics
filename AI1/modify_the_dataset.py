import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam

from scipy import interpolate


plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"

torch.manual_seed(0)
np.random.seed(0)

def my_to_torch(x):
    if not torch.is_tensor(x):
        x = x.astype(np.float32)
        return torch.from_numpy(x)
    else:
        return x

def fs(F):
    n = int(len(F)/2)
    F = np.fft.rfft(F)
    F /= n
    F[0]*= 0.5
    return F[:-1]  # Cutoff frequency removed

def ifs(F):
    m = len(F)
    G = m*np.concatenate((F,[0.0]))
    G[0] *= 2
    return np.fft.irfft(G)

def get_first_harmonic(data):
    for i in range(data.shape[0]):
        data_f     = fs(data[i])
        data_f[0 ] = 0 
        data_f[2:] = 0
        data[i]    = ifs(data_f)
    return data

def filter_higher_harmonics(data,n):
    for i in range(data.shape[0]):
        data_f     = fs(data[i])
        data_f[n:] = 0
        data[i]    = ifs(data_f)
    return data

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

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

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
        self.fc0 = nn.Linear(3, self.width)
        # Apply a 1D convolution in the spectral case (from class SpectralConv1d)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv4 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv5 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv6 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv7 = SpectralConv1d(self.width, self.width, self.modes1)
        # Apply a 1D convolution over an input signal
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size,...)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.w4 = nn.Conv1d(self.width, self.width, 1)
        self.w5 = nn.Conv1d(self.width, self.width, 1)
        self.w6 = nn.Conv1d(self.width, self.width, 1)
        self.w7 = nn.Conv1d(self.width, self.width, 1)
        # This is the step before the last application of the non-linearity
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        
        # Fourier layer (1)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer (2)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer (3)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)
        #
        # Fourier layer (4)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer (5)
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer (6)
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x = x1 + x2
        x = F.relu(x)
        
        # Fourier layer (7)
        x1 = self.conv6(x)
        x2 = self.w6(x)
        x = x1 + x2
        x = F.relu(x)

        # Fourier layer (8)
        x1 = self.conv7(x)
        x2 = self.w7(x)
        x = x1 + x2

        # change the ordering of the columns of x
        # from [batch_size,self.width,s] to [batch_size,s,self.width]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
   
   

################################################################
# configs
################################################################

model_name = 'w128_FL8'

params = np.load('model/'+model_name+'_other_params.npy')
model  = FNO1d(params[0],params[1]).to(device)

checkpoint = torch.load('model/'+model_name)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(count_params(model))

sub = 1

################################################################
# load data
################################################################
case    = 'training_dataset_AI1'
#
TEST_PATH  = '../data/training/'+case+'.npy'
data       = np.load(TEST_PATH)

data[:,0,:] = get_first_harmonic(data[:,0,:]) # x
data[:,1,:] = get_first_harmonic(data[:,1,:]) # y
data[:,2,:] = get_first_harmonic(data[:,2,:]) # θ
data[:,3,:] = get_first_harmonic(data[:,3,:]) # θs
data[:,4,:] = get_first_harmonic(data[:,4,:]) # cox
data[:,5,:] = get_first_harmonic(data[:,5,:]) # cfd

S       = data.shape[2]
ndata   = data.shape[0]
#
u    = np.arange(S) * 2*np.pi/S
grid = u.reshape(1, S, 1)
grid = torch.tensor(grid, dtype=torch.float)
#
#
x_in       = my_to_torch(data[:,0,::sub]).reshape(ndata,S,1)
y_in       = my_to_torch(data[:,1,::sub]).reshape(ndata,S,1)
centroid_x = torch.mean(x_in,1).reshape(ndata,1,1).repeat(1,S,1)
centroid_y = torch.mean(y_in,1).reshape(ndata,1,1).repeat(1,S,1)
rx         = x_in-centroid_x
ry         = y_in-centroid_y
r          = np.sqrt(rx**2+ry**2)
test_a     = torch.cat([\
                        my_to_torch(data[:,2,::sub]).reshape(ndata,S,1), \
                        my_to_torch(data[:,3,::sub]).reshape(ndata,S,1), \
                        ],dim=-1)
test_u     = my_to_torch(data[:,5,::sub]-data[:,4,::sub]).reshape(ndata,S,1)
#


################################################################
# evaluation
################################################################

model_first_harmonic = data[:,0,:]*0

for sample in range(ndata):
    t1 = default_timer()
    #
    pred = torch.zeros(test_u.shape)
    myloss    = LpLoss(size_average=False)
    xx = test_a[sample:sample+1]
    yy = test_u[sample:sample+1]
    loss = 0
    test_l2_step = 0
    #
    xx = xx.to(device)
    yy = yy.to(device)
    # 
    x  = torch.cat([xx,grid.repeat(1,1,1).to(device)],-1)
    #
    im = model(x)
    #
    loss = myloss(im.reshape(1, -1), yy.reshape(1, -1))
    #
    model_first_harmonic[sample] = im[0,:,0].cpu().detach().numpy()
    #
    test_l2_step += loss.item()
    t2 = default_timer()
    if sample%100==0:
        print('sample = '+str(sample)+'/'+str(ndata)+' ==> Loss = '+str(loss.item()))


data       = np.load(TEST_PATH)
x_in       = data[:,0,::sub]
y_in       = data[:,1,::sub]
θ          = data[:,2,::sub]
θs         = data[:,3,::sub]
vel_cox    = data[:,4,::sub]
vel_cfd    = data[:,5,::sub]

harmonics = 16
vel_base = filter_higher_harmonics(np.array(vel_cox),harmonics) + model_first_harmonic


DATA = x_in,y_in,θ,θs,vel_base,vel_cfd

np.save('../data/training/training_dataset_AI2.npy',DATA)
