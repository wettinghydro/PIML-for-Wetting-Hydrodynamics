
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from   scipy import interpolate
import matplotlib as mpl
mpl.use('Agg') # use matplotlib without graphics
import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer

from Adam import Adam


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
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
   
   

################################################################
# configs
################################################################

case    = 'training_dataset_AI1'
#
TRAIN_PATH = '../data/training/'+case+'.npy'
TEST_PATH  = '../data/training/'+case+'.npy'
#
ntrain     = 64000 
ntest      = 16000 
modes      = 2
width      = 128
batch_size = 20
#
epochs          = 500
learning_rate   = 1.e-3
scheduler_step  = 50 
scheduler_gamma = 0.5
#
text = 'w' + str(width) + '_FL8'
#
print('epochs           = '+str(epochs),flush=True)
print('iterations       = '+str(int(epochs*ntrain/batch_size)),flush=True)
print('learning rate    = '+str(learning_rate),flush=True)
print('scheduler step   = '+str(scheduler_step),flush=True)
print('scheduler gamma  = '+str(scheduler_gamma),flush=True)
#
path           = text
path_model     = 'model/'+path
path_plots     = 'figures/'+path
#
np.save(path_model+'_other_params',[modes,width])
#
sub  = 1   # sub-sampling
step = 1   # prediction step, should be 1


################################################################
# load data
################################################################
data     = np.load(TRAIN_PATH)

data[:,0,:] = get_first_harmonic(data[:,0,:]) # x
data[:,1,:] = get_first_harmonic(data[:,1,:]) # y
data[:,2,:] = get_first_harmonic(data[:,2,:]) # apparent theta
data[:,3,:] = get_first_harmonic(data[:,3,:]) # local theta
data[:,4,:] = get_first_harmonic(data[:,4,:]) # cox
data[:,5,:] = get_first_harmonic(data[:,5,:]) # cfd
#
S    = data.shape[2]
u    = np.arange(S) * 2*np.pi/S
grid = u.reshape(1, S, 1)
grid = torch.tensor(grid, dtype=torch.float)
np.random.shuffle(data)
#
x_in       = my_to_torch(data[:ntrain,0,::sub]).reshape(ntrain,S,1)
y_in       = my_to_torch(data[:ntrain,1,::sub]).reshape(ntrain,S,1)
centroid_x = torch.mean(x_in,1).reshape(ntrain,1,1).repeat(1,S,1)
centroid_y = torch.mean(y_in,1).reshape(ntrain,1,1).repeat(1,S,1)
rx         = x_in-centroid_x
ry         = y_in-centroid_y
r          = np.sqrt(rx**2+ry**2)
#
train_a    = torch.cat([ my_to_torch(data[:ntrain,2,::sub]).reshape(ntrain,S,1), \
                         my_to_torch(data[:ntrain,3,::sub]).reshape(ntrain,S,1), \
                         ],dim=-1)
train_u    = my_to_torch(data[:ntrain,5,::sub]-data[:ntrain,4,::sub]).reshape(ntrain,S,1)
#
x_in       = my_to_torch(data[-ntest:,0,::sub]).reshape(ntest,S,1)
y_in       = my_to_torch(data[-ntest:,1,::sub]).reshape(ntest,S,1)
centroid_x = torch.mean(x_in,1).reshape(ntest,1,1).repeat(1,S,1)
centroid_y = torch.mean(y_in,1).reshape(ntest,1,1).repeat(1,S,1)
rx         = x_in-centroid_x
ry         = y_in-centroid_y
r          = np.sqrt(rx**2+ry**2)
#
test_a     = torch.cat([ my_to_torch(data[-ntest:,2,::sub]).reshape(ntest,S,1), \
                         my_to_torch(data[-ntest:,3,::sub]).reshape(ntest,S,1), \
                         ],dim=-1)
test_u     = my_to_torch(data[-ntest:,5,::sub]-data[-ntest:,4,::sub]).reshape(ntest,S,1)
#
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True )
test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset( test_a,  test_u), batch_size=batch_size, shuffle=False)



################################################################
# training and evaluation
################################################################

model = FNO1d(modes,width)
model.to(device)
#
print('model parameters = {}'.format(count_params(model)),flush=True)
#
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
myloss    = LpLoss(size_average=False)

min_error = 1.0
plot_error = np.array([[],[]])

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_mse     = 0
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        #
        x  = torch.cat([xx,grid.repeat(batch_size,1,1).to(device)],-1)
        #
        im = model(x)
        #
        loss = myloss(im.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        mse = F.mse_loss(im.view(batch_size, -1), yy.view(batch_size, -1), reduction='mean')
        #
        train_l2_step += loss.item()
        train_mse += mse.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            # 
            x  = torch.cat([xx,grid.repeat(batch_size,1,1).to(device)],-1)
            #
            im = model(x)
            #
            loss = myloss(im.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            #
            test_l2_step += loss.item()

    t2 = default_timer()
    scheduler.step()
    #
    # Print the output
    #
    train_mse /= len(train_loader)
    separ_train_error = train_l2_step / ntrain * step
    separ_test_error  = test_l2_step  / ntest  * step
    print('===========================',flush=True)
    print('epoch = '+str(ep+1)+' (wall time = '+str(t2-t1)+')',flush=True)
    print('===========================',flush=True)
    print('training error = '+str(separ_train_error),flush=True)
    print('testing error  = '+str(separ_test_error),flush=True)
    print('train mse      = '+str(train_mse),flush=True)
    #
    train_error = separ_train_error
    test_error  = separ_test_error
    if test_error<min_error:
        print('Validation error reduced. Printing model parameters...',flush=True)
        torch.save({'epoch': ep, \
                    'model_state_dict': model.state_dict(), \
                    'optimizer_state_dict': optimizer.state_dict(), \
                    'scheduler_state_dict': scheduler.state_dict(), \
                    'loss': loss, \
                    }, path_model)
        min_error = test_error
    #
    # Plot the error
    #
    plot_error=np.concatenate([plot_error,[[train_error],[test_error]]],-1)
    plt.plot(np.linspace(1,ep+1,ep+1),plot_error[0,:],"r",label='training error')
    plt.plot(np.linspace(1,ep+1,ep+1),plot_error[1,:],"b",label='validation error')
    plt.legend()
    plt.yscale("log")
    plt.savefig(path_plots+'_errors.pdf',bbox_inches='tight',transparent=True)
    plt.clf()
    
    print(ep+1, train_error, test_error, file=open(path_model+'_err.out', 'a'),flush=True)

