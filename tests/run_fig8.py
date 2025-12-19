import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from evolver import *
from model_1 import *
from model_2 import *
from frechet import improved_frechet,arc_length_parametrization
#
case   = int(sys.argv[1])
data_folder = f"../data/testing/fig8/case{case:1d}"
param_file  = f"../data/testing/fig8/case{case:1d}/params.in"
plot_folder = "plots/"
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

if case == 1:
    t_end  = 10
    plot_times = [0,t_end]
elif case == 2:
    t_end  = 10
    plot_times = [0,t_end]
elif case == 3:
    t_end  = 10
    plot_times = [0,t_end]
elif case == 4:
    t_end  = 40
    plot_times = [0,t_end]
elif case == 5:
    t_end  = 40
    plot_times = [0,t_end]
elif case == 6:
    t_end  = 40
    plot_times = [0,t_end]
#
def filter_higher_harmonics(data,n):
    data_f     = fs(data)
    data_f[n:] = 0
    data       = ifs(data_f)
    return data

def tikzplot(file,X,Y,style='',forget=True):
   with open(file,"w") as fid:
       if forget:
           style = style+',forget plot'

       if len(style)>0:
           fid.write('\\addplot['+style+']\n')
       else:
           fid.write('\\addplot\n')
       fid.write('table[row sep=crcr]{\n')
       dataXY = [X,Y]
       for x in zip(*dataXY):
           fid.write("%1.8f\t%1.8f\\\\\n" % (x))
       fid.write('};\n')

def calculate_area(x, y):
    n = len(x)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += (x[i] * y[j]) - (x[j] * y[i])

    return abs(area) / 2.0

################################
# Function for interpolating the contact line
################################
def D(F,ds):
    dF= d(F)/ds
    a = ds + np.roll(ds,-1)
    return (dF*np.roll(ds,-1) + np.roll(dF,-1)*ds)/a

d = lambda f: f - np.roll(f,1)

def get_cl(Φ,data):
    x_loc = data[0]
    y_loc = data[1]

    ds = np.sqrt(d(x_loc[:-1])**2 + d(y_loc[:-1])**2)

    # Compute the unit outward normal    
    Nx = +D(y_loc[:-1],ds)
    Ny = -D(x_loc[:-1],ds) 
    absN = np.sqrt(Nx**2+Ny**2)
    Nx /= absN
    Ny /= absN

    A  = 0.5*np.sum((x_loc[:-1]*Nx+y_loc[:-1]*Ny)*ds)
    xc = 0.5*np.sum(x_loc[:-1]**2*Nx*ds)/A
    yc = 0.5*np.sum(y_loc[:-1]**2*Ny*ds)/A 
    
    phi = []
    rho = []
    for i in range(len(x_loc)):
        dum = np.arctan2(y_loc[i]-yc,x_loc[i]-xc)
        if dum>0:
            phi.append(dum)
        else:
            phi.append(dum+2.*np.pi)
        rho.append(np.sqrt((y_loc[i]-yc)**2+(x_loc[i]-xc)**2))
    imax = phi.index(max(phi))
    imin = phi.index(min(phi))
    phi.append(phi[imax]-2.*np.pi)
    rho.append(rho[imax])
    phi.append(phi[imin]+2.*np.pi)
    rho.append(rho[imin])

    R_interp = scipy.interpolate.interp1d(phi, rho)
    R = []
    for phi_loc in Φ:
        R.append(float(R_interp(phi_loc)))
    
    return xc+R*np.cos(Φ), yc+R*np.sin(Φ)    


###########################
# Setting the heterogeneity
###########################

f     = open(param_file, "r")
lines = f.readlines()
f.close()
p     = [float(value) for value in lines[:7]]

def het(xx,yy):
    x   = np.cos(p[6])*xx - np.sin(p[6])*yy
    y   = np.sin(p[6])*xx + np.cos(p[6])*yy
    het = np.pi/2. + p[0]*(x) +\
                     p[1]*np.tanh(p[2]*np.cos(p[3]*np.pi*(x*np.sin(p[5])+y*np.cos(p[5])))*\
                                       np.cos(p[4]*np.pi*x))                    
    return het


#
S    = 128
u    = np.arange(S)*2*np.pi/S
cosφ = np.cos(u)
sinφ = np.sin(u)
grid = torch.tensor(u.reshape(1, S, 1), dtype=torch.float).to(device)
#
ic_xc = float(lines[7])
ic_yc = float(lines[8])
ic_r  = np.array([0.125]*S)
ic    = np.append(ic_r,[ic_xc,ic_yc])
#
mu  = 10.0
σ   = 1.0
q   = 0.1 # density ratio mu2/mu1
#
r_ini     = ic_r[0]
vol       = 0.5*(4./3.)*np.pi*r_ini**3
tmpfs_dir = 'ram_tmp/'
se_file   = os.path.join(tmpfs_dir,'drop.fe')  
mesh      = 'SEmesh128.mat'
# Create the SE class
se = evolver(mesh=mesh, V = vol, tmpfile=se_file, n=S,  R=ic_r, ic='BIMSE')
#

###########################
# progress bar
###########################
def evpbar(t,y,pbar,state):
    last_t, dt = state
    n = int((t-last_t)/dt)
    pbar.update(n)
    pbar.set_description("t = %1.2e" % t)
    state[0] = last_t + dt*n
    
    return 1 

################################
# Function for azimuthal differentiation
################################
kr = np.fft.rfftfreq(S,d=1/S)
def rdiff(f):
    f_hat = np.fft.rfft(f)
    # f_hat[-2:] = 0
    Fx = np.fft.irfft(1j*kr*f_hat)
    Fxx = np.fft.irfft(-kr**2*f_hat)
    return Fx,Fxx

####################################################
N_pol = 10
# Legendre-Gauss Nodes and Quadrature Points
x, W = np.polynomial.legendre.leggauss(N_pol)
f = lambda x,q: 2*np.sin(x)*(q**2*(x**2-np.sin(x)**2)+2*q*(x*(np.pi-x)+np.sin(x)**2)+((np.pi-x)**2-np.sin(x)**2)) \
                /(q*(x**2-np.sin(x)**2)*(np.pi-x+np.cos(x)*np.sin(x))+((np.pi-x)**2-np.sin(x)**2)*(x-np.cos(x)*np.sin(x)))
F = lambda x: 1/f(x,q)
####################################################

################################
# AI model for first harmonic
################################
path = '../AI1/model/trained_model/'
model_name = 'w128_FL8'
params = np.load(path+model_name+'_other_params.npy')
model_first_harmonic  = FNO1d_first_harmonic(params[0],params[1]).to(device)
checkpoint = torch.load(path+model_name)
model_first_harmonic.load_state_dict(checkpoint['model_state_dict'])
model_first_harmonic.eval()

################################
# AI model for higher order corrections
################################
path = '../AI2/model/trained_model/'
model_name = 'w64_FL8'
params = np.load(path+model_name+'_other_params.npy')
model_corrections  = FNO1d_corrections(params[0],params[1]).to(device)
checkpoint = torch.load(path+model_name)
model_corrections.load_state_dict(checkpoint['model_state_dict'])
model_corrections.eval()

# Cox fit params
β = [-1.51607591,1.94268716,-0.53678806]

def ode_blended(t,sol,pbar,state): 
    ode_r = sol[:S]
    ode_x = sol[-2]+ode_r*cosφ
    ode_y = sol[-1]+ode_r*sinφ
    θs    = het(ode_x,ode_y)
    θ     = se.get_data(R=np.array(ode_r))[:,0]
    Gs    = np.array([0.5*θs[i]*F(0.5*θs[i]*(x+1))@W for i in range(S)])
    G     = np.array([0.5*θ[i]*F(0.5*θ[i]*(x+1))@W for i in range(S)])
    # Calculate Cox
    ode_v_cox = (σ/mu)*(G-Gs)/(β[0]-np.log(np.average(ode_r))+\
                               β[1]/f(θs,q)+\
                               β[2]/f(θ,q))
    # AI model for first harmonic
    θ   =  my_to_torch(get_first_harmonic(θ)).reshape(1,S,1).to(device)
    θs   =  my_to_torch(get_first_harmonic(θs)).reshape(1,S,1).to(device)
    v_first_harmonic = model_first_harmonic(torch.cat([θ,θs,grid],dim=-1))
    # AI model for corrections
    v_base     = my_to_torch(filter_higher_harmonics(ode_v_cox,16)).reshape(1,S,1).to(device) + v_first_harmonic
    v_corrections = model_corrections(torch.cat([my_to_torch(ode_r).reshape(1,S,1).to(device), v_base, grid],dim=-1))    
    # Total velocity normal to the contact line
    ode_v = filter_higher_harmonics((v_base[0,:,0]+v_corrections[0,:,0]).cpu().detach().numpy(),16)
    # Calculate the final time derivatives
    drho_dφ,_ = rdiff(ode_r)
    M     = np.sqrt(1+drho_dφ**2/ode_r**2)*ode_v
    Xdot  = 2*np.sum(ode_r**2*M*cosφ)/sum(ode_r**2)
    Ydot  = 2*np.sum(ode_r**2*M*sinφ)/sum(ode_r**2)
    rho_t =  M + drho_dφ/ode_r*(Ydot*cosφ-Xdot*sinφ)-(Xdot*cosφ+Ydot*sinφ)
    rho_t = filter_higher_harmonics(rho_t,16)
    return np.append(rho_t,[Xdot,Ydot])


with tqdm(total=100,unit="%") as pbar:    
    solution = solve_ivp(ode_blended, (0,t_end),\
                          ic,\
                          method='RK23',dense_output=True,atol=1e-6,rtol=1e-3,\
                          events=evpbar,args=[pbar,[0,t_end/100]])


#########
# Plots #
#########
def plot_arrays(times):
    for time in times:
        x_plot = solution.sol(t=time)[-2]+solution.sol(t=time)[:S]*cosφ
        y_plot = solution.sol(t=time)[-1]+solution.sol(t=time)[:S]*sinφ
        plt.plot(x_plot,y_plot,'r')
        tikzplot(plot_folder+f'/case{case:1d}_t_{float(time):06.3f}_AI.txt',np.append(x_plot,x_plot[0]),np.append(y_plot,y_plot[0]),style='pyred,thick',forget=False)
        #
        PATH = data_folder+f"/cl_t_{float(time):06.3f}.npy"
        cfd_x,cfd_y = np.load(PATH)
        plt.plot(cfd_x,cfd_y,'--k')
        tikzplot(plot_folder+f'/case{case:1d}_t_{float(time):06.3f}_CFD.txt',np.append(cfd_x,cfd_x[0]),np.append(cfd_y,cfd_y[0]),style='pyblue,thick,fill=pyblue,fill opacity=0.5',forget=False)
        #
    Area  = calculate_area(cfd_x,cfd_y) 
    cfd_x,cfd_y = get_cl(u,np.array([cfd_x,cfd_y]))
    error = improved_frechet(np.array([x_plot,y_plot]).T,np.array([cfd_x,cfd_y]).T,num_points=200,closed=True)*np.sqrt(np.pi/(4*Area))
    print(f'auxiliary error = {error:.3%}')

def plot_centroid():
    times  = np.arange(0,t_end+0.1,1)
    x_plot = []
    y_plot = []
    x_plot_ref = []
    y_plot_ref = []
    flag = True
    for time in times:
        x_plot.append(solution.sol(t=time)[-2])
        y_plot.append(solution.sol(t=time)[-1])
        PATH = data_folder+f"/cl_t_{float(time):06.3f}.npy"
        cfd_x,cfd_y = np.load(PATH)
        x_plot_ref.append(np.mean(cfd_x))
        y_plot_ref.append(np.mean(cfd_y))
    tikzplot(plot_folder+f'/case{case:1d}_centroid_t_{float(time):06.3f}_AI.txt',x_plot,y_plot,style='pyred,thick,dashed',forget=False)
    tikzplot(plot_folder+f'/case{case:1d}_centroid_t_{float(time):06.3f}_CFD.txt',x_plot_ref,y_plot_ref,style='thick',forget=False)
    _, _, arc_length =  arc_length_parametrization(np.array([x_plot_ref,y_plot_ref]).T)
    error = improved_frechet(np.array([x_plot,y_plot]).T,np.array([x_plot_ref,y_plot_ref]).T,num_points=200)/arc_length
    print(f'auxiliary error on centroid = {error:.3%}')
        
def plot_frechet_in_time():
    times  = np.arange(0,t_end+0.1,1)
    frechet_plot = []
    for time in times:
        x_plot = solution.sol(t=time)[-2]+solution.sol(t=time)[:S]*cosφ
        y_plot = solution.sol(t=time)[-1]+solution.sol(t=time)[:S]*sinφ
        PATH = data_folder+f"/cl_t_{float(time):06.3f}.npy"
        cfd_x,cfd_y = np.load(PATH)
        Area  = calculate_area(cfd_x,cfd_y) 
        cfd_x,cfd_y = get_cl(u,np.array([cfd_x,cfd_y]))
        frechet_plot.append(improved_frechet(np.array([x_plot,y_plot]).T,np.array([cfd_x,cfd_y]).T,num_points=200)*np.sqrt(np.pi/(4*Area)))
        
    tikzplot(plot_folder+f'/case{case:1d}_frechet_t_{float(time):06.3f}_AI.txt',times,frechet_plot,style='pyred,thick,dashed',forget=False)
        #
 

size_title  = 20
size_ticks  = 15
size_legend = 15
size_line   = 1.5
plt.clf()
# DRAW SURFACE
fig, ax = plt.subplots(figsize=(5,5))
cmap = LinearSegmentedColormap.from_list('custom greys', [(1,1,1),(0.6,0.6,0.6)], N=256)
plt.axis('equal')
plt.xlim([0,1])
plt.ylim([0,1])
xf,yf = np.meshgrid(np.linspace(0,1,200),np.linspace(0,1,200))
Gf = het(xf,yf)
ax.imshow(Gf,extent=[0,1,0,1],cmap=cmap,origin='lower')
# DRAW LINES
plot_arrays(plot_times)
plot_centroid()
plot_frechet_in_time()
#
plt.xticks(fontsize = size_ticks)
plt.yticks(fontsize = size_ticks)
plt.savefig(plot_folder+'/case'+str(case)+'_figure.png',bbox_inches='tight',transparent=True,format='png',dpi=300)
# plt.clf()

