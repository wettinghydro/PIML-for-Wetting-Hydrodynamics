from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from bimdrop import *
import platform
import tiotrap
system = platform.system()
if system == 'Linux':
   import pexpect
elif system == 'Windows':
   import wexpect

# Fourier Interpolation
def fourmat(n,y):
    x = 2*np.pi*np.arange(n)/n
    w = (-1.)**np.arange(n)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = 1/np.tan(0.5*(x-y[:,None]))
        P = w*P
        P = P/np.sum(P,axis=1,keepdims=True)
        P[np.isnan(P)] = 1
    return P

# Cartesian to polar converter
def cart2pol(x, y):
    phi = np.arctan2(y, x) 
    phi[phi<0]+=2*np.pi
    return( np.sqrt(x**2 + y**2), np.abs(phi))

# Complex Fourier Series
def fs(F):
    n = int(len(F)/2)
    F = np.fft.rfft(F)
    F /= n
    F[0]*= 0.5
    return F[:-1]  # Cutoff frequency removed

# The evolver class
class evolver(object):
    def __init__(self,mesh=None, V=1, n=None, evolverpath='',order=2,\
                 tmpfile='drop.fe',R=[1],ic='BIM',H=None):
        
        if (mesh is None):
            raise Exception('"mesh" must be provided!')
        
        self.harmonics = None
        self.path = evolverpath
        self.ic = ic
        self.tmpfile = tmpfile
        self.V = V
        self.order = order

        # Load MATLAB mesh
        data = loadmat(mesh)
        self.tri = data['t']
        self.p = data['p']
        
        # Find where the contact line is
        ids = np.where(np.linalg.norm(self.p,axis=1)>0.99)[0]
        self.ncl = len(ids)
        
        # Assign n if not given
        if n is None:
            self.n = self.ncl
        else:
            self.n = n

        # Assign u (polar angle) and radius
        self.u = 2*np.pi*np.arange(self.n)/self.n
        self.R = R

        # Sort indices on contact line based on angle (ids start from 0) 
        _, θ = cart2pol(self.p[ids,0],self.p[ids,1])
        ids = list(ids[list(np.argsort(θ))])
        
        # Loop through triangulation to obtain edge and face connectivity
        list_of_edges = []
        list_of_faces = []
        list_of_edges_fixed = []
        e = 0

        # Convert triangulation
        for triangle in self.tri:
            f_row = [0,0,0,0]
            
            # Process edges
            for e1,e2,j in [[0,1,0],[1,2,1],[2,0,2]]:
                edg = [triangle[e1],triangle[e2]]
                Ledg = [edg[::-1],edg]
           
                # Check if edge was already assigned and determine its orientation
                s = [(2*Ledg.index(i)-1)*(list_of_edges.index(i)+1) for i in Ledg if i in list_of_edges]
                
                if s:   # Existing edge
                    prc = s[0]
                else:   # New edge append in list of edges and contact line edges
                    list_of_edges.append(edg)
                    list_of_edges_fixed.append(all(item-1 in ids for item in edg))
                    e += 1
                    prc = e   
                f_row[j] = prc
           
            # Append face
            list_of_faces.append(f_row)
        
        # List of edges and faces
        Nfaces = len(list_of_faces)
        E = ['{ids}   {v1}   {v2}   {state}\n'.format(ids=i+1, v1=list_of_edges[i][0],v2=list_of_edges[i][1],state=' boundary 1    fixed' if list_of_edges_fixed[i] else '') for i in range(len(list_of_edges))]
        F = ['{ids}   {e1}   {e2}   {e3}\n'.format(ids=i+1, e1=list_of_faces[i][0],e2=list_of_faces[i][1],e3=list_of_faces[i][2]) for i in range(Nfaces)]
        EdgesNFaces = ['\n\nedges\n']
        EdgesNFaces.extend(E)
        EdgesNFaces.append('\n\nfaces\n')
        EdgesNFaces.extend(F)
        EdgesNFaces.append('\n\nbodies\n')
        EdgesNFaces.append('1  '+np.array2string(np.arange(1,Nfaces+1),threshold=Nfaces)[1:-1].replace('\n', ' \ \n') + '     volume %f  density 1' % self.V)

        # Order of accuracy of computation
        order = ''
        if self.order<=-2:
            for iorder in np.arange(2,-self.order+1):
                order += 'lagrange %d; {g 2;  V;} 1; hessian;hessian;\n' % iorder
        
        if self.order >=2 :
            order = 'lagrange %d; {g 2;  V;} 1; hessian;hessian;\n' % self.order
        
        EdgesNFaces.extend(['\nread\n',\
                            'run:={\n',\
                            'quiet on;\n',\
                            '{g 2; V 10;} 2; U; g 2; V 10; hessian;\n',\
                            'do {oldenergy := total_energy; g 2}\n',\
                            'while (oldenergy-total_energy < 1e-10);\n',\
                            order,\
                            'foreach vertex vv where original and fixed do{\n',\
                            'Nx := vv.vertex_normal[1];\n',\
                            'Ny := vv.vertex_normal[2];\n',\
                            'normN:=sqrt(Nx*Nx+Ny*Ny);\n',\
                            'printf "%d\\t%1.10f\\t%1.10f\\t%1.10f\\n",vv.id-1, acos(vv.vertex_normal[3]), Nx/normN,Ny/normN;\n};}\n\n',\
                            'printH:={\n',\
                            'foreach vertex vv where original and not fixed do {\n',\
                            'printf "%d\\t%1.10f\\n",vv.id-1,vv.z;};}\n'])
        
        self.edgesfaces = EdgesNFaces 
        self.N = len(self.p)
        self.cline = ids
        self.interior =  list(set(range(self.N))-set(self.cline))
        self.r_, self.θ_ = cart2pol(self.p[:,0],self.p[:,1])
        self.cosθ = np.cos(self.θ_)
        self.sinθ = np.sin(self.θ_)
        self.D = fourmat(self.n,self.θ_)
        self.proc = None
        self.io = None
        self.X = []
        self.Y = []
        self.Hdata = None
        if H is None:
            self.H = np.zeros(self.N)
        else:
            self.H = np.array(H)

        # If BIM is used create an instance of bimdrop
        if self.ic=='BIM' or self.ic=='BIMSE':
            self.drop = bimdrop(n=self.n)
    
    # Intial Condition
    @property
    def R(self):
        return self._R
    
    @R.setter
    def R(self,value):
        if not callable(value):
            if len(value)==1:
                self._R = np.full(self.n,value,dtype='float64')
            elif len(value)==self.n:
                self._R = value
            else:
                self._R =0        
        else:
            self._R = value(self.u)

        # Fourier Series of R
        fsR = fs(self._R)
        FSR = np.array([fsR.real,-fsR.imag])
        FSR[abs(FSR)<1e-8] = 0 
        self.harmonics = FSR[:,:np.sum(np.abs(FSR),axis=0).nonzero()[0][-1]+1]

    # Make fe file
    def make_fe(self,R=None):
        
        if R is not None:
            self.R = R
        
        R_ = self.r_*(self.D@self.R)
        self.X = R_*np.cos(self.θ_)
        self.Y = R_*np.sin(self.θ_)
        
        if self.ic=='BIM' or self.ic=='BIMSE':
            self.drop.R = self.R
            self.drop.V = self.V
            self.H[self.interior] = self.drop.get_thickness(R_[self.interior], self.θ_[self.interior])

            if self.ic=='BIMSE':
                self.ic='H'
        elif self.ic=='cap':
            if np.min(self.R)==np.max(self.R):
                Rm = self.R[0] 
                θα = 2* np.arctan(2*np.sinh(np.arcsinh(3*self.V/np.pi/Rm**3)/3))
                self.H[self.interior] = Rm*(1/np.tan(θα) + np.sqrt(1/np.sin(θα)**2-self.p[self.interior,0]**2 - self.p[self.interior,1]**2))
            else:
                raise Exception('Invalid input with "cap" option')
            
        #Write surface evolver file
        # Note: at the moment, we re-load the run procedure, this can be avoided
        #       by using replace_load, but this will fix the filename of the file
        #       with the new vertices
        VV = [None]*self.N
        # Vertices for Surface evolver
        for ii in range(self.N):
            if ii not in self.cline:
                VV[ii] = '{ids}  {x}  {y}  {z}\n'.format(ids=ii+1,x=self.X[ii],y=self.Y[ii],z=self.H[ii])
            else:
                k = self.cline.index(ii)
                VV[ii] = '{ids} {u}.*pi/{n}.\tboundary 1\tfixed\n'.format(ids=ii+1,u=k,n=int(self.ncl/2))

        se_contents =  ['gravity_constant 0 \nINTEGRAL_ORDER 5\n','PARAMETER modes = {n}\n'.format(n=self.harmonics.shape[1])]
        se_contents.extend(['define harmonics real[2][modes] = { {' + \
                            np.array2string(self.harmonics[0,:],separator=',')[1:-1].replace('\n', ' \ \n')+'},{' + \
                            np.array2string(self.harmonics[1,:],separator=',')[1:-1].replace('\n', ' \ \n')+'} }\n\n'])
        se_contents.extend(['function real Radius(real phi){\n',\
                            'local output;\n',\
                            'output := harmonics[1][1];\n',\
                            'for (inx:=2; inx<= modes; inx++){\n',\
                            '   output+= harmonics[1][inx]*cos((inx-1.0)*phi) + harmonics[2][inx]*sin((inx-1.0)*phi);\n',\
                            '};\nreturn output;\n};\n\n'])
        se_contents.extend(['boundary 1 parameters 1 \n','x1:  Radius(p1)*cos(p1) \nx2:  Radius(p1)*sin(p1) \nx3:  0.0\n\nvertices\n'])
        se_contents.extend(VV)
        se_contents.extend(self.edgesfaces)

        with open(self.tmpfile, 'w') as f:
            f.writelines(se_contents)

    # Show triangulation
    def show_mesh(self):
        plt.triplot(self.X,self.Y,self.tri-1)

    # Show output surface
    def show_surf(self,ax=None,elev=None, azim=None):
        if ax is None:
            ax = plt.axes(projection='3d')
        ax.set_box_aspect((np.ptp(self.X),np.ptp(self.Y),np.ptp(self.H)))
        ax.plot_trisurf(self.X, self.Y, self.H, triangles=self.tri-1)
        ax.view_init(elev,azim) 

    # Get normals and angles on the contact line
    def get_data(self,R=None):
        # Spawn the process
        if self.io is None:
            self.io = tiotrap.TextIOTrap(store=True)
            if system == 'Linux':
                self.proc = pexpect.spawn(self.path+'evolver -Q', encoding='utf-8')
            elif system == 'Windows':
                self.proc = wexpect.spawn(self.path+'evolver -Q', encoding='utf-8')
            self.proc.logfile = self.io
        
        # Produce the fe file
        self.make_fe(R=R)  
        self.proc.expect_exact('Enter new datafile name (none to continue, q to quit): ')
        self.proc.sendline(self.tmpfile)

        # Run and clear 
        self.proc.expect('Enter command:')
        self.proc.sendline('run')
        self.io._entries = []

        # Once complete get the data
        self.proc.expect('Enter command:')
        # data = self.io.entries()
        data = self.proc.before[6:]
      
        # Droplet profile
        if self.ic == 'H':
            self.proc.sendline('printH')
            self.io._entries = []

            self.proc.expect('Enter command:')
            self.Hdata = self.io.entries()
            iH = np.array(self.Hdata.split()[1:-2]).astype(np.float32).reshape((-1,2))   
            self.H[iH[:,0].astype(int)] = iH[:,1]
            
        self.proc.sendline('q')

        # Normals and angles
        data_list = data.split()
        index=-2
        for idx, value in enumerate(data_list):
            if value == "line":
                index = idx 
        data = np.array(data_list[index+2:]).astype(np.float32).reshape((-1,4))   
        idx = data[:,0].astype(int)

        # Sort output
        id_sort = [list(idx).index(xx) for xx in self.cline]
        
        return data[id_sort,1:]

    # Kill Surface evolver
    def close(self):
        if self.proc is not None:
            self.proc.expect_exact('Enter new datafile name (none to continue, q to quit): ')
            self.proc.sendline('q')
            if system == 'Linux':
                self.proc.expect(pexpect.EOF)
            elif system == 'Windows':
                self.proc.expect(wexpect.EOF)
            self.proc.close()
