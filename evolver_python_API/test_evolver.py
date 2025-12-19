from evolver import evolver
import numpy as np
import matplotlib . pyplot as plt

# Initialize an evolver object with the provided mesh
ev = evolver (
    mesh = 'SEmesh128.mat' , # Mesh file
    V    = 3*np.pi/512 , # Droplet volume
    evolverpath = '' # Path to evolver executable
)

# Custom contact line shape as a function of the polar angle
R_cust = lambda u : 0.3*(1+0.2*np.sin (2*u)+0.3*np.cos(3*u))
ev.R   = R_cust(ev.u)

# Returns array with [ contact_angle , normal_x , normal_y ]
# at each point on the contact line
data = ev.get_data()
contact_angles = data[:,0]
normal_vectors = data[:,1:]

# Show mesh triangulation and draw normals
ev.show_mesh()
plt.gca().set_aspect('equal')
for i in range (128):
    plt.plot(ev.X[ev.cline[i]]+0.05*np.array([0,data[i,1]]),\
             ev.Y[ev.cline[i]]+0.05*np.array([0,data[i,2]]))

# Show 3 D surface
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111,projection='3d')
ev.show_surf(ax=ax,elev=30,azim =45)
#plt.show()
plt.savefig('SE_figure.png')
