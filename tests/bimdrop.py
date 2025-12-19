# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import toeplitz
import warnings

class bimdrop(object):
    """
    A class for a drop object.
    
    Attributes
    ----------
    slip: float  [1e-3]
        Slip length.
    V: float [2*np.pi]
        Droplet volume. It can be a constant or a function of time.
    n: int [100]
        Number of discrete points for solving. It must be even.
    ic: float or callable [1.0]
        The initial condition for the radius.
    φ,u: array_like
        Polar angle.
    
    
    Methods
    -------
    R(ic), V(vol) 
        Sets the attributes for ic het and V
    
    get_angle()
        Returns the angle on the contact line
        
    get_thickness(R,T)
        Returns the thickness of the profile given R and T
    """
    def __init__(self,  V=2*np.pi, n = 100,  R = [1.0]):
        if (n%2 != 0):
            raise Exception("An even number of points are required")

        # Discretization
        self.n = n
        self.m = int(n/2-1)
        self.φ = 2*np.pi*np.arange(n)/n
        self.u = self.φ
        
        self.V = V
        self.R = R
        
        # Define a set of private variables for BIM
        self.__mm = np.arange(1,self.m+1)
        self.__m0 = np.arange(self.m)
        self.__j = np.arange(n)
        self.__W = toeplitz(1.0/self.n*np.sum(np.cos((self.__j[:,None]*self.__mm)*np.pi/(self.m+1))/self.__mm,axis=1) \
                           + 0.25*(-1.)**self.__j/(self.m+1)**2)
        self.__δφ = self.φ - self.φ[:,None]
        self.__sin_δφ = np.sin(self.__δφ)
        self.__cos_δφ = np.cos(self.__δφ)
        self.__sin2_δφ = np.sin(.5*self.__δφ)**2
        self.__k = np.fft.rfftfreq(self.n,d=1/self.n)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            self.__log_4_sin2_δφ = np.log(4*self.__sin2_δφ)
    
        self.__cosφ = np.cos(self.φ)
        self.__sinφ = np.sin(self.φ)
        
                         
    # Volume property
    @property
    def V(self):
        return self._V
    
    @V.setter
    def V(self,value):
        self._V = value 
                        
            
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
            self._R = value(self.φ)
        self.solution = None
   
   
    # Evaluate Radius via BIM
    def get_angle(self):
        Vo = self.V
        Ro = self.R
        # Scale data to avoid the Γ-contour
        self.__scale = 0.8/np.max(Ro)
        R = Ro*self.__scale
        V = Vo*self.__scale**3
        
        # Derivatives
        self.__Rhat = np.fft.rfft(R)
        self.__Rhat[-1] = 0
        self.__Ru = np.fft.irfft(1j*self.__k*self.__Rhat)
        self.__Ruu = np.fft.irfft(-self.__k**2*self.__Rhat)
        
        # Other variables
        self.__D = R**2 + self.__Ru**2
        self.__sqD = np.sqrt(self.__D)
        self.__RRo = R*R[:,None]
        self.__x_xo = (R-R[:,None])**2 + 4*self.__RRo*self.__sin2_δφ
        self.__x_dot_n = R**2/self.__sqD
        
        # Curvature (times sqD)
        self.__K = (R**2 + 2*self.__Ru**2 - R*self.__Ruu)/self.__D
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normal derivative of Green's function
            #   to obtain Gn multiply by  n/(0.5*π*sqD)
            self.__Gn = - (0.25/self.n)*(R**2 - self.__RRo*self.__cos_δφ \
                        - (R[:,None]*self.__Ru)*self.__sin_δφ)/self.__x_xo        
            np.fill_diagonal(self.__Gn,-(0.125/self.n)*self.__K)
            
            # Green's function
            #    2*π/m * (G - 0.5*log(4*sin(δφ)^2))
            self.__Gm = -0.5*(np.log(self.__x_xo)-self.__log_4_sin2_δφ)/self.n
            np.fill_diagonal(self.__Gm,-0.5*np.log(self.__D)/self.n)

        # Solve and determine the local angle
        self.__Wn = np.linalg.solve((self.__Gm + self.__W)*self.__sqD,0.125*R**2 + self.__Gn@(R**2))
        self.__kk = 4*V*self.n/(2*np.pi*np.sum((0.25*self.__x_dot_n-self.__Wn)*R**2*self.__sqD))
        
        return self.__kk*(0.5*self.__x_dot_n-self.__Wn)
 
    # get_thickness method
    def get_thickness(self,Rin,Tin):
        Vo, Ro = self.V, self._R
        
        # Scale data to avoid the Γ-contour
        self.__scale = 0.8/np.max(Ro)
        R = Ro*self.__scale
        V = Vo*self.__scale**3
        
        # Derivatives
        self.__Rhat = np.fft.rfft(R)
        self.__Rhat[-1] = 0
        self.__Ru = np.fft.irfft(1j*self.__k*self.__Rhat)
        self.__Ruu = np.fft.irfft(-self.__k**2*self.__Rhat)
        
        # Other variables
        self.__D = R**2 + self.__Ru**2
        self.__sqD = np.sqrt(self.__D)
        self.__RRo = R*R[:,None]
        self.__x_xo = (R-R[:,None])**2 + 4*self.__RRo*self.__sin2_δφ
        self.__x_dot_n = R**2/self.__sqD
        
        # Curvature (times sqD)
        self.__K = (R**2 + 2*self.__Ru**2 - R*self.__Ruu)/self.__D
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # Normal derivative of Green's function
            #   to obtain Gn multiply by  n/(0.5*π*sqD)
            self.__Gn = - (0.25/self.n)*(R**2 - self.__RRo*self.__cos_δφ \
                        - (R[:,None]*self.__Ru)*self.__sin_δφ)/self.__x_xo        
            np.fill_diagonal(self.__Gn,-(0.125/self.n)*self.__K)
            
            # Green's function
            #    2*π/m * (G - 0.5*log(4*sin(δφ)^2))
            self.__Gm = -0.5*(np.log(self.__x_xo)-self.__log_4_sin2_δφ)/self.n
            np.fill_diagonal(self.__Gm,-0.5*np.log(self.__D)/self.n)

        # Solve and determine Wn
        self.__Wn = np.linalg.solve((self.__Gm + self.__W)*self.__sqD,0.125*R**2 + self.__Gn@(R**2))
        self.__kk = 4*V*self.n/(2*np.pi*np.sum((0.25*self.__x_dot_n-self.__Wn)*R**2*self.__sqD))
         
        # Evaluate
        with warnings.catch_warnings():
            δφ = self.φ - Tin[:,None]
            Rin *=self.__scale
            sin_δφ = np.sin(δφ)
            cos_δφ = np.cos(δφ)
            sin2_δφ = np.sin(.5*δφ)**2
            RRo = R*Rin[:,None]
            Ix_xo = (R-Rin[:,None])**2 + 4*RRo*sin2_δφ

            H = 0.25*Rin**2 + (0.5/self.n)*((np.log(Ix_xo)*self.__sqD)@self.__Wn) \
                - ((0.25/self.n)*(R**2 - RRo*cos_δφ - (Rin[:,None]*self.__Ru)*sin_δφ)/Ix_xo )@(R**2)
        
        H[np.isnan(H)] = 0
        H[H>0] = 0     
        return -H*self.__kk/self.__scale