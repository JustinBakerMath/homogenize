import numpy as np
from scipy.ndimage import rotate

def gauss2D(x,y,mu=[0,0],sig=[.5,.5],theta_deg=0,scale=1):

    theta_rad = 2*np.pi*theta_deg/360
    y = y[:,np.newaxis]

    sx = sig[0]
    sy = sig[-1]

    x0 = mu[0]
    y0 = mu[-1]

    a=np.cos(theta_rad)*x-np.sin(theta_rad)*y
    b=np.sin(theta_rad)*x+np.cos(theta_rad)*y

    a0=np.cos(theta_rad)*x0-np.sin(theta_rad)*y0
    b0=np.sin(theta_rad)*x0+np.cos(theta_rad)*y0

    return scale*np.exp(-(((a-a0)**2)/(2*(sx**2)) + ((b-b0)**2) /(2*(sy**2))))

def square(xx,yy,mu,sig,theta_deg=0,scale=1):
  
    mask_left = xx<(mu[0]-sig[0])
    mask_right = (mu[0]+sig[0])<xx
    mask_down = yy<(mu[-1]-sig[-1])
    mask_up = (mu[-1]+sig[-1])<yy

    grid = np.ones_like(xx)
    grid[mask_left]=0
    grid[mask_right]=0
    grid[mask_up]=0
    grid[mask_down]=0

    grid = rotate(grid,theta_deg,reshape=False)

    return scale*grid


def circle(xx,yy,mu,rad,scale=1):
  
    mask = (xx-mu[0])**2+(yy-mu[-1])**2<rad**2

    grid = np.zeros_like(xx)
    grid[mask]=1

    return scale*grid




