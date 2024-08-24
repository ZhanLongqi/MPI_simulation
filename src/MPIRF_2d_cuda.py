from numba import cuda
import numba
import numpy as np
import math
import Phantom as Phantom
import matplotlib.pyplot as plt
from rich.progress import track

from commonFunc import Kaczmarz
import tomopy
import os
from scipy.ndimage import rotate
from MPIRf_func import *

args = simulation_args()
phantom = Phantom.phantom(type='p',shape = args.shape[0:2],name = '../assets/icon.png')
res = simulation2d(args,phantom.data)
proj = np.expand_dims(res,axis=1)
theta = np.linspace(0,np.pi,args.num_angle)
rot_center = proj.shape[2]/2
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='fbp', sinogram_order=False)
plt.imshow(recon[0,:,:])
plt.savefig(args.dest_path + 'recon.png')