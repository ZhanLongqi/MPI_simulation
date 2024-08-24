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
shape = np.array([1,1])*2**5
num_angle = 30
num_ffl = shape[0]
beta = 8e-1
showPSF = 0
showPhantom = 0
showDeltaKernel = 0
need_convolution = False
gradient = 45 / shape[0]
need_new_res = True
dest_path = os.path.join(os.path.dirname(__file__),'../res/') 
@cuda.jit
def gpu_get_signal(phantom,width,height,radian,gradient,beta,res_by_grid):
    radian =  2*np.pi / num_angle * radian
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    idy = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    idz = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z
    if idx >= width or idy >= height or idz >=num_ffl:
        return
    ffl_dis_from_center = idz - num_ffl/2
    ffl_dir_x = np.cos(radian)
    ffl_dir_y = np.sin(radian)
    center2pos_x = width//2 + ffl_dis_from_center * -math.sin(radian) - idx
    center2pos_y = height//2 + ffl_dis_from_center * math.cos(radian) - idy
    len = center2pos_x* ffl_dir_x + center2pos_y* ffl_dir_y
    x = center2pos_x - len*ffl_dir_x
    y = center2pos_y - len*ffl_dir_y
    H = math.sqrt(x**2 + y**2) * gradient  + 1e-4
    cuda.atomic.add(res_by_grid,idz,beta * (1/(beta * H)**2 - 1/math.sinh(beta * H)**2) * phantom[idx,idy])
    
def simulation():
    delta_phantom = Phantom.phantom(type='delta',shape=shape,radius=1/30*shape[0])#phantom.get_sphere_phantom('single sphere',shape,1/30 * shape[0])
    double_sphere_phantom = Phantom.phantom(type='p',shape=shape)#phantom.get_p_phantom(shape=shape)#phantom.get_p_phantom(shape)phantom.get_png_phantom('icon.png')#phantom.get_sphere_phantom('double sphere',shape,65)

    res_by_grid_device = cuda.to_device(np.zeros((shape[0])))
    vis = cuda.to_device(np.zeros(shape))
    delta_phantom_device = cuda.to_device(delta_phantom.data)
    double_sphere_phantom_device = cuda.to_device(double_sphere_phantom.data)
    
    threads_per_block = (8, 8 ,16)
    blocks_per_grid_x = int(math.ceil(shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(shape[1] / threads_per_block[1]))
    blocks_per_grid_z = int(math.ceil(num_angle * num_ffl / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y,blocks_per_grid_z)
    res = np.zeros((num_angle,num_ffl))
    
    gpu_get_signal[blocksPerGrid, threads_per_block](delta_phantom_device,shape[0],shape[1],0,gradient,beta,res_by_grid_device)
    delta_kernel = np.array(res_by_grid_device.copy_to_host().tolist())
    plt.plot(np.arange(shape[0]),delta_kernel)
    plt.savefig(dest_path+'delta_kernel.png')
    plt.close()
    if os.path.isfile('signal/signal.txt') and not need_new_res:
        with open('signal/signal.txt') as f:
            line = f.readline()
            degree = 0
            while line:
                res[degree,:] = [float(i) for i in line[:-2].split(' ')]
                degree = degree + 1
                line = f.readline()
    else:  
        for idx in track(range(num_angle),description="Simulating,please wait:"): 
            res_by_grid_device = cuda.to_device(np.zeros((num_ffl)))
            gpu_get_signal[blocksPerGrid, threads_per_block](double_sphere_phantom_device,shape[0],shape[1], idx,gradient,beta,res_by_grid_device)
            cuda.synchronize()
            res[idx] = np.array(res_by_grid_device.copy_to_host().tolist())
            # print('{:02d}/{:02d} is finished\r'.format(idx,num_angle))

        with open('signal/signal.txt','w') as f:
            for i in range(num_angle):
                for j in range(num_ffl):
                    f.write('{} '.format(res[i,j]))
                f.write('\n')
    return res,delta_kernel

def deconvolve(res,delta_kernel):
    kernel = np.zeros((num_ffl,3*num_ffl))
    for i in range(num_ffl):
        kernel[i,i:i+num_ffl] = delta_kernel - min(delta_kernel)
    kernel = kernel[:,num_ffl//2:num_ffl//2*3]
    plt.plot(np.arange(num_ffl),delta_kernel)
    plt.savefig(dest_path+'kernel.png')
    plt.close()
    for degree in range(num_angle):
        # plt.plot(np.arange(num_ffl), (res[degree,:] - np.min(res[degree,:]))/(np.max(res[degree,:] - np.min(res[degree,:])))*500)
        res[degree,:] = Kaczmarz(kernel,res[degree,:],iterations=5,lambd=1e-20,shuffle=True)
        # plt.plot(np.arange(num_ffl), (res[degree,:] - np.min(res[degree,:]))/(np.max(res[degree,:] - np.min(res[degree,:])))*500)
        # plt.savefig('sig_deconvolved.png')
        # plt.close()

res,delta_kernel = simulation()
deconvolve(res,delta_kernel)
proj = np.expand_dims(res,axis=1)
theta = np.linspace(0,2*np.pi,num_angle)
rot_center = res.shape[1]/2#tomopy.find_center(proj, theta, init=2500, ind=0, tol=0.5)
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='fbp', sinogram_order=False)
plt.imshow(recon[0, :, :],cmap='rainbow')
plt.savefig(dest_path + 'recon_deconvolved.png')