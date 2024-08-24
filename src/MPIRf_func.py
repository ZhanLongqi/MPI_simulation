from numba import cuda
import numba
import numpy as np
import math
from Phantom import phantom3d
import matplotlib.pyplot as plt
from rich.progress import track
from vedo import Volume,Plotter,show
from commonFunc import Kaczmarz
import tomopy
import os
from scipy.ndimage import rotate
shape = np.array([1,1,1])*2**7
num_angle = 30 
num_ffl = 2**7#shape[0]
ffl_step = shape[0] / num_ffl
beta = 10e-1 
showPSF = 0
showPhantom = 0
showDeltaKernel = 0
need_convolution = False
gradient = 45 / shape[0] 
need_new_res = True
dest_path = os.path.join(os.path.dirname(__file__),'../res/') 
vedo_plt = Plotter()
@cuda.jit
def gpu_get_signal(phantom,width,height,depth,radian,ffl_idx,gradient,beta,res_by_grid):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    idy = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
    idz = cuda.threadIdx.z + cuda.blockDim.z * cuda.blockIdx.z
    posx = idx - width//2
    posy = idy - height//2
    posz = idz - depth//2
    
    if idx >= width or idy >= height or idz >=depth:
        return
    cos_theta = math.cos(radian)
    sin_theta = math.sin(radian)
    ffl_x = (ffl_idx % num_ffl * ffl_step  - width//2 ) * cos_theta
    ffl_y = (ffl_idx // num_ffl * ffl_step - height//2) * 1 
    ffl_z = (ffl_idx % num_ffl * ffl_step  - width//2 ) * sin_theta * -1 
    
    ffl2pos_x = posx - ffl_x
    ffl2pos_y = posy - ffl_y
    ffl2pos_z = posz - ffl_z
    
    # len = ffl2pos_x * math.sin(radian) + ffl2pos_z * math.cos(radian)
     
    u = cos_theta * ffl2pos_x - sin_theta * ffl2pos_z
    v = ffl2pos_y
    w = ffl2pos_x * sin_theta + ffl2pos_z * cos_theta
    H = math.sqrt(u**2 + v**2) * gradient / (1 + abs(w) * 1e-2)  + 1e-4

    cuda.atomic.add(res_by_grid,(ffl_idx//num_ffl,ffl_idx%num_ffl),  (1 / (beta * H)**2 - 1 / (math.sinh(beta * H)** 2)) * phantom[idx,idy,idz])
    
def simulation(phantom,num_angle):

    # vis = cuda.to_device(np.zeros(shape))
    delta_phantom_device =  cuda.to_device(phantom.data)
    threads_per_block = (8, 8 ,16)
    blocks_per_grid_x = int(math.ceil(shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(shape[1] / threads_per_block[1]))
    blocks_per_grid_z = int(math.ceil(num_angle * num_ffl / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y,blocks_per_grid_z)
    res = np.zeros((num_angle,num_ffl,num_ffl))

    for idx in track(range(num_angle),description="Simulating,please wait:"): 
        res_by_grid_device = cuda.to_device(np.zeros((num_ffl,num_ffl)))
        for ffl_idx in range(num_ffl * num_ffl):

            gpu_get_signal[blocksPerGrid, threads_per_block](delta_phantom_device,shape[0],shape[1],shape[2],  idx * 2*np.pi / num_angle,ffl_idx,gradient,beta,res_by_grid_device)
            cuda.synchronize()

        tmp = np.array(res_by_grid_device.copy_to_host().tolist())

        res[idx] = tmp
        del(res_by_grid_device)
    del(delta_phantom_device)

    # with open('signal/signal.txt','w') as f:
    #     for i in range(num_angle):
    #         for j in range(num_ffl):
    #             f.write('{} '.format(res[i,j]))
    #         f.write('\n')
    return res

def deconvolve1d(res,delta_kernel):

    kernel = np.zeros((num_ffl,3*num_ffl))
    for i in range(num_ffl):
        kernel[i,i:i+num_ffl] = delta_kernel - min(delta_kernel)
    kernel = kernel[:,num_ffl//2:num_ffl//2*3]

    # plt.plot(np.arange(num_ffl),delta_kernel)
    # plt.savefig(dest_path+'kernel.png')
    # plt.close()
    for degree in range(num_angle):
        res[degree,:] = Kaczmarz(kernel,res[degree,:],iterations=10,lambd=1e-20,shuffle=True)


def deconvolve2d(res,delta_kernel):
    f = open('res/kernel.txt','w')
    f2 = open('res/sig.txt','w')
    kernel = np.zeros((num_ffl*num_ffl,2 * num_ffl,2 * num_ffl))
    for i in range(num_ffl * num_ffl):
        kernel[i, i //  num_ffl: i // num_ffl + num_ffl,i % num_ffl : i% num_ffl + num_ffl]  = delta_kernel
    kernel = kernel[:,num_ffl//2:num_ffl//2*3,num_ffl//2:num_ffl//2*3]
    kernel = kernel.reshape((num_ffl*num_ffl,num_ffl*num_ffl))
    
    # for i in range(num_ffl):
    #     for j in range(num_ffl):
    #         f.write('{} '.format(kernel[i,j]))
    #     f.write('\n')
    # f.close()

    for degree in range(num_angle):
        tmp = res[degree].reshape((num_ffl * num_ffl))
        res[degree,:] = Kaczmarz(np.fft.fft(kernel,axis=0),np.fft.fft(tmp),iterations=10,lambd=1e-20,shuffle=True).reshape((num_ffl,num_ffl))
        # res[degree,:] = Kaczmarz(kernel,tmp,iterations=100,lambd=1e-20,shuffle=True).reshape((num_ffl,num_ffl))
        # for i in range(num_ffl):
        #     f2.write('{} '.format(res[degree,i]))
        # f2.write('\n')
    f2.close()