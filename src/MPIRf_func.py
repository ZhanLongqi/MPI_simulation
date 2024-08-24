from numba import cuda
import numpy as np
import math
from Phantom import phantom3d
import matplotlib.pyplot as plt
from rich.progress import track
from vedo import Volume,Plotter,show
from commonFunc import Kaczmarz
import os
from scipy.ndimage import rotate

class simulation_args:
    def __init__(self):
        self.shape = np.array([1,1,1])*2**8
        self.num_angle = 30 
        self.num_ffl = 2**8
        self.ffl_step = self.shape[0] / self.num_ffl
        self.beta = 10e-1 
        self.showPSF = 0
        self.showPhantom = 0
        self.showDeltaKernel = 0
        self.need_convolution = False
        self.gradient = 45 / self.shape[0] 
        self.need_cache = False
        self.use_cache = False
        self.dest_path = os.path.join(os.path.dirname(__file__),'../res/') 
        
    def load_args(path2file):
        pass
        

@cuda.jit
def mpi_get_signal_3d(phantom,width,height,depth,radian,ffl_idx,gradient,beta,num_ffl,res_by_grid):
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
    ffl_step = width / num_ffl
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
    
@cuda.jit
def mpi_get_signal_2d_gpu(phantom,width,height,radian,gradient,beta,num_ffl,res_by_grid):

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
    
    
def simulation2d(args,phantom):

    res_by_grid_device = cuda.to_device(np.zeros((args.shape[0])))
    vis = cuda.to_device(np.zeros(args.shape))
    phantom_device = cuda.to_device(phantom)
    
    threads_per_block = (8, 8 ,16)
    blocks_per_grid_x = int(math.ceil(args.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(args.shape[1] / threads_per_block[1]))
    blocks_per_grid_z = int(math.ceil(args.num_angle * args.num_ffl / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y,blocks_per_grid_z)
    res = np.zeros((args.num_angle,args.num_ffl))
    

    if os.path.isfile('signal/signal.txt') and args.use_cache:
        with open('signal/signal.txt') as f:
            line = f.readline()
            degree = 0
            while line:
                res[degree,:] = [float(i) for i in line[:-2].split(' ')]
                degree = degree + 1
                line = f.readline()
    else:  
        for idx in track(range(args.num_angle),description="Simulating,please wait:"): 
            res_by_grid_device = cuda.to_device(np.zeros((args.num_ffl)))
            mpi_get_signal_2d_gpu[blocksPerGrid, threads_per_block](phantom_device,args.shape[0],args.shape[1], idx * np.pi / args.num_angle,args.gradient,args.beta,args.num_ffl,res_by_grid_device)
            cuda.synchronize()
            res[idx] = np.array(res_by_grid_device.copy_to_host().tolist())
            # print('{:02d}/{:02d} is finished\r'.format(idx,num_angle))

        if args.need_cache:
            with open('signal/signal.txt','w') as f:
                for i in range(args.num_angle):
                    for j in range(args.num_ffl):
                        f.write('{} '.format(res[i,j]))
                    f.write('\n')
    return res

def simulation3d(args:simulation_args,phantom):
    delta_phantom_device =  cuda.to_device(phantom.data)
    threads_per_block = (8, 8 ,16)
    blocks_per_grid_x = int(math.ceil(args.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(args.shape[1] / threads_per_block[1]))
    blocks_per_grid_z = int(math.ceil(args.num_angle * args.num_ffl / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y,blocks_per_grid_z)
    res = np.zeros((args.num_angle,args.num_ffl,args.num_ffl))

    for idx in track(range(args.num_angle),description="Simulating,please wait:"): 
        res_by_grid_device = cuda.to_device(np.zeros((args.num_ffl,args.num_ffl)))
        for ffl_idx in range(args.num_ffl * args.num_ffl):

            mpi_get_signal_3d[blocksPerGrid, threads_per_block](delta_phantom_device,args.shape[0],args.shape[1],args.shape[2],  idx *np.pi / args.num_angle,ffl_idx,args.gradient,args.beta,args.num_ffl,res_by_grid_device)
            cuda.synchronize()
        tmp = np.array(res_by_grid_device.copy_to_host().tolist())

        res[idx] = tmp
        del(res_by_grid_device)
    del(delta_phantom_device)

    if args.need_cache:
        with open('signal/signal.txt','w') as f:
            for i in range(num_angle):
                for j in range(num_ffl):
                    f.write('{} '.format(res[i,j]))
                f.write('\n')
    return res

def deconvolve1d(res,delta_kernel,num_ffl,num_angle):

    kernel = np.zeros((num_ffl,3*num_ffl))
    for i in range(num_ffl):
        kernel[i,i:i+num_ffl] = delta_kernel - min(delta_kernel)
    kernel = kernel[:,num_ffl//2:num_ffl//2*3]

    for degree in range(num_angle):
        res[degree,:] = Kaczmarz(kernel,res[degree,:],iterations=10,lambd=1e-20,shuffle=True)


def deconvolve2d(res,delta_kernel,num_ffl,num_angle):
    f = open('res/kernel.txt','w')
    f2 = open('res/sig.txt','w')
    kernel = np.zeros((num_ffl*num_ffl,2 * num_ffl,2 * num_ffl))
    for i in range(num_ffl * num_ffl):
        kernel[i, i //  num_ffl: i // num_ffl + num_ffl,i % num_ffl : i% num_ffl + num_ffl]  = delta_kernel
    kernel = kernel[:,num_ffl//2:num_ffl//2*3,num_ffl//2:num_ffl//2*3]
    kernel = kernel.reshape((num_ffl*num_ffl,num_ffl*num_ffl))
    
    for degree in range(num_angle):
        tmp = res[degree].reshape((num_ffl * num_ffl))
        res[degree,:] = Kaczmarz(np.fft.fft(kernel,axis=0),np.fft.fft(tmp),iterations=10,lambd=1e-20,shuffle=True).reshape((num_ffl,num_ffl))

    f.close()
    f2.close()