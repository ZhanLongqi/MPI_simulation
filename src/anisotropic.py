from MPIRf_func import *
import imageio
from PIL import Image
# p_phantom = phantom3d(type='col',shape=shape,radius=1/30*shape[0])#phantom3d(type='p',shape=shape,radius=3/30*shape[0])
num_delta_sample = 2**5
t = np.zeros(shape=(num_delta_sample,shape[1]))
for i in range(num_delta_sample):
    delta_phantom = phantom3d(type='delta',shape=shape,radius=1/30*shape[0],pos = (shape[0]//2,shape[1]//2,i * shape[2]//num_delta_sample))
    conv_kernel = simulation(delta_phantom.data,num_angle=1)
    t[i ,:] = conv_kernel[0,shape[0]//2,:]
    # plt.imshow(conv_kernel[0])
    # plt.savefig(dest_path + '/anisotropic_kernel/tmp_{:02d}.png'.format(i))
    # plt.close()
plt.imshow(t)
plt.savefig(dest_path + '/anisotropic_kernel/3d.png')
