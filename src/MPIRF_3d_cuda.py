from MPIRf_func import *
import vtk
p_phantom = phantom3d(type='p',shape=shape,radius=3/30*shape[0])
delta_phantom = phantom3d(type='delta',shape=shape,radius=1/30*shape[0])

# conv_kernel = simulation(delta_phantom.data,num_angle=30)
res = simulation(p_phantom.data,num_angle=num_angle)

proj = res
theta = np.linspace(0,2*np.pi,num_angle)
rot_center = proj.shape[1]/2#tomopy.find_center(proj, theta, init=2500, ind=0, tol=0.5)
# for i in range(num_ffl):
#     deconvolve(proj[:,:,i],delta_kernel)
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='fbp', sinogram_order=False)

with open(dest_path + 'sig.txt','w') as f:
    for i in range(recon.shape[0]):
        for j in range(recon.shape[1]):
            for k in range(recon.shape[2]):
                f.write('{} '.format(recon[i,j,k]))

# plt.plot(np.arange(num_ffl),recon[num_ffl//2,num_ffl//2,:]*500)
plt.imshow(recon[num_ffl//2,:,:],cmap='binary')
plt.savefig(dest_path + 'recon_deconv.png')
