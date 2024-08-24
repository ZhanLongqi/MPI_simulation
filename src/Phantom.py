import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
class phantom:

    def get_phantom(self,type ='delta sphere',shape=(256,256),radius=1,name='default'):
        if type == 'delta':
            return get_sphere_phantom_2d(type,shape = shape,radius=radius)
        elif type == 'double sphere':
            return get_sphere_phantom_2d(type,shape=shape,radius=radius)
        elif type == 'p':
            return get_p_phantom_2d(shape=shape)
        elif type == 'png':
            if os.path.isfile(name):
                return get_png_phantom_2d(name=name)
            else:
                return np.zeros(shape)
        else:
            raise Exception('Error: unsupported type!')

    def __init__(self,type='delta',shape=(128,128),radius=1,name='dafault'):
        self.data = self.get_phantom(type,shape,radius,name)

def get_sphere_phantom_2d(arg,shape,radius):
    phantom = np.zeros(shape)
    if(arg == 'double sphere'):
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.sqrt((i - shape[0]//8*3)**2 + (j - shape[1]//8*3)**2) < radius or np.sqrt((i - shape[0]//8 * 5)**2 + (j - shape[1]//8 * 5)**2) < radius:
                    phantom[i,j] = 1
                else:
                    phantom[i,j] = 0
    elif arg == 'delta':
        for i in range(shape[0]):
            for j in range(shape[1]):
                if np.sqrt((i - shape[0]//2)**2 + (j - shape[1]//2)**2) < radius:
                    phantom[i,j] = 1
                else:
                    phantom[i,j] = 0
    return phantom

def get_p_phantom_2d(shape=(512,512)):
    concentration = 10.0
    # 创建一个全零的二维数组
    phantom = np.zeros(shape, dtype=np.uint8)
    p_thickness = int(shape[0]/15)
    # 定义P的大小
    p_height = shape[0] // 2
    p_width = shape[1] // 3
    
    # 设置P的起始位置
    p_start_x = shape[0] // 4
    p_start_y = shape[1] // 4

        # 绘制P的垂直部分
    phantom[p_start_y:p_start_y + p_height, p_start_x:p_start_x + p_thickness] = concentration
    
    # 绘制P的顶部横向部分
    phantom[p_start_y:p_start_y + p_thickness, p_start_x:p_start_x + p_width] = concentration
    
    # 绘制P的中间横向部分
    phantom[p_start_y + p_height // 2:p_start_y + p_height // 2 + p_thickness, p_start_x:p_start_x + p_width] = concentration
    
    # 绘制P的右边部分
    phantom[p_start_y:p_start_y + p_height // 2, p_start_x + p_width - p_thickness:p_start_x + p_width] = concentration
    
    return phantom

def get_png_phantom_2d(name,shape=(512,512)):
    Img = Image.open(name)
    Img = Img.resize(shape).convert('L')
    Img = np.array(Img)
    Img = 1 - (Img - np.min(Img)) / (np.max(Img) - np.min(Img))
    plt.imshow(Img)
    # plt.show()
    plt.savefig('tmp.png')
    return Img

class phantom3d:

    def get_phantom(self,type ='delta sphere',shape=(256,256,256),radius=1,name='default',pos=(0,0,0)):
        if type == 'delta':
            return get_sphere_phantom_3d(arg=type,shape = shape,radius=radius,pos=pos)
        elif type == 'double sphere':
            return get_sphere_phantom_3d(type,shape=shape,radius=radius)
        elif type == 'p':
            return get_p_phantom_3d(shape=shape)
        elif type == 'png':
            if os.path.isfile(name):
                return get_png_phantom_2d(name=name)
            else:
                return np.zeros(shape)
        elif type == 'square':
            return get_square_phantom_3d(shape=shape)
        elif type == 'col':
            return get_col_phantom_3d(radius = radius, shape=shape)
        else:
            raise Exception('Error: unsupported type!')

    def __init__(self,type='delta',shape=(128,128),radius=1,name='dafault',pos=(0,0,0)):
        self.data = self.get_phantom(type,shape,radius,name,pos=pos)

def get_sphere_phantom_3d(arg,shape,radius,pos):
    phantom = np.zeros(shape)
    if(arg == 'double sphere'):
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if np.sqrt((i - shape[0]//8*3)**2 + (j - shape[1]//8*3)**2 + (k - shape[2]//2) ** 2) < radius or np.sqrt((i - shape[0]//8 * 5)**2 + (j - shape[1]//8 * 5)**2 + (k - shape[2]//2) ** 2) < radius:
                        phantom[i,j,k] = 1
                    else:
                        phantom[i,j,k] = 0
    elif arg == 'delta':
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if np.sqrt((i - pos[0])**2 + (j - pos[1])**2 + (k - pos[2]) ** 2) < radius:
                        phantom[i,j,k] = 1
                    else:
                        phantom[i,j,k] = 0
    return phantom

def get_p_phantom_3d(shape=(512,512,512)):
    concentration = 10.0
    # 创建一个全零的二维数组
    phantom = np.zeros(shape, dtype=np.uint8)
    p_thickness = int(shape[0]/15)
    # 定义P的大小
    p_height = shape[0] // 10 * 6
    p_width = shape[1] // 3
    
    # 设置P的起始位置
    p_start_x = shape[0] // 4
    p_start_y = shape[1] // 4

        # 绘制P的垂直部分
    phantom[p_start_y:p_start_y + p_height,shape[0]//8*3:shape[0]//8*5, p_start_x:p_start_x + p_thickness] = concentration
    
    # 绘制P的顶部横向部分
    phantom[p_start_y:p_start_y + p_thickness,shape[0]//8*3:shape[0]//8*5, p_start_x:p_start_x + p_width] = concentration
    
    # 绘制P的中间横向部分
    phantom[p_start_y + p_height // 2:p_start_y + p_height // 2 + p_thickness,shape[0]//8*3:shape[0]//8*5, p_start_x:p_start_x + p_width] = concentration
    
    # 绘制P的右边部分
    phantom[p_start_y:p_start_y + p_height // 2,shape[0]//8*3:shape[0]//8*5, p_start_x + p_width - p_thickness:p_start_x + p_width] = concentration

    
    return phantom

def get_square_phantom_3d(shape):
    return np.zeros(shape)

def get_col_phantom_3d(radius,shape):
    phantom = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if np.sqrt((i - shape[0]//2)**2 + (j - shape[1]//2)**2) < radius:
                    phantom[i,j,k] = 1
                else:
                    phantom[i,j,k] = 0
    return phantom

