"""add Poisson-Gaussian noise to image"""
import numpy as np

def add_PoissonGaussianNoise_RGB(image,param_a,param_b,bits):
    """add Poisson-Gaussian noise with parameters a and b to input image"""
    if param_a == 0:
        output_y = image
    else:
        output_y = np.random.poisson(image/param_a)*param_a
    output_y = output_y + np.sqrt(param_b)*np.random.randn(output_y.shape[0],
                                    output_y.shape[1],output_y.shape[2])
    output_y[output_y>pow(2,bits)-1] = pow(2,bits)-1
    output_y[output_y<0] = 0
    return output_y.round(0)  

