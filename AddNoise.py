import numpy as np

def AddPoissonGaussianNoise_RGB(image,a,b,bits):
    if a == 0:
        y = image
    else:
        y = np.random.poisson(image/a)*a
    y = y + np.sqrt(b)*np.random.randn(y.shape[0],y.shape[1],y.shape[2])
    y[y>pow(2,bits)-1] = pow(2,bits)-1
    y[y<0] = 0
    return y.round(0)