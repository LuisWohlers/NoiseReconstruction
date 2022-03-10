import scipy
import numpy as np
from math import sqrt
import NoiseParameterEstimation as est
import time


def im2col(A, M1, M2):
    # according to https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    A = A.T
    m, n = A.shape
    s0, s1 = A.strides
    nrows = m - M1 + 1
    ncols = n - M2 + 1
    shp = M1, M2, nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(M1 * M2, -1)


def pca_svd_latent(x):
    x -= np.mean(x, axis=0)
    S = scipy.linalg.svd(x, full_matrices=False, compute_uv=False)
    return (S ** 2) / (x.shape[0] - 1)


def VSTab(t, a, b, sigma):
    if a == 0 and b == 0:
        return t
    if a > 2.220446049250313e-16:  # np.finfo(float).eps
        return (2 * sigma / a) * np.sqrt(a * t + b)
    else:
        return t / np.sqrt(b)


def VSTreverse(y, a, b, sigma, bitdepth):
    image = (np.square((y * a) / (2 * sigma)) - b) / a
    image[image > (pow(2, bitdepth) - 1)] = pow(2, bitdepth) - 1
    image[image < 0] = 0
    return image.round()


def lambda_adjust(regionsize, blocksize):
    lambdas = np.zeros(10000)
    for j in range(0, 10000):
        noisematrix = np.random.normal(0, 1, (regionsize, regionsize))
        dataset = im2col(noisematrix, blocksize, blocksize)
        latent = pca_svd_latent(dataset.T)
        lambdaP = latent[-1]
        lambdas[j] = lambdaP
        s_lambda = (1 / np.mean(lambdas)).round(3)
    return s_lambda


def renoisePCA(image, regionsize, blocksize, sigma):
    imgpadded = np.pad(image, np.floor(regionsize / 2).astype(int), "symmetric")
    M = image.shape[0]
    N = image.shape[1]
    renoised = np.empty((M, N))
    s_lambda = lambda_adjust(regionsize, blocksize)
    for i in range(0, M):
        for j in range(0, N):
            latent = pca_svd_latent(im2col(imgpadded[i:i + regionsize, j:j + regionsize], blocksize, blocksize).T)
            lambda_p_adjusted = (latent[-1] * s_lambda)
            if lambda_p_adjusted < np.square(sigma):
                vartoadd = abs(np.square(sigma) - lambda_p_adjusted)
                renoised[i, j] = image[i, j] + np.random.normal(0, sqrt(vartoadd))
            else:
                renoised[i, j] = image[i, j]
    return renoised


def reconstructNoise(originalimage, degradedimage, bitdepth, analyse_blocksize, renoise_regionsize, renoise_blocksize,
                     sigma):
    starttime = time.time()
    a_original, b_original = est.estimate_noise_parameters(originalimage, analyse_blocksize)
    endtime = time.time() - starttime
    print("Analysis of a and b params of original image finished in (seconds):")
    print(endtime)
    print("a param of original image:")
    print(a_original)
    print("b param of original image:")
    print(b_original)

    VSTdegradedimage = VSTab(degradedimage, a_original, b_original, sigma).astype(float)
    starttime = time.time()
    renoisedVST = renoisePCA(VSTdegradedimage, renoise_regionsize, renoise_blocksize, sigma)
    endtime = time.time() - starttime
    print("Renoise time:")
    print(endtime)
    finalimage = VSTreverse(renoisedVST, a_original, b_original, sigma, bitdepth)
    starttime = time.time()
    a_final, b_final = est.estimate_noise_parameters(finalimage, analyse_blocksize)
    endtime = time.time() - starttime
    print("Analysis of a and b params of final image finished in (seconds):")
    print(endtime)
    print("a param of final image:")
    print(a_final)
    print("b param of final image:")
    print(b_final)

    return finalimage, a_original, b_original, a_final, b_final

def reconstructNoise_RGB(originalimage, degradedimage, bitdepth, analyse_blocksize, renoise_regionsize, renoise_blocksize,
                     sigma,a=-1,b=-1):

    starttime = time.time()
    finalimage = np.zeros(originalimage.shape)
    for i in range(0,3):
        if a == -1 and b == -1:
            a_original, b_original = est.estimate_noise_parameters(originalimage[:,:,i],analyse_blocksize)
        VSTdegradedimage = VSTab(degradedimage[:,:,i], a_original, b_original, sigma).astype(float)
        renoisedVST = renoisePCA(VSTdegradedimage,renoise_regionsize,renoise_blocksize,sigma)
        finalimage[:,:,i] = VSTreverse(renoisedVST,a_original,b_original,sigma,bitdepth)
    endtime = time.time()
    print("time expired: " + endtime-starttime + " seconds")

    return finalimage

