"""image noise reconstruction"""
import time
import numpy as np
import scipy
import NoiseParameterEstimation_Fast as est
from joblib import Parallel, delayed, parallel_backend



def im_2col(input_image, m_1, m_2):
    """like MatLab's im_2col, according to
    https://stackoverflow.com/questions/30109068/implement-matlabs-im_2col-sliding-in-python"""
    input_rows, input_cols = input_image.shape
    s_0, s_1 = input_image.strides
    n_rows = input_rows - m_1 + 1
    n_cols = input_cols - m_2 + 1
    shape = m_1, m_2, n_rows, n_cols
    strides = s_0, s_1, s_0, s_1

    out_view = np.lib.stride_tricks.as_strided(input_image, shape=shape, strides=strides)
    return out_view.reshape(m_1 * m_2, -1)


def im_2block(input_image, m_1, m_2):
    """like MatLab's im_2col, according to
    https://stackoverflow.com/questions/30109068/implement-matlabs-im_2col-sliding-in-python,
    keeping block view structure"""
    input_rows, input_cols = input_image.shape
    s_0, s_1 = input_image.strides
    n_rows = input_rows - m_1 + 1
    n_cols = input_cols - m_2 + 1
    shape = m_1, m_2, n_rows, n_cols
    strides = s_0, s_1, s_0, s_1

    out_view = np.lib.stride_tricks.as_strided(input_image, shape=shape, strides=strides)
    return out_view.reshape(m_1, m_2, n_rows * n_cols)


def blocks_2col(input_data, m_1, m_2):
    """like MatLab's im_2col, according to
    https://stackoverflow.com/questions/30109068/implement-matlabs-im_2col-sliding-in-python"""
    input_rows, input_cols, n_blocks = input_data.shape
    s_0, s_1, s_3 = input_data.strides
    n_rows = input_rows - m_1 + 1
    n_cols = input_cols - m_2 + 1
    shape = m_1, m_2, n_rows, n_cols, n_blocks
    strides = s_0, s_1, s_0, s_1, s_3

    out_view = np.lib.stride_tricks.as_strided(input_data, shape=shape, strides=strides)
    return out_view.reshape(m_1 * m_2, n_rows * n_cols, n_blocks)


def pca_svd_latent(data):
    """perform PCA"""
    data -= np.mean(data, axis=0)
    _, mat_s, _, _ = scipy.linalg.lapack.sgesdd(data, full_matrices=False, compute_uv=False)
    return (mat_s ** 2) / (data.shape[0] - 1)


def VSTab(input_image, param_a, param_b, sigma):
    """variance stabilizing transformation"""
    if param_a == 0 and param_b == 0:
        return input_image
    if param_a > 2.220446049250313e-16:  # np.finfo(float).eps
        return (2 * sigma / param_a) * np.sqrt(param_a * input_image + param_b)
    return input_image / np.sqrt(param_b)


def VSTreverse(input_image, param_a, param_b, sigma, bitdepth):
    """inverse variance stabilizing transformation"""
    image = (np.square((input_image * param_a) / (2 * sigma)) - param_b) / param_a
    image[image > (pow(2, bitdepth) - 1)] = pow(2, bitdepth) - 1
    image[image < 0] = 0
    return image.round()


#def lambda_adjust(regionsize, blocksize):
#    """return factor for adjusting estimated lambda values at a given window size"""
#    lambdas = np.zeros(10000)
#    for itr in range(0, 10000):
#        noisematrix = np.random.normal(0, 1, (regionsize, regionsize))
#        dataset = im_2col(noisematrix, blocksize, blocksize)
#        latent = pca_svd_latent(dataset.T)
#        lambdaP = latent[-1]
#        lambdas[itr] = lambdaP
#        s_lambda = (1 / np.mean(lambdas)).round(3)
#    return s_lambda

def lambda_adjust(regionsize, blocksize):
    """return factor for adjusting estimated lambda values at a given window size"""
    noisematrix = np.random.normal(0, 1, (regionsize, regionsize,10000))
    dataset = blocks_2col(noisematrix, blocksize, blocksize)
    latent = np.array(Parallel(n_jobs=4)(delayed(pca_svd_latent)\
                                                    (dataset[:, :, i].T) for i in \
                                                   range(10000)))
    lambdas = latent[:,-1]
    s_lambda = (1 / np.mean(lambdas)).round(3)
    return s_lambda


# def renoisePCA(image, regionsize, blocksize, sigma):
#    """noise reconstruction on input image, which needs to be variance stabilized already"""
#    imgpadded = np.pad(image, np.floor(regionsize / 2).astype(int), "symmetric")
#    image_rows = image.shape[0]
#    image_cols = image.shape[1]
#    renoised = np.empty((image_rows, image_cols))
#    s_lambda = lambda_adjust(regionsize, blocksize)
#    for itr_i in range(0, image_rows):
#        for itr_j in range(0, image_cols):
#            latent = pca_svd_latent(im_2col(imgpadded[itr_i:itr_i + regionsize,
#                            itr_j:itr_j + regionsize], blocksize, blocksize).T)
#            lambda_p_adjusted = (latent[-1] * s_lambda)
#            if lambda_p_adjusted < np.square(sigma):
#                vartoadd = abs(np.square(sigma) - lambda_p_adjusted)
#                renoised[itr_i, itr_j] = image[itr_i, itr_j] + np.random.normal(0, np.sqrt(vartoadd))
#            else:
#                renoised[itr_i, itr_j] = image[itr_i, itr_j]
#    return renoised

def renoisePCA(image, regionsize, blocksize, sigma):
    """noise reconstruction on input image, which needs to be variance stabilized already"""
    imgpadded = np.pad(image, np.floor(regionsize / 2).astype(int), "symmetric")
    image_rows = image.shape[0]
    image_cols = image.shape[1]
    reshaped_image = image.reshape(image_rows * image_cols, -1)
    renoised = reshaped_image.copy()
    s_lambda = lambda_adjust(regionsize, blocksize)
    windows_dataset = im_2block(imgpadded, regionsize, regionsize)
    windows_dataset = blocks_2col(windows_dataset, blocksize, blocksize)
    with parallel_backend('multiprocessing',n_jobs=-1):
        latent_per_block = np.array(Parallel(batch_size='auto')(delayed(pca_svd_latent)\
                                                        (windows_dataset[:, :, i]) for i in \
                                                       range(windows_dataset.shape[-1])))
    lambdas_p_adjusted = latent_per_block[:, -1] * s_lambda
    indices = np.where(lambdas_p_adjusted < np.square(sigma))[0]
    varstoadd = np.ones(lambdas_p_adjusted.shape) * np.abs(np.square(sigma)) - lambdas_p_adjusted
    renoised = renoised.flatten()
    renoised[indices] = renoised[indices] + np.random.normal(0, np.sqrt(varstoadd))
    return np.lib.stride_tricks.as_strided(renoised, shape=image.shape, strides=image.strides)


def reconstructNoise(originalimage, degradedimage, bitdepth, analyse_blocksize,
                     renoise_regionsize, renoise_blocksize, sigma):
    """noise parameter estimation on original image and reconstruction on degraded image"""
    overall_starttime = time.time()
    starttime = time.time()
    a_original, b_original = est.estimate_noise_parameters(originalimage, analyse_blocksize)
    endtime = time.time() - starttime
    print("Analysis of a and b params of original image finished in (seconds):")
    print(endtime)
    print("a param of original image:")
    print(a_original)
    print("b param of original image:")
    print(b_original)
    vst_degradedimage = VSTab(degradedimage, a_original, b_original, sigma).astype(float)
    starttime = time.time()
    renoised_vst = renoisePCA(vst_degradedimage, renoise_regionsize, renoise_blocksize, sigma)
    endtime = time.time() - starttime
    print("Renoise time:")
    print(endtime)
    finalimage = VSTreverse(renoised_vst, a_original, b_original, sigma, bitdepth)
    starttime = time.time()
    a_final, b_final = est.estimate_noise_parameters(finalimage, analyse_blocksize)
    endtime = time.time() - starttime
    print("Analysis of a and b params of final image finished in (seconds):")
    print(endtime)
    print("a param of final image:")
    print(a_final)
    print("b param of final image:")
    print(b_final)
    overall_endtime = time.time()
    print("Overall time in seconds: ")
    print(overall_endtime-overall_starttime)
    return finalimage, a_original, b_original, a_final, b_final


def reconstructNoise_RGB(originalimage, degradedimage, bitdepth, analyse_blocksize,
                         renoise_regionsize, renoise_blocksize,
                         sigma, param_a=-1, param_b=-1):
    """estimate noise on originalimage and reconstruct on degraded image, rgb"""
    starttime = time.time()
    finalimage = np.zeros(originalimage.shape)
    for itr in range(0, 3):
        if param_a == -1 and param_b == -1:
            param_a, param_b = est.estimate_noise_parameters(originalimage[:, :, itr],
                                                             analyse_blocksize)
        VSTdegradedimage = VSTab(degradedimage[:, :, itr], param_a, param_b, sigma).astype(float)
        renoisedVST = renoisePCA(VSTdegradedimage, renoise_regionsize, renoise_blocksize, sigma)
        finalimage[:, :, itr] = VSTreverse(renoisedVST, param_a, param_b, sigma, bitdepth).astype(np.uint8)
    endtime = time.time()
    print("time expired: " + str(endtime - starttime) + " seconds")

    return finalimage


def reconstructNoise_RGB_ab_perchannel(originalimage, degradedimage, bitdepth, analyse_blocksize,
                                       renoise_regionsize, renoise_blocksize,
                                       sigma, param_a=(-1, -1, -1), param_b=(-1, -1, -1)):
    """estimate noise on originalimage and reconstruct on degraded image, rgb, different params
    per channel"""
    starttime = time.time()
    finalimage = np.zeros(originalimage.shape)
    for itr in range(0, 3):
        if param_a[itr] == -1 and param_b[itr] == -1:
            param_a[itr], param_b[itr] = est.estimate_noise_parameters(originalimage[:, :, itr],
                                                                       analyse_blocksize)
        VSTdegradedimage = VSTab(degradedimage[:, :, itr],
                                 param_a[itr], param_b[itr], sigma).astype(float)
        renoisedVST = renoisePCA(VSTdegradedimage, renoise_regionsize, renoise_blocksize, sigma)
        finalimage[:, :, itr] = VSTreverse(renoisedVST, param_a[itr], param_b[itr], sigma, bitdepth).astype(np.uint8)
    endtime = time.time()
    print("time expired: " + str(endtime - starttime) + " seconds")

    return finalimage
