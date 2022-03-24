"""implementation of Pytaykh's noise parameter estimation algorithm"""
from scipy.stats import kurtosis
from scipy.linalg.lapack import sgesdd
from scipy.optimize import fminbound
import numpy as np

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

def get_valid_block_index(image, m_1, m_2):
    """extract valid block indices"""
    block = im_2col(image, m_1, m_2)
    minimums = np.min(block,axis=0)
    maximums = np.max(block,axis=0)
    equal_minmax = minimums==maximums
    invalid_grayvalue = np.unique(minimums[equal_minmax])
    blocks_ok = ((np.isin(block,invalid_grayvalue) + (block <= 0) + (block >= 255)) >= 1) == 0
    blocks_ok_rows = np.all(blocks_ok,axis=0)
    valid_block_index = np.where(blocks_ok_rows)
    return np.array(valid_block_index).astype(int).T

def get_blocks(image, phi, row_parity, valid_block_index, m_1, m_2):
    """extract block structure from image"""
    block = im_2col(vst(image, phi), m_1, m_2)
    block = block[int(row_parity) - 1::2, valid_block_index]
    return np.squeeze(block).T

def sort_blocks(image, phi, valid_block_index, m_1, m_2):
    """sort blocks according to Pyatykh's method"""
    block = get_blocks(image, phi, 2, valid_block_index, m_1, m_2)
    block = pca_svd_score(block)
    block = np.square(block)
    blocks_t = np.column_stack((valid_block_index, np.sum(block[:, 3::], axis=1)))
    blocks_t = np.array(blocks_t)
    blocks_t = blocks_t[np.argsort(blocks_t[:, 1])]
    return blocks_t[:, 0]

def vst(input_image, phi):
    """variance stabilizing transformation"""
    param_a = np.cos(phi)
    param_b = np.sin(phi)
    if param_a > 2.220446049250313e-16:  # np.finfo(float).eps
        return (2 / param_a) * np.sqrt(param_a * input_image + param_b)
    return input_image / np.sqrt(param_b)

def compute_std(image, phi, tau, block_count, m_1, m_2):
    """compute standard deviation from blocks"""
    block = get_blocks(image, phi, 1, tau[0:block_count], m_1, m_2)
    latent = pca_svd_latent(block)
    return np.sqrt(latent[-1])

def optimize(func, transformed_a, transformed_b, accuracy, image, tau, block_count, m_1, m_2):
    """optimize transformed parameters"""
    opt_x = fminbound(func, transformed_a, transformed_b,
                                    args=(image, tau, block_count, m_1, m_2),
                                    xtol=accuracy, maxfun=10000,
                                    full_output=0, disp=0)
    return opt_x

def pca_svd_score(data):
    """perform pca using scipy's lapack interface"""
    data -= np.mean(data, axis=0)
    u_mat, s_mat, _, _ = sgesdd(data, full_matrices=False)
    return u_mat * s_mat

def pca_svd_latent(data):
    """perform pca using scipy's lapack interface"""
    data -= np.mean(data, axis=0)
    _,s_mat,_,_ = sgesdd(data, full_matrices=False, compute_uv=False)
    return (s_mat ** 2) / (data.shape[0] - 1)

def compute_kurtosis(phi, image, tau, block_count, m_1, m_2):
    """compute kurtosis from blocks"""
    block = get_blocks(image, phi, 1, tau[0:block_count], m_1, m_2)
    score = pca_svd_score(block)
    result_g = (kurtosis(score[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return result_g

def estimate_noise_parameters(image, blocksize):
    """main noise parameter estimation function"""
    m_1 = blocksize
    m_2 = blocksize

    valid_block_index = get_valid_block_index(image,m_1,m_2)

    tau = sort_blocks(image, 0.0, valid_block_index, m_1, m_2).astype(int)
    block_count = 500
    curr_phi = 0.0
    curr_sigma = 0.0

    while block_count <= len(tau):
        opt_phi = optimize(compute_kurtosis, 0, np.pi / 2 - 0.001, 0.01,
                        image, tau, block_count, m_1,
                        m_2)
        opt_kurtosis = compute_kurtosis(opt_phi, image, tau, block_count, m_1, m_2)
        if opt_kurtosis < 3 or curr_phi == 0:
            phi_converged = np.abs(opt_phi - curr_phi) < 0.0005
            curr_phi = opt_phi
            curr_sigma = compute_std(image, opt_phi, tau, block_count, m_1, m_2)
            if phi_converged:
                break
        else:
            break
        block_count = block_count + 5000

    param_a = np.square(curr_sigma) * np.cos(curr_phi)
    param_b = np.square(curr_sigma) * np.sin(curr_phi)
    return param_a, param_b
