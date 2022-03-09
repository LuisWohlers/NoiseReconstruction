from scipy.stats import kurtosis
import scipy
import numpy as np


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

def get_blocks(image, phi, row_parity, valid_block_index, M1, M2):
    block = im2col(VST(image.T, phi), M1, M2)
    block = block[int(row_parity) - 1::2, valid_block_index]
    return np.squeeze(block).T


def sort_blocks(image, phi, valid_block_index, M1, M2):
    block = get_blocks(image, phi, 2, valid_block_index, M1, M2)
    block = pca_svd_score(block)
    block = np.square(block)
    T = np.column_stack((valid_block_index, np.sum(block[:, 3::], axis=1)))
    T = np.array(T)
    T = T[np.argsort(T[:, 1])]
    return T[:, 0]


def VST(t, phi):
    a = np.cos(phi)
    b = np.sin(phi)
    if a > 2.220446049250313e-16:  # np.finfo(float).eps
        return (2 / a) * np.sqrt(a * t + b)
    else:
        return t / np.sqrt(b)


def compute_std(image, phi, tau, block_count, M1, M2):
    block = get_blocks(image, phi, 1, tau[0:block_count], M1, M2)
    latent = pca_svd_latent(block)
    return np.sqrt(latent[-1])


def optimize(func, a, b, accuracy, image, tau, block_count, M1, M2):
    opt_x = scipy.optimize.fminbound(func, a, b, args=(image, tau, block_count, M1, M2), xtol=accuracy, maxfun=500,
                                     full_output=0, disp=0)
    return opt_x


def pca_svd_score(x):
    x -= np.mean(x, axis=0)
    U, S, _ = scipy.linalg.svd(x, full_matrices=False)
    return U * S


def pca_svd_latent(x):
    x -= np.mean(x, axis=0)
    S = scipy.linalg.svd(x, full_matrices=False, compute_uv=False)
    return (S ** 2) / (x.shape[0] - 1)


def compute_kurtosis(phi, image, tau, block_count, M1, M2):
    block = get_blocks(image, phi, 1, tau[0:block_count], M1, M2)
    score = pca_svd_score(block)
    G = (kurtosis(score[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return G


def estimate_noise_parameters(image, blocksize):
    M1 = blocksize
    M2 = blocksize

    block = im2col(image.T, M1, M2)
    valid_block_index = np.arange(0, block.shape[1]).astype(int)

    phi = [0.0]
    sigma = [0.0]
    while (len(phi) < 20) and (len(phi) < 3 or np.min(np.abs(np.array(phi)[1:-1] - np.array(phi)[-1])) > 0.001):

        a = np.square(sigma[-1]) * np.cos(phi[-1])
        b = np.square(sigma[-1]) * np.sin(phi[-1])
        tau = sort_blocks(image, phi[-1], valid_block_index, M1, M2).astype(int)
        block_count = 5000
        curr_phi = 0
        curr_sigma = 0

        while block_count <= len(tau):

            opt_phi = optimize(compute_kurtosis, 0, np.pi / 2 - 0.001, 0.001, image, tau, block_count, M1,
                               M2)
            opt_kurtosis = compute_kurtosis(opt_phi, image, tau, block_count, M1, M2)
            if opt_kurtosis < 3 or curr_phi == 0:
                phi_converged = np.abs(opt_phi - curr_phi) < 0.0001
                curr_phi = opt_phi
                curr_sigma = compute_std(image, opt_phi, tau, block_count, M1, M2)
                if phi_converged:
                    break
            else:
                break
            block_count = block_count + 25000
        phi.append(curr_phi)
        sigma.append(curr_sigma)
    return a, b


