from my_pyhessian import hessian
import numpy as np
from torch.nn import CrossEntropyLoss


def _gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

def _density_generate(eigenvalues, weights, num_bins=10000, sigma_squared=1e-5, overhead=0.01):
    """
    :param eigenvalues:
    :param weights:
    :param num_bins:
    :param sigma_squared:
    :param overhead:
    :return:
    """

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = _gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids

def _get_ratio(t, density_eigen, density_weight):
    """
    get pr from the probability density of eigens
    :param t:
    :param density_eigen:
    :param density_weight:
    :return:
    """
    density, grids = _density_generate(density_eigen, density_weight)
    sum = 0
    for i in range(len(grids - 1) - 1):
        if grids[i + 1] <= t:
            sum += density[i] * (grids[i + 1] - grids[i])
            i += 1
    ratio = 1 - sum
    return ratio

def get_ratio(model, batch, device="cpu", imc_t=0.0):
    """
    get pr from the probability density of eigens
    :param model:
    :param dataset:
    :param idxs:
    :param device:
    :return:
    """
    print("=== get pr ===")
    model.to(device)
    model.eval()
    model.zero_grad()
    batch = tuple(t.to(device) for t in batch)
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
    }
    
    inputs["seq_lengths"] = batch[3]
    inputs['labels'] = batch[0]
    if inputs['labels'] is not None:
        loss_fct = CrossEntropyLoss()
    hessian_comp = hessian(model, loss_fct, data=(inputs, inputs['labels']), cuda=True if device != "cpu" else False)
            
    density_eigen, density_weight = hessian_comp.density(iter=50, n_v=1)
    inc = 0.1
    while True:
        t = imc_t
        ratios = []
        flag = False
        while True:
            ratio = _get_ratio(t, density_eigen=density_eigen, density_weight=density_weight)
            ratios.append(ratio)
            if ratio < ratios[0] / 2:
                break
            if len(ratios) >= 4:
                if abs(ratios[-1] - ratios[-2]) < 0.005 and abs(ratios[-2] - ratios[-3]) < 0.005 and abs(ratios[-3] - ratios[-4]) < 0.005:
                    flag = True
                    break
            t += inc
        if flag:
            break
        inc /= 2
    return ratios[-1]