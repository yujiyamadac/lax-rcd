import numpy as np
from scipy.stats import multivariate_normal

def kl_divergence(p_mean, p_cov, q_mean, q_cov):
    p_dist = multivariate_normal(mean=p_mean, cov=p_cov)
    q_dist = multivariate_normal(mean=q_mean, cov=q_cov)
    kl_div = np.mean(np.log(p_dist.pdf(p_mean) / q_dist.pdf(p_mean)))
    return kl_div

# Conjunto de dados 1
data_1 = {
    "mean": np.array([0.04468380999999999, 0.41643859]),
    "median": np.array([0.041161, 0.4503125]),
    "standard_deviation": np.array([0.005512592469175635, 0.12067816698295328]),
    "skewness": np.array([1.3432237564033542, -1.0002694187277232]),
    "kurtosis": np.array([0.3530438547165642, 0.09670778778405165])
}

# Conjunto de dados 2
data_2 = {
    "mean": np.array([0.04468380999999999, 0.41643859]),
    "median": np.array([0.041161, 0.4503125]),
    "standard_deviation": np.array([0.005512592469175635, 0.12067816698295328]),
    "skewness": np.array([1.3432237564033542, -1.0002694187277232]),
    "kurtosis": np.array([0.3530438547165642, 0.09670778778405165])
}

# Parâmetros das distribuições multivariadas
data_1_mean = np.concatenate([data_1["mean"], data_1["median"], data_1["standard_deviation"], data_1["skewness"], data_1["kurtosis"]])
data_2_mean = np.concatenate([data_2["mean"], data_2["median"], data_2["standard_deviation"], data_2["skewness"], data_2["kurtosis"]])

# Covariância, como os dados são univariados, a covariância é 0
data_cov = np.zeros((10, 10))

# Calculando a Divergência de Kullback-Leibler
kl_div = kl_divergence(data_1_mean, data_cov, data_2_mean, data_cov)
print("Divergência de Kullback-Leibler entre os conjuntos de dados 1 e 2:", kl_div)
