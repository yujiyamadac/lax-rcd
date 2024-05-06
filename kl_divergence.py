import numpy as np
from scipy.stats import norm

def kl_divergence(p, q):
    """
    Calcula o KL Divergence entre duas distribuições de probabilidade.
    p: Distribuição de probabilidade P
    q: Distribuição de probabilidade Q
    """
    return np.sum(p * np.log(p / q))

# Exemplo hipotético:
# Vamos comparar duas distribuições normais com médias e desvios padrão diferentes.
mean_a, std_a = 5, 1
mean_b, std_b = 7, 1.5

# Gerando dados de amostra para as duas distribuições
samples_a = np.random.normal(mean_a, std_a, size=1000)
samples_b = np.random.normal(mean_b, std_b, size=1000)

# Calculando as médias e desvios padrão das amostras
mean_sample_a = np.mean(samples_a)
mean_sample_b = np.mean(samples_b)
std_sample_a = np.std(samples_a)
std_sample_b = np.std(samples_b)

# Criando distribuições normais com base nas médias e desvios padrão amostrados
dist_a = norm(mean_sample_a, std_sample_a)
dist_b = norm(mean_sample_b, std_sample_b)

# Calculando o KL Divergence entre as distribuições
kl_value = kl_divergence(dist_a.pdf(samples_a), dist_b.pdf(samples_a))

print(f"KL Divergence entre as distribuições: {kl_value:.4f}")
