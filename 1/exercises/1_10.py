import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style = 'darkgrid')

n_runs = 100000
n_coins = 1000
n_flips = 10
mu = 0.5

heads = np.random.binomial(n_flips, mu, size = (n_runs, n_coins))
nu = heads / n_flips

nu_1 = nu[:, 0]
random_indices = np.random.randint(0, n_coins, size = n_runs)
nu_rand = nu[np.arange(n_runs), random_indices]
nu_min = np.min(nu, axis = 1)

plt.figure(figsize = (12, 5))
sns.histplot(nu_1, color = 'dodgerblue', label = r'$\nu_1$ (First)', stat = 'probability', binwidth = 0.1, alpha = 0.4)
sns.histplot(nu_rand, color = 'yellow', label = r'$\nu_{rand}$ (Random)', stat = 'probability', binwidth = 0.1, alpha = 0.4, kde = False)
sns.histplot(nu_min, color = 'crimson', label = r'$\nu_{min}$ (Minimum)', stat = 'probability', binwidth = 0.1, alpha = 0.6)

plt.title(r'Distribution of Fraction of Heads ($\nu$) across 1,00,000 Runs', fontsize = 14)
plt.xlabel(r'Fraction of Heads ($\nu$)', fontsize = 12)
plt.ylabel('Probability', fontsize = 12)
plt.xlim(-0.05, 1.05)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()

epsilons = np.linspace(0, 1, 100)

p_1 = np.array([np.mean(np.abs(nu_1 - mu) > eps) for eps in epsilons])
p_rand = np.array([np.mean(np.abs(nu_rand - mu) > eps) for eps in epsilons])
p_min = np.array([np.mean(np.abs(nu_min - mu) > eps) for eps in epsilons])

hoeffding_bound = 2 * np.exp(-2 * (epsilons ** 2) * n_flips)

plt.figure(figsize = (10, 6))
plt.plot(epsilons, hoeffding_bound, 'k--', linewidth = 2.5, label = 'Hoeffding Bound ($2e^{-2\\epsilon^2 N}$)')
plt.plot(epsilons, p_1, color = 'purple', linewidth = 2, label = r'Empirical $c_1$')
plt.plot(epsilons, p_rand, color = 'yellow', linewidth = 2, linestyle = ':', label = r'Empirical $c_{rand}$')
plt.plot(epsilons, p_min, color = 'crimson', linewidth = 2, label = r'Empirical $c_{min}$')

plt.title('Hoeffding Bound vs. Empirical Probabilities', fontsize = 14)
plt.xlabel(r'$\epsilon$', fontsize = 12)
plt.ylabel(r'$\mathbb{P}[|\nu - \mu| > \epsilon]$', fontsize = 12)
plt.ylim(-0.05, 2.05)
plt.legend(fontsize = 12)
plt.tight_layout()
plt.show()




