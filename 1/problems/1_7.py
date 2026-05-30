import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style = 'darkgrid')

epsilons = np.linspace(0, 1, 2000)

def get_exact_max_prob(eps):

    if eps < 1/6:
        return 231 / 256

    elif eps < 1/3:
        return 399 / 1024

    elif eps < 0.5:
        return 63 / 1024

    else:
        return 0.0


exact_curve = [get_exact_max_prob(e) for e in epsilons]

hoeffding_single = 2 * np.exp(-2 * 6 * epsilons**2)
hoeffding_union = 4 * np.exp(-2 * 6 * epsilons**2)

plt.figure(figsize = (10, 6.5))

plt.step(epsilons, exact_curve, label = r'Exact $P[\max_i |\nu_i - \mu_i| > \epsilon]$', 
         color = 'crimson', linewidth = 2.5, where = 'post')

plt.plot(epsilons, hoeffding_single, label = r'Single-Coin Hoeffding Bound ($2e^{-12\epsilon^2}$)', 
         color = 'dodgerblue', linestyle = '--', linewidth = 2)
plt.plot(epsilons, hoeffding_union, label = r'Multi-Coin Union Hoeffding Bound ($4e^{-12\epsilon^2}$)', 
         color = 'forestgreen', linestyle = ':', linewidth = 2.5)

plt.title('Exact Maximum Error Deviation vs. Hoeffding Bounds ($N = 6, M = 2$)', fontsize = 14, pad = 15)
plt.xlabel(r'Deviation Tolerance ($\epsilon$)', fontsize = 12)
plt.ylabel('Probability', fontsize = 12)

plt.xlim(-0.02, 1.02)
plt.ylim(-0.05, 1.5)
plt.axvline(0.5, color = 'purple', alpha = 0.5, linestyle = '--', label = r'Max Possible Deviation ($\epsilon = 0.5$)')

plt.legend(loc = 'upper right', frameon = True, fontsize = 11)
plt.tight_layout()
plt.show()






