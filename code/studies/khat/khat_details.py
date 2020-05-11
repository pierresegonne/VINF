import matplotlib.pyplot as plt
import numpy as np
import seaborn as snb
from scipy.integrate import quad

snb.set_style('darkgrid')
from psis import psislw

# define normal densities
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*np.log(2*np.pi*v)
npdf = lambda x, m, v: np.exp(log_npdf(x, m, v))

# target density
p = lambda x: 0.8*npdf(x, -3, 6) + 0.2*npdf(x, 3, 0.5)
log_p = lambda x: np.log(p(x))

# the approximate distribution is univeriate normal
qmean = 2
qvars = [1, 2, 3, 5, 10]

# compute true posterior mean
m0 = quad(lambda x: x*p(x), -np.Inf, np.Inf)[0]

for qvar in qvars:

    # log ratio function
    log_r = lambda x: log_p(x) - log_npdf(x, qmean, qvar)

    # estimate khat
    N = 1000
    z = np.random.normal(qmean, np.sqrt(qvar), size=N)

    # compute importance sampling estimate
    mhat = np.mean(z*np.exp(log_r(z)))

    # estimate khat
    _, khat = psislw(log_r(z))

    # visualize
    xs = np.linspace(-10, 10, 1000)
    plt.figure(figsize=(10, 8))
    plt.plot(xs, np.exp(log_npdf(xs, qmean, qvar)), label='q')
    plt.plot(xs, p(xs), label='p')
    plt.axvline(m0, color='g', label='True posterior mean')
    plt.axvline(mhat, color='r', linestyle='--', label='Importance sampling estimate')
    plt.legend()
    plt.title('$\hat{k} = %3.2f$' % khat + f"$\sigma$ = {qvar}")

plt.show()