import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import halfcauchy

"""
Example of transformation of constrained variables. From ADVI Section 2.3
"""

theta = np.linspace(0+1e-2,10,1000)
zeta = np.linspace(-5,5,1000)

pdf_original = halfcauchy(loc=0, scale=5).pdf(theta)
pdf_transformed = halfcauchy(loc=0, scale=5).pdf(np.exp(zeta)) * np.exp(zeta)

plt.plot(theta, pdf_original, label='p(tau)')
plt.plot(zeta, pdf_transformed, label='p(log(tau))')
plt.legend()
plt.show()