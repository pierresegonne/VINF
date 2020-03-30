from tensorflow import math

AVAILABLE_DISTRIBUTIONS = ['', 'banana', 'circle']

def pdf_2D(z, density_name=''):
    assert density_name in AVAILABLE_DISTRIBUTIONS, "Incorrect density name."
    if density_name == '':
        return 1
    elif density_name == 'banana':
        raise Exception('Not implemented yet.')
    elif density_name == 'circle':
        z1, z2 = z[:, 0], z[:, 1]
        norm = (z1**2 + z2**2)**0.5
        exp1 = math.exp(-0.2 * ((z1 - 2) / 0.8) ** 2)
        exp2 = math.exp(-0.2 * ((z1 + 2) / 0.8) ** 2)
        u = 0.5 * ((norm - 4) / 0.4) ** 2 - math.log(exp1 + exp2)
        return math.exp(-u)
