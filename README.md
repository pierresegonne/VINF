![ex_space_warp](https://raw.githubusercontent.com/pierresegonne/VINF/master/space_warp/Figure_2%20copy.png)

# Variational Inference using Normalizing Flows (VINF)

This repository provides a hands-on `tensorflow` implementation of Normalizing Flows as presented in the [paper](https://arxiv.org/pdf/1505.05770.pdf)
introducing the concept (D. Rezende & S. Mohamed). This code was developed as part of a Special Course at DTU (Denmarks Tekniske Universitet), supervised
by Michael Riis Andersen. The final report of the course, that details all experiments run with this repository can directly be accessed at [https://pierresegonne.github.io/VINF/](https://pierresegonne.github.io/VINF/)

## Implementation
This repository provides an implementation of 
- ADVI (Automatic Differential Variational Inference, with Diagonal Gaussian, baseline) 
- Planar Flow
- Radial Flow

## Demonstrative distributions 

True posterior

![true_energies](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_densities.png)

Samples generated from the trained variational approximation

![energy_1](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_1_pf_hexbin.png)
![energy_2](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_2_pf_hexbin.png)
![energy_3](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_3_pf_hexbin.png)
![energy_4](https://raw.githubusercontent.com/pierresegonne/VINF/master/assets/energy_4_pf_hexbin.png)


## TODO
- [ ] Run additional experiments on radial flows
- [ ] Add requirements.txt
- [ ] Improve models with the use of bijectors. See [this thread](https://stackoverflow.com/questions/61717694/embed-trainable-bijector-into-keras-model/62284510#62284510) for a starting point
- [ ] Include new flow models.
  - [ ] Glow [paper](https://papers.nips.cc/paper/8224-glow-generative-flow-with-invertible-1x1-convolutions.pdf)
  - [ ] MAF [paper](https://papers.nips.cc/paper/6581-improved-variational-inference-with-inverse-autoregressive-flow.pdf)
  - [ ] IAF [paper](https://papers.nips.cc/paper/6828-masked-autoregressive-flow-for-density-estimation.pdf)
