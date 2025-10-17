# BGP Methods (2025)

This repository contains code used to generate the figures in [Quasinormal modes from numerical relativity with Bayesian inference (Dyer & Moore, submitted)](https://arxiv.org/pdf/2510.11783).

<div align="center">
  
![Amplitude spiral gif](outputs/amplitude_spiral.gif)

*Figure 1: A `BGP' fit of a (2,2,n) overtone model to the (2,2) spherical mode. The figure shows the posterior distribution for the first overtone ($n=1$) animated across 
a range of start times. The line traced out in the top right figure shows the mean value of the amplitude tracing out a spiral in the complex plane.*

</div>

# Requirements & usage 

To use this code you will need to install [bgp_qnm_fits](https://github.com/Richardvnd/bgp_qnm_fits). 

This code uses NR waveforms from the Spectral Einstein Code. CCE waveforms can be obtained from the [SXS Gravitational Waveform Database](https://data.black-holes.org/waveforms/extcce_catalog.html). All simulations were transformed into the superrest frame using the [`scri`](https://pypi.org/project/scri/) package.