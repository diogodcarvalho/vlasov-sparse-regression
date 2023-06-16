# vlasov-sparse-regression

This repository contains the project developed for the Statistical Learning 2022/2023 Course (IST) on the topic:

_["Sparse regression for the recovery of nonlinear dynamics: SINDy, thoughts on correlated variables and optimization algorithms"](SL2023_Project.ipynb)_

The Jupyter notebook provides an overview of common sparse regression methods and uses as a test case the discovery of the Vlasov equation from Particle-In-Cell (PIC) plasma physics simulations. Additionally, it illustrates some of the issues associated with performing sparse regression on highly correlated data, and the influence of using backward _versus_ forward best subset selection optimization algorithms for this task.

The work is heavily inspired by E. P. Alves and F. Fiúza (2022) _["Data-driven discovery of reduced plasma physics models from fully kinetic simulations"](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.033192)_ paper and the corresponding example [code](https://github.com/epalves/data-driven-plasma/).

