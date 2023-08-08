# ltcurveMCMC

This repository experiments with running MCMC chains for fitting light curves.\
The primary focus would be to use the technique presented in [arXiv:2210.17433](https://arxiv.org/abs/2210.17433) which involves fitting lightcurves as a linear combination of 3 templates/basis vectors generated using PCA.

Installation:
```
git clone https://github.com/b-biswas/ltcurveMCMC
cd ltcurveMCMC
pip install -e .
```

Use: \
Refer to [this notebook](https://github.com/b-biswas/ltcurveMCMC/blob/main/notebooks/ZTF_mcmc_pcs.ipynb)