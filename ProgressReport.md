Zexin:

Chart about the performance of the current version of recurrent autoencoder (RAE) has been included. Parameters are: time series length is 60, input data dimension is (1, 60, 2). (bivariate time series)
Comparing the performance of RAE, we can see that with enough internal units, RAE can compress data well enough so that MSE goes to zero eventually.

Checked some times series GAN models:
1. The NIPS paper, which is not financially oriented: https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html
2. A recent financially oriented time series GAN paper: https://ui.adsabs.harvard.edu/abs/2019PhyA..52721261T/abstract
  I made a workable implementation of it, following https://github.com/stakahashy/fingan (particularly messy implementation)
3. QuantGAN: https://arxiv.org/pdf/1907.06673v2.pdf

To do:
1. Write recurrent autoencoder into python script.
2. Implementation of HPCA: https://gmarti.gitlab.io/qfin/2020/07/05/hierarchical-pca-avellaneda-paper.html
