# GMM-VAE
This project implements a variant of Variational AutoEncoder (VAE) with Gaussian Mixture as a prior distribution.
The goal is providing a tool for unsupervised clustering through deep generative models.
Contrary to classical VAE models, this model assume that that the observed data (inputs) is generated from a mixture of K normal distributions, K being number of clusters data may lay on.

For example this kind of models can serve for:
- automatic labeling of unlabeled data, 
- "plausible" data reconstruction for unbalanced dataset,
- anomaly detection 
- etc.

The implementation of this generative model is based on the paper: "Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"

By : Nat Dilokthanakul and Pedro A. M. Mediano and Marta Garnelo and Matthew C. H. Lee and Hugh Salimbeni and Kai Arulkumaran and Murray Shanahan.

(https://arxiv.org/abs/1611.02648)




