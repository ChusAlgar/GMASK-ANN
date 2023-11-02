# MASK - Multilevel Approximate Similarity Search with *k*-means

## Description

MASK is an algorithm to solve the approximate nearest neighbours (ANN) search problem with an unconventional application of the
*k*-means clustering algorithm.

The initial version of this algorithm is based on *k*-means and it only supports the Euclidean distance function. The project is 
currently under development to achieve several improvements:

- Generalizing MASK to accept arbitrary distance functions: Minkowski distances (e.g. Manhattan, Chebyshev), cosine similarity, Jaccard, etc.
- Accelerating code execution:
    - Exploring [Numba](https://numba.pydata.org/) to seep up calculations involving arrays and matrices.
    - Parallel/multithreading code execution.
    - Integrating CUDA-compatible code implementing clustering algorithms.
 
## References

- Ortega, F., Algar, M. J., de Diego, I. M., & Moguerza, J. M. (2023). Unconventional application of k-means for distributed approximate similarity search.
*Information Sciences, 619*, 208-234. [[HTML] sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0020025522013056).
- [Pre-print article (arXiv)](https://arxiv.org/abs/2208.02734).
