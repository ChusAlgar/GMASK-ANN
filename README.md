# GMASK - Generalised Multilevel Approximate Similarity Search with *k*-medoids

## Description

GMASK is a generalised algorithm to solve the approximate nearest neighbours (ANN) search problem for distributed data that accepts any arbitrary distance function by employing data partitioning algorithms that induce Voronoi regions in a dataset and yield a representative element, such as *k*-medoids.

The project is currently under development to achieve several improvements:

- Integrating distance functions beyond Minkowski, such as pseudo-metrics (Minkowski distances with *p* < 1) or non-metric dissimilarities.
- Accelerating code execution:
    - Exploring [Numba](https://numba.pydata.org/) to seep up calculations involving arrays and matrices.
    - Parallel/multithreading code execution.
    - Integrating CUDA-compatible code implementing clustering algorithms.
 
## References

- Ortega, F., Algar, M. J., de Diego, I. M., & Moguerza, J. M. (2023). Unconventional application of k-means for distributed approximate similarity search.
*Information Sciences, 619*, 208-234. [[HTML] sciencedirect.com](https://www.sciencedirect.com/science/article/pii/S0020025522013056).
