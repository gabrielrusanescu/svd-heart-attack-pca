# Cardiovascular Risk Analysis

This repository contains two implementations for analyzing cardiovascular risk datasets using Principal Component Analysis (PCA) and Singular Value Decomposition (SVD).

## Versions

### 1. Custom Implementation (Without Linear Algebra Libraries)
The file `proiect fara linalg.py` provides a custom implementation of SVD and PCA from scratch.
- **Key Techniques**: Implements Householder reflections for tridiagonalization and the QR algorithm to compute eigenvalues and eigenvectors without relying on high-level `numpy.linalg` routines.
- **Purpose**: Educational exploration of numerical linear algebra algorithms.

### 2. Standard Implementation (Using `np.linalg`)
The file `proiect cu linalg.py` provides an optimized implementation using standard NumPy linear algebra functions.
- **Key Techniques**: Uses `np.linalg.svd` for efficient matrix decomposition.
- **Purpose**: Provides a baseline for performance comparison and cleaner, more maintainable code for production-like environments.

## Comparison

| Feature | `proiect_fara_linalg.py` | `proiect_cu_linalg.py` |
| :--- | :--- | :--- |
| **SVD Method** | Custom QR Iteration | `np.linalg.svd` |
| **Performance** | Slower (Iterative) | Faster (Optimized BLAS/LAPACK) |
| **Complexity** | High (Implementation from scratch) | Low (Calls library functions) |

## Features

- **Data Preprocessing**: Handles missing data and performs standard scaling.
- **Dimensionality Reduction**: Ranks features and approximates matrices using Rank-k approximations.
- **Analysis**: Provides weights/importance of features contributing to cardiovascular risk.
- **Visualization**: Generates plots for explained variance and clusters patients by risk category.

## Documentation 

Documentation (`doc.pdf`) was made by [Edd12321](https://github.com/Edd12321)

## Usage

1. Place your dataset `cardiovascular_risk_dataset.csv` in the project directory.
2. Run either implementation:
   - For custom version: `python proiect fara linalg.py`
   - For standard version: `python proiect cu linalg.py`

## License

This project is intended for educational use.
