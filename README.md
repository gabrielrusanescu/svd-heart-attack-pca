# Cardiovascular Risk Analysis Without Linear Algebra Libraries

This project provides a custom implementation of Singular Value Decomposition (SVD) and Principal Component Analysis (PCA) to analyze cardiovascular risk datasets. The implementation is designed to perform these operations from scratch, avoiding reliance on high-level linear algebra routines like `numpy.linalg.svd`.

## Features

- **Custom Matrix Decomposition:** Implements Householder reflections to tridiagonalize matrices, followed by the QR algorithm to compute eigenvalues and eigenvectors from scratch.
- **SVD Implementation:** Computes the Singular Value Decomposition by leveraging the QR iteration method.
- **PCA Analysis:** Performs dimensionality reduction and identifies key features contributing to cardiovascular risk.
- **Data Preprocessing:** Handles missing data by filling with column means and normalizes features using standard scaling.
- **Visualization:** Includes modules to plot the explained variance ratio and visualize patient clusters based on principal components.

## How It Works

1. **Tridiagonalization:** Uses Householder transformations to convert the covariance matrix into a tridiagonal form.
2. **QR Iteration:** Applies the QR algorithm to iteratively approximate the eigenvalues and eigenvectors of the tridiagonal matrix.
3. **Dimensionality Reduction:** Uses the computed singular values to rank features and perform PCA.
4. **Analysis:** Generates reports on the weight of different health indicators (e.g., smoking status, family history) on the calculated risk categories.

## Prerequisites

To run this script, ensure you have the following installed:

- Python 3.x
- NumPy
- Pandas
- Matplotlib

## Usage

1. Place your dataset file named `cardiovascular_risk_dataset.csv` in the same directory as the script.
2. Run the script:
   `python proiect_fara_linalg.py`

The script will output:
- A table showing the approximation error for different ranks (k).
- An analysis of the most influential features (weights) for the top components.
- Execution time performance metrics.
- Visualizations for singular values and a 2D plot of patient risk categories.

## License

This project is provided for educational purposes in numerical linear algebra.
