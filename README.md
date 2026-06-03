# Cardiovascular Risk Analysis

This repository contains two implementations for analyzing cardiovascular risk datasets using Principal Component Analysis (PCA) and Singular Value Decomposition (SVD).
The dataset used is available at [https://www.kaggle.com/datasets/vishardmehta/heart-risk-progression-dataset](https://www.kaggle.com/datasets/vishardmehta/heart-risk-progression-dataset)

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

## Advantages

- **Dimensionality Reduction**: Medical datasets often contain dozens of variables (e.g., age, blood pressure, cholesterol, weight, family history). The application effectively compresses all this information into just 2-3 "Principal Components," making the dataset much easier to manage.

- **Identification of Dominant Risk Factors**: Through the function that displays weights (afisponderi), the application shows exactly which clinical variables have the strongest impact or dominate the formation of cardiovascular risk. There is no need for guesswork.

- **Noise Filtering**: By eliminating components with very small singular values (the last ones in the table), the application discards measurement errors and statistical "noise," retaining only the strong patterns within the data.

## Disadvantages

- **Loss of direct interpretability (the biggest disadvantage)**: A doctor perfectly understands what "Blood Pressure = 140" means. However, Principal Component 1 is a mathematical equation (e.g., 0.5 x Blood Pressure + 0.3 x Age + 0.1 x Cholesterol). There is no clinical unit of measurement for "Component 1," which makes the result harder to explain to a patient.

- **Sensitivity to anomalies (outliers)**: PCA is based on variance (squaring distances). If there are patients in the dataset with incorrectly entered data or extremely high/abnormal values, they will "strongly pull" the axes toward them, distorting the entire model.

- **Maximum variance does not necessarily mean good prediction**: PCA looks for the directions in which the data spreads the most. However, it is possible that the maximum spread is driven by age, while the actual risk of heart attack is defined by a more subtle variable (with lower variance) that PCA discards into the lower components.

## Documentation 

Documentation (`doc.pdf`) was made by [Edd12321](https://github.com/Edd12321)

## Usage

1. Place your dataset `cardiovascular_risk_dataset.csv` in the project directory.
2. Run either implementation:
   - For custom version: `python proiect fara linalg.py`
   - For standard version: `python proiect cu linalg.py`
