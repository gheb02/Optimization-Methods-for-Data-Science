# Neural Network & SVM from Scratch

This repository implements **Machine Learning models from scratch** (using only NumPy and basic Python).  

It covers both **regression** with **Neural Networks** and **classification** with **Support Vector Machines (SVM)**, including binary and multiclass strategies.

All models were trained and evaluated on features extracted by a Convolutional Neural Network (CNN) from a subset of the **[UTKFace dataset](https://susanqq.github.io/UTKFace/)**.  

The focus is on implementing and understanding the optimization and learning of **SVMs** and **Neural Networks** from scratch, without relying on high-level ML libraries.

---

## Implemented Featutres

- ### Neural Network for Regression
    - Fully connected feedforward network  
    - Backpropagation  
    - Optimizers: **Adam**, **Batch Gradient Descent**, **Mini-Batch GD**, **Stochastic GD**  
    - Configurable hidden layers, activation functions, and initialization strategies  
    - Evaluation metrics: **MSE**, **MAE**, **MAPE**  
    - **k-fold cross-validation** for hyperparameter tuning  

- ### Support Vector Machine (SVM)
    - Dual optimization with **CVXOPT**  
    - Sequential Minimal Optimization (**SMO**) solver  
    - Gaussian (RBF) and Polynomial kernels  
    - **k-fold cross-validation** for hyperparameter tuning  
    - Binary classification support  

- ### Multiclass SVM
    - **One-vs-One (OvO)** and **One-vs-All (OvR)** strategies  
    - Extends binary SVM to handle **3+ classes**  

---

## Repository Structure

- `NN_Functions.py` – Neural Network implementation (architecture + training)  
- `Neural Network Application.ipynb` – Notebook with hyperparameter tuning and regression applications
- `SVM.py` – Binary SVM implementation (CVXOPT & SMO) + Multiclass wrapper (OvO & OvR)  
- `SVM Application.ipynb` – Notebook with hyperparameter tuning and classification applications

---


## Results

A summary of the models' performance is provided below. For a complete
breakdown of the training process, hyperparameter tuning, and cross-validation results, please see the full reports.

* **Neural Network for Regression:**
    * The model was tuned using k-fold cross-validation, achieving a **test loss** of 45.53, a **test MAE** of 7.21 and a **test MAPE** of 23.46%.
    * [Neural Network Results Report](<Reports/Neural Network Final Report.pdf>)

* **Support Vector Machine (SVM):**
    * The final SVM achieved a **test accuracy** of 93% for binary classification.
    * Multiclass strategies (OvO/OvR) performance details are available in the
        full report.
    * [SVM Application Report](<Reports/SVM Final Report.pdf>)

---