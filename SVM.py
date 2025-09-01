import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, C=1.0, kernel='Gaussian', gamma=1, p=3,
                 tol=1e-6):
        # hyperparams
        self.C = float(C)
        self.kernel_type = kernel 
        self.gamma = gamma    
        self.p = p
        self.tol = tol

        # learned attributes
        self.alphas = None        # all alpha (length n)
        self.support_ = None      # indices of support vectors
        self.X_sv = None
        self.y_sv = None
        self.alpha_sv = None      # alphas corresponding to sv
        self.b = 0.0

        # preprocessing
        self.scaler = None
        # mapping labels -> {-1, +1}
        self.label_map = None
        self.inv_label_map = None



    def kernel(self, X1, X2=None):
            
            if X2 is None:
                X2 = X1

            if self.kernel_type == "Gaussian":
                # Compute squared norms
                X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
                X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
                # Pairwise squared distance
                dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
                # Gaussian kernel
                K = np.exp(-self.gamma * dist_sq)
                return K

            elif self.kernel_type == "Polynomial":
                K = (X1 @ X2.T + 1)**self.p
                return K
            

    def encode_labels(self, y):
        uniq = np.unique(y)
        if uniq.shape[0] != 2:
            raise ValueError("This implementation supports only binary classification.")
        # map unique[0] -> -1, unique[1] -> +1
        self.label_map = {uniq[0]: -1, uniq[1]: 1}
        self.inv_label_map = {-1: uniq[0], 1: uniq[1]}
        y_mapped = np.array([self.label_map[yy] for yy in y], dtype=float)
        return y_mapped
    
    
    def fit(self, X, y):
        """
        X: (n_samples, n_features) numpy array
        y: array-like labels (two classes)
        """
        X = np.asarray(X, dtype=float)
        y_in = np.asarray(y).ravel()

        # encode labels to -1 / +1
        y = self.encode_labels(y_in)

        # scaling
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        n_samples = Xs.shape[0]

        # compute full kernel matrix
        K = self.kernel(Xs)    # (n_samples, n_samples)

        # Build Q = (y y^T) * K
        Y = y.reshape(-1, 1)
        Q = (Y @ Y.T) * K  # elementwise multiply
        # numerical stabilizer: symmetrize & add tiny jitter
        # Q = 0.5 * (Q + Q.T) + 1e-12 * np.eye(n_samples)

        P = matrix(Q)                         # cvxopt matrix
        q = matrix(-np.ones((n_samples, 1)))  # minimize 1/2 a^T P a + q^T a -> q = -1

        # G and h for 0 <= alpha <= C
        G_std = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h_std = np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C))
        G = matrix(G_std)
        h = matrix(h_std)

        # equality A y = 0  (sum alpha_i y_i = 0)
        A = matrix(y.reshape(1, -1))
        b = matrix(0.0)

        sol = solvers.qp(P, q, G, h, A, b)

        # extract alphas (as numpy array)
        alphas_all = np.array(sol['x']).reshape(-1)

        # keep full alphas array for diagnostics
        self.alphas = alphas_all.copy()

        # support vectors: alpha > tol
        sv_mask = alphas_all > self.tol
        sv_indices = np.where(sv_mask)[0]
        self.support_ = sv_indices
        self.X_sv = Xs[sv_mask]
        self.y_sv = y[sv_mask]
        self.alpha_sv = alphas_all[sv_mask]

        # compute bias b: average over alphas with 0 < alpha < C (free SV)
        free_sv_mask = (alphas_all > self.tol) & (alphas_all < self.C - self.tol)
        free_indices = np.where(free_sv_mask)[0]

        if free_indices.size > 0:
            b_vals = []
            for i in free_indices:
                # f(x_i) = sum_j alpha_j y_j K(x_j, x_i)
                f_i = np.sum(alphas_all * y * K[:, i])
                b_vals.append(y[i] - f_i)
            self.b = np.mean(b_vals)
        else:
            # fallback: average over all support vectors
            b_vals = []
            for idx in sv_indices:
                f_i = np.sum(alphas_all * y * K[:, idx])
                b_vals.append(y[idx] - f_i)
            self.b = np.mean(b_vals)

        # keep training-data-size alphas if needed (self.alphas already)
        return self
    

    def decision_function(self, X):
        """Return raw decision scores (float), using only SV."""
        X = np.asarray(X, dtype=float)
        Xs = self.scaler.transform(X)
        K = self.kernel(Xs, self.X_sv)  # shape (m, n_sv)
        # weights = alpha_sv * y_sv (elementwise)
        weights = self.alpha_sv * self.y_sv
        scores = K @ weights + self.b    # shape (m,)
        return np.asarray(scores).reshape(-1)

    def predict(self, X):
        scores = self.decision_function(X)
        preds = np.where(scores >= 0, 1, -1)
        # map back to original labels
        return np.array([self.inv_label_map[int(p)] for p in preds])
    
    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    

    def get_support(self):
        return self.support_
    
    def fit_smo(self, X, y, tol=1e-3, max_passes=10, max_iter=10):
        X = np.asarray(X, dtype=float)
        y_in = np.asarray(y).ravel()
        y_enc = self.encode_labels(y_in)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n = y_enc.shape[0]
        alphas = np.zeros(n)
        b = 0.0
        passes = 0
        iteration = 0
        K = self.kernel(Xs)

        def f(i):
            return np.sum(alphas * y_enc * K[:, i]) + b

        # Inizializza E in modo incrementale
        E = np.array([f(i) - y_enc[i] for i in range(n)])

        while passes < max_passes and iteration < max_iter:
            print(f"\n--- Iteration {iteration + 1}, consecutive passes without updates: {passes} ---")
            num_changed = 0
            tested_pairs = np.zeros((n, n), dtype=bool)  # Uso di una matrice booleana per il controllo

            for i in range(n):
                ei = E[i]
                ai = alphas[i]
                yi = y_enc[i]

                # Controllo KKT
                kkt_violation = ((ai < self.C and yi * ei < -tol) or (ai > 0 and yi * ei > tol))
                if not kkt_violation:
                    continue

                # Seleziona il secondo esempio in modo più efficiente
                candidates = np.where(np.abs(E - ei) > 1e-3)[0]  # Seleziona solo quelli con errore significativo
                candidates = candidates[candidates != i]  # Escludi il punto corrente

                if len(candidates) == 0:
                    continue

                j = np.argmax(np.abs(E[candidates] - ei))  # Seleziona il punto con l'errore maggiore

                # Se la coppia è già stata testata, salta
                if tested_pairs[i, j]:
                    continue
                tested_pairs[i, j] = True
                tested_pairs[j, i] = True  # Per evitare di testare di nuovo la stessa coppia (simmetria)

                aj = alphas[j]
                yj = y_enc[j]
                ej = E[j]

                # Calcola i limiti L e H
                if yi != yj:
                    L = max(0, aj - ai)
                    H = min(self.C, self.C + aj - ai)
                else:
                    L = max(0, ai + aj - self.C)
                    H = min(self.C, ai + aj)

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                old_aj = aj
                alphas[j] -= yj * (ei - ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)

                if abs(alphas[j] - old_aj) < 1e-5:
                    continue

                alphas[i] += yi * yj * (old_aj - alphas[j])

                # Aggiorna il bias b
                b1 = b - ei - yi * (alphas[i] - ai) * K[i, i] - yj * (alphas[j] - old_aj) * K[i, j]
                b2 = b - ej - yi * (alphas[i] - ai) * K[i, j] - yj * (alphas[j] - old_aj) * K[j, j]

                if 0 < alphas[i] < self.C:
                    b = b1
                elif 0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                # Aggiorna l'errore
                E = np.array([f(k) - y_enc[k] for k in range(n)])
                num_changed += 1

            print(f"Number of alphas updated in this iteration: {num_changed}")

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

            iteration += 1

        # Salva il modello finale
        sv_mask = alphas > 1e-6
        self.alpha_sv = alphas[sv_mask]
        self.b = b
        self.X_sv = Xs[sv_mask]
        self.y_sv = y_enc[sv_mask]

        print(f"\nTraining completed in {iteration} iterations. Total support vectors: {len(self.alpha_sv)}")
        return self