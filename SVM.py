import numpy as np
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers


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
    
    
    def fit(self, X, y, verbose=False):
        """
        X: (n_samples, n_features) numpy array
        y: array-like labels (two classes)
        """

        if verbose:
            solvers.options['show_progress'] = True
        else:
            solvers.options['show_progress'] = False



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
            if len(b_vals) == 0:
                self.b = 0.0
            else:
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

    def fit_smo(self, X, y, tol=1e-3, max_passes=10, max_iter=1000, verbose=False):
        
        X = np.asarray(X, dtype=float)
        y_in = np.asarray(y).ravel()
        y_enc = self.encode_labels(y_in)

        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)

        n = Xs.shape[0]
        C = float(self.C)
        K = self.kernel(Xs)

        alphas = np.zeros(n, dtype=float)
        b = 0.0

        def f(i):
            return np.sum(alphas * y_enc * K[:, i]) + b

        # initial E
        E = np.array([f(i) - y_enc[i] for i in range(n)], dtype=float)

        passes = 0
        it = 0

        if verbose:
            print("Starting SMO (MVP) training...")

        while passes < max_passes and it < max_iter:
            num_changed = 0
            it += 1

            for i in range(n):
                Ai = alphas[i]
                yi = y_enc[i]
                Fi = f(i)
                Ei = Fi - yi

                # KKT check
                violates = False
                if (Ai < C - tol and yi * Fi < 1 - tol) or (Ai > tol and yi * Fi > 1 + tol):
                    violates = True

                if not violates:
                    continue

                # MVP: pick j that maximizes |Ei - Ej|
                abs_diff = np.abs(E - Ei)
                abs_diff[i] = -1.0  # exclude i
                j = int(np.argmax(abs_diff))
                if abs_diff[j] < 1e-12:
                    # fallback random j
                    cand = list(range(n))
                    cand.remove(i)
                    j = np.random.choice(cand)

                Aj_old = alphas[j]
                yj = y_enc[j]
                Ej = E[j]

                # compute L and H
                if yi != yj:
                    L = max(0.0, Aj_old - Ai)
                    H = min(C, C + Aj_old - Ai)
                else:
                    L = max(0.0, Ai + Aj_old - C)
                    H = min(C, Ai + Aj_old)

                if L == H:
                    continue

                Kii = K[i, i]
                Kjj = K[j, j]
                Kij = K[i, j]

                eta = Kii + Kjj - 2.0 * Kij
                if eta <= 1e-12:
                    continue

                # store previous values for incremental updates
                prev_b = b
                prev_Ai = Ai
                prev_Aj = Aj_old

                # analytic update
                new_Aj = Aj_old + yj * (Ei - Ej) / eta
                new_Aj = np.clip(new_Aj, L, H)
                if abs(new_Aj - Aj_old) < 1e-8:
                    continue

                new_Ai = Ai + yi * yj * (Aj_old - new_Aj)

                alphas[i] = new_Ai
                alphas[j] = new_Aj

                # update b
                b1 = prev_b - Ei - yi * (alphas[i] - prev_Ai) * Kii - yj * (alphas[j] - prev_Aj) * Kij
                b2 = prev_b - Ej - yi * (alphas[i] - prev_Ai) * Kij - yj * (alphas[j] - prev_Aj) * Kjj

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                # incremental update of E for all k:
                # E_k := E_k + (alphas[i] - prev_Ai) * yi * K[k,i] + (alphas[j] - prev_Aj) * yj * K[k,j] + (b - prev_b)
                delta_ai = alphas[i] - prev_Ai
                delta_aj = alphas[j] - prev_Aj
                delta_b = b - prev_b

                if abs(delta_ai) > 0 or abs(delta_aj) > 0 or abs(delta_b) > 0:
                    E += delta_ai * yi * K[:, i] + delta_aj * yj * K[:, j] + delta_b

                num_changed += 1

            if verbose:
                print(f"Iteration {it}: num_changed = {num_changed}")

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # finalize model storage
        self.alphas = alphas.copy()
        sv_mask = alphas > 1e-6
        self.support_ = np.where(sv_mask)[0]
        self.alpha_sv = alphas[sv_mask]
        self.X_sv = Xs[sv_mask]
        self.y_sv = y_enc[sv_mask]
        self.b = b

        if verbose:
            print(f"SMO finished in {it} iterations. Support vectors: {len(self.alpha_sv)}")

        return self