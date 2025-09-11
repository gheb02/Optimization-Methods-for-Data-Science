import numpy as np
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
from sklearn.model_selection import StratifiedKFold
from itertools import combinations  



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

    def fit_smo(self, X, y, tol=1e-5, max_passes=20, max_iter=1000, verbose=False):
        """
        SMO (MVP) training for binary SVM.
        """
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

        # Incremental error cache
        E = np.array([f(i) - y_enc[i] for i in range(n)])

        while passes < max_passes and iteration < max_iter:
            num_changed = 0

            for i in range(n):
                ei = E[i]
                ai = alphas[i]
                yi = y_enc[i]

                eps = 1e-8  # tolleranza numerica

                fi = f(i)

                violates = (
                    (ai < eps and yi * fi < 1 - tol) or
                    (eps < ai < self.C - eps and abs(yi * fi - 1) > tol) or
                    (ai > self.C - eps and yi * fi > 1 + tol)
                )

                if not violates:
                    continue

                # Most violating pair selection (heuristic)
                candidates = np.where((np.arange(n) != i) & (np.abs(E - ei) > 1e-5))[0]
                if len(candidates) == 0:
                    continue
                j = candidates[np.argmax(np.abs(E[candidates] - ei))]

                aj = alphas[j]
                yj = y_enc[j]
                ej = E[j]

                # Compute bounds L and H
                if yi != yj:
                    L = max(0, aj - ai)
                    H = min(self.C, self.C + aj - ai)
                else:
                    L = max(0, ai + aj - self.C)
                    H = min(self.C, ai + aj)
                if L == H:
                    continue

                # Correct eta
                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue

                old_aj = aj
                alphas[j] += yj * (ei - ej) / eta
                alphas[j] = np.clip(alphas[j], L, H)
                if abs(alphas[j] - old_aj) < 1e-12:
                    continue
                alphas[i] += yi * yj * (old_aj - alphas[j])

                # Update bias
                b1 = b - ei - yi * (alphas[i] - ai) * K[i, i] - yj * (alphas[j] - old_aj) * K[i, j]
                b2 = b - ej - yi * (alphas[i] - ai) * K[i, j] - yj * (alphas[j] - old_aj) * K[j, j]
                if 0 < alphas[i] < self.C:
                    b = b1
                elif 0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                # Incremental error update
                E[i] = f(i) - yi
                E[j] = f(j) - yj

                num_changed += 1

            if verbose:
                print(f"Iteration {iteration+1}, num_changed = {num_changed}, passes = {passes}")

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

            iteration += 1

        # Extract support vectors
        sv_mask = alphas > 1e-8
        self.alpha_sv = alphas[sv_mask]
        self.X_sv = Xs[sv_mask]
        self.y_sv = y_enc[sv_mask]
        # Calcolo più preciso del bias usando solo i support vector liberi
        free_sv_mask = (alphas > tol) & (alphas < self.C - tol)
        free_indices = np.where(free_sv_mask)[0]

        if len(free_indices) > 0:
            b_vals = []
            for i in free_indices:
                f_i = np.sum(alphas * y_enc * K[:, i])
                b_vals.append(y_enc[i] - f_i)
            self.b = np.mean(b_vals)
        else:
            # Fallback: usa il valore di b dell’ultima iterazione
            self.b = b
        
        obj_val = np.sum(alphas) - 0.5 * np.sum(
        np.outer(alphas * y_enc, alphas * y_enc) * K
        )

        self.alphas = alphas.copy()

        if verbose:
            print(f"SMO finished in {iteration} iterations. Support vectors: {len(self.alpha_sv)}")
            print(f"Final value of the dual SVM objective: {obj_val:.6f}")

        return self
    
    def optimality_gap(self, X, y):
        """
        Compute final difference m(lambda) - M(lambda).
        """
        # ensure labels are mapped
        y_enc = self.encode_labels(y)

        # decision values for training set
        scores = self.decision_function(X)

        # gradient of dual: g_i = 1 - y_i f(x_i)
        g = 1 - y_enc * scores

        # apply projected gradient rules
        proj_grad = np.zeros_like(g)
        for i in range(len(g)):
            if 0 < self.alphas[i] < self.C:
                proj_grad[i] = g[i]
            elif self.alphas[i] <= self.tol:
                proj_grad[i] = min(0, g[i])
            elif self.alphas[i] >= self.C - self.tol:
                proj_grad[i] = max(0, g[i])

        # m = max over allowed gradients, M = min over allowed gradients
        m_val = np.max(proj_grad)
        M_val = np.min(proj_grad)

        return m_val - M_val


class MultiSVM(SVM):
    def __init__(self, method="OvR", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.models = {}  # dict per OvR, sarà sovrascritto se OvO
        self.pair_classes = []  # usata solo in OvO
        self.classes = None
        self.binary_model = None

    def fit(self, X, y, verbose=False):
        classes = np.unique(y)
        if self.method not in ["OvO", "OvR"]:
            raise ValueError('Please select "OvO" or "OvR"')

        if len(classes) < 2:
            raise ValueError("Please select 2 or more classes")

        self.classes = np.array(classes)

        if len(classes) == 2:
            self.binary_model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
            self.binary_model.fit(X, y, verbose=verbose)
        elif len(classes) >= 3 and self.method == "OvR":
            self.models = {}  # dict
            for cls in classes:
                y_bin = np.where(y == cls, 1, -1)
                model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
                model.fit(X, y_bin, verbose=verbose)
                self.models[cls] = model
        elif len(classes) >= 3 and self.method == "OvO":
            self.models = []  # lista per modelli
            self.pair_classes = []
            for cls1, cls2 in combinations(self.classes, 2):
                idx = np.where((y == cls1) | (y == cls2))[0]
                X_pair = X[idx]
                y_pair = y[idx]
                y_bin = np.where(y_pair == cls1, 1, -1)
                model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
                model.fit(X_pair, y_bin, verbose=verbose)
                self.models.append(model)
                self.pair_classes.append((cls1, cls2))
        return self

    def fit_smo(self, X, y, max_iter=10, verbose=False):
        classes = np.unique(y)
        if self.method not in ["OvO", "OvR"]:
            raise ValueError('Please select "OvO" or "OvR"')

        if len(classes) < 2:
            raise ValueError("Please select 2 or more classes")

        self.classes = np.array(classes)

        if len(classes) == 2:
            self.binary_model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
            self.binary_model.fit_smo(X, y, max_iter=max_iter, verbose=verbose)
        elif len(classes) >= 3 and self.method == "OvR":
            self.models = {}
            for cls in classes:
                y_bin = np.where(y == cls, 1, -1)
                model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
                model.fit_smo(X, y_bin, max_iter=max_iter, verbose=verbose)
                self.models[cls] = model
        elif len(classes) >= 3 and self.method == "OvO":
            self.models = []
            self.pair_classes = []
            for cls1, cls2 in combinations(self.classes, 2):
                idx = np.where((y == cls1) | (y == cls2))[0]
                X_pair = X[idx]
                y_pair = y[idx]
                y_bin = np.where(y_pair == cls1, 1, -1)
                model = SVM(C=self.C, kernel=self.kernel_type, gamma=self.gamma, p=self.p, tol=self.tol)
                model.fit_smo(X_pair, y_bin, max_iter=max_iter, verbose=verbose)
                self.models.append(model)
                self.pair_classes.append((cls1, cls2))
        return self

    def predict(self, X):
        if len(self.classes) == 2:
            return self.binary_model.predict(X)
        elif len(self.classes) >= 3 and self.method == "OvR":
            scores = np.column_stack([self.models[cls].decision_function(X) for cls in self.classes])
            idx = np.argmax(scores, axis=1)
            return self.classes[idx]
        elif len(self.classes) >= 3 and self.method == "OvO":
            votes = np.zeros((X.shape[0], len(self.classes)), dtype=int)
            for model, (cls1, cls2) in zip(self.models, self.pair_classes):
                pred = model.predict(X)
                for i, p in enumerate(pred):
                    if p == 1:
                        votes[i, np.where(self.classes == cls1)[0][0]] += 1
                    else:
                        votes[i, np.where(self.classes == cls2)[0][0]] += 1
            return self.classes[np.argmax(votes, axis=1)]




def grid_search_cv(X, y, C_grid, gamma_grid=None, degree_grid=None,
                   k=5, random_state=0, smo=False, max_iter=10):
    """
    Grid search for Gaussian and Polynomial SVM with k-fold CV.
    Returns: best_params, best_score, all_results (list of dicts)
    """


    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params = {}
    all_results = []


    if not smo:
        for C in C_grid:
            # --- Gaussian kernel ---
            if gamma_grid:
                for gamma in gamma_grid:
                    scores = []
                    for train_idx, val_idx in skf.split(X, y):
                        model = SVM(C=C, kernel='Gaussian', gamma=gamma)
                        model.fit(X[train_idx], y[train_idx])
                        scores.append(model.score(X[val_idx], y[val_idx]))
                    mean_score = np.mean(scores)
                    all_results.append({
                        'kernel': 'Gaussian', 'C': C, 'gamma': gamma,
                        'mean_score': mean_score
                    })
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'gamma': gamma, 'kernel': 'Gaussian'}

            # --- Polynomial kernel ---
            if degree_grid:
                for degree in degree_grid:
                    scores = []
                    for train_idx, val_idx in skf.split(X, y):
                        model = SVM(C=C, kernel='Polynomial', p=degree)
                        model.fit(X[train_idx], y[train_idx])
                        scores.append(model.score(X[val_idx], y[val_idx]))
                    mean_score = np.mean(scores)
                    all_results.append({
                        'kernel': 'Polynomial', 'C': C, 'degree': degree,
                        'mean_score': mean_score
                    })
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'degree': degree, 'kernel': 'Polynomial'}
    else:
        for C in C_grid:
            # --- Gaussian kernel ---
            if gamma_grid:
                for gamma in gamma_grid:
                    scores = []
                    for train_idx, val_idx in skf.split(X, y):
                        model = SVM(C=C, kernel='Gaussian', gamma=gamma)
                        model.fit_smo(X[train_idx], y[train_idx], max_iter=max_iter)
                        scores.append(model.score(X[val_idx], y[val_idx]))
                    mean_score = np.mean(scores)
                    all_results.append({
                        'kernel': 'Gaussian', 'C': C, 'gamma': gamma,
                        'mean_score': mean_score})
                    print(f"Trying C={C}, gamma={gamma} -> mean score={mean_score:.4f}")
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'gamma': gamma, 'kernel': 'Gaussian'}

            # --- Polynomial kernel ---
            if degree_grid:
                for degree in degree_grid:
                    scores = []
                    for train_idx, val_idx in skf.split(X, y):
                        model = SVM(C=C, kernel='Polynomial', p=degree)
                        model.fit_smo(X[train_idx], y[train_idx], max_iter=max_iter)
                        scores.append(model.score(X[val_idx], y[val_idx]))
                    mean_score = np.mean(scores)
                    all_results.append({
                        'kernel': 'Polynomial', 'C': C, 'degree': degree,
                        'mean_score': mean_score})
                    print(f"Trying C={C}, degree={degree} -> mean score={mean_score:.4f}")

                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'degree': degree, 'kernel': 'Polynomial'}

    print(best_params, best_score)
    return best_params, best_score, all_results
