import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(42)

def f_runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def make_design(x, degree):
    x = np.asarray(x)
    X = np.ones((x.size, degree + 1))
    for j in range(1, degree + 1):
        X[:, j] = x**j
    return X

def train_test_split(x, y, test_fraction=0.2, seed=0):
    rng_local = np.random.default_rng(seed)
    n = len(x)
    idx = rng_local.permutation(n)
    m = int(n * test_fraction)
    te = idx[:m]
    tr = idx[m:]
    return x[tr], y[tr], x[te], y[te]

def scale_X_pair(Xtr, Xte):
    Xtr_s = Xtr.copy()
    Xte_s = Xte.copy()
    for j in range(1, Xtr.shape[1]):
        mu = Xtr[:, j].mean()
        sd = Xtr[:, j].std()
        if sd == 0:
            sd = 1.0
        Xtr_s[:, j] = (Xtr[:, j] - mu) / sd
        Xte_s[:, j] = (Xte[:, j] - mu) / sd
    return Xtr_s, Xte_s

def ols_fit(X, y):
    return np.linalg.pinv(X) @ y

def ridge_fit(X, y, lam):
    p = X.shape[1]
    A = X.T @ X + lam * np.eye(p)
    b = X.T @ y
    return np.linalg.solve(A, b)

def lasso_cd_fit(X, y, lam, tol=1e-6, max_iter=10000):
    n, p = X.shape
    theta = np.zeros(p)
    col_norm = (X**2).sum(axis=0) / n
    for it in range(max_iter):
        prev = theta.copy()
        r = y - X @ theta + theta[0] * X[:, 0]
        theta[0] = r.mean()
        for j in range(1, p):
            rj = y - (X @ theta) + theta[j] * X[:, j]
            rho = (X[:, j] @ rj) / n
            denom = col_norm[j] if col_norm[j] > 0 else 1.0
            z = rho / denom
            thr = lam / (2 * n * denom)
            theta[j] = np.sign(z) * max(abs(z) - thr, 0.0)
        if np.max(np.abs(theta - prev)) < tol:
            break
    return theta

def mse(y, yhat):
    return float(np.mean((y - yhat)**2))

def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

def run_part_a(n=50, sigma=1.0, seed=1, degrees=range(0, 16)):
    rng_local = np.random.default_rng(seed)
    x = np.sort(rng_local.uniform(-1, 1, n))
    y_true = f_runge(x)
    y = y_true + rng_local.normal(0, sigma, size=n)
    xtr, ytr, xte, yte = train_test_split(x, y, test_fraction=0.2, seed=seed)

    tr_mse = []; te_mse = []; tr_r2 = []; te_r2 = []
    for d in degrees:
        Xtr = make_design(xtr, d); Xte = make_design(xte, d)
        Xtr_s, Xte_s = scale_X_pair(Xtr, Xte)
        beta = ols_fit(Xtr_s, ytr)
        tr_mse.append(mse(ytr, Xtr_s @ beta))
        te_mse.append(mse(yte, Xte_s @ beta))
        tr_r2.append(r2(ytr, Xtr_s @ beta))
        te_r2.append(r2(yte, Xte_s @ beta))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(list(degrees), tr_mse, marker='o', label='Train MSE')
    axs[0].plot(list(degrees), te_mse, marker='o', label='Test MSE')
    axs[0].set_xlabel('Polynomial degree'); axs[0].set_ylabel('MSE'); axs[0].legend()
    axs[1].plot(list(degrees), tr_r2, marker='o', label='Train R2')
    axs[1].plot(list(degrees), te_r2, marker='o', label='Test R2')
    axs[1].set_xlabel('Polynomial degree'); axs[1].set_ylabel('R^2'); axs[1].legend()
    fig.tight_layout(); fig.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/OLS_MSE_R2_vs_degree.png')

    max_abs = []
    for d in range(1, max(degrees)+1):
        Xtr = make_design(xtr, d)
        Xtr_s, _ = scale_X_pair(Xtr, Xtr)
        beta = ols_fit(Xtr_s, ytr)
        max_abs.append(np.max(np.abs(beta)))
    plt.figure(figsize=(6,4))
    plt.semilogy(range(1, max(degrees)+1), max_abs, marker='o')
    plt.xlabel('Polynomial degree'); plt.ylabel('Max |coef|'); plt.tight_layout()
    plt.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/OLS_coef_growth.png')

    return dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte)

def ols_effects_of_n_and_noise(seed=1):
    degrees = list(range(0, 16))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for n, lab in [(20, 'n=20, σ=1'), (100, 'n=100, σ=1')]:
        data = run_part_a(n=n, sigma=1.0, seed=seed, degrees=degrees)
        xtr, ytr, xte, yte = data['xtr'], data['ytr'], data['xte'], data['yte']
        te_curve = []
        for d in degrees:
            Xtr = make_design(xtr, d); Xte = make_design(xte, d)
            Xtr_s, Xte_s = scale_X_pair(Xtr, Xte)
            beta = ols_fit(Xtr_s, ytr)
            te_curve.append(mse(yte, Xte_s @ beta))
        axs[0].plot(degrees, te_curve, marker='o', label=lab)
    axs[0].set_xlabel('Polynomial degree'); axs[0].set_ylabel('Test MSE'); axs[0].legend()

    for sig, lab in [(0.0, 'σ=0'), (1.0, 'σ=1')]:
        data = run_part_a(n=50, sigma=sig, seed=seed, degrees=degrees)
        xtr, ytr, xte, yte = data['xtr'], data['ytr'], data['xte'], data['yte']
        te_curve = []
        for d in degrees:
            Xtr = make_design(xtr, d); Xte = make_design(xte, d)
            Xtr_s, Xte_s = scale_X_pair(Xtr, Xte)
            beta = ols_fit(Xtr_s, ytr)
            te_curve.append(mse(yte, Xte_s @ beta))
        axs[1].plot(degrees, te_curve, marker='o', label=lab)
    axs[1].set_xlabel('Polynomial degree'); axs[1].set_ylabel('Test MSE'); axs[1].legend()
    fig.tight_layout(); fig.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/OLS_TestMSE_panels.png')

def ridge_and_lasso_panels(seed=1, degree=15):
    rng_local = np.random.default_rng(seed)
    x = np.sort(rng_local.uniform(-1, 1, 50))
    y = f_runge(x) + rng_local.normal(0, 1.0, size=x.size)
    xtr, ytr, xte, yte = train_test_split(x, y, 0.2, seed)

    degrees = list(range(0, 16))
    lam_list = [0.0, 1e-3, 1e-2, 1e-1, 1.0]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    for lam in lam_list:
        curve = []
        for d in degrees:
            Xtr = make_design(xtr, d); Xte = make_design(xte, d)
            Xtr_s, Xte_s = scale_X_pair(Xtr, Xte)
            beta = ols_fit(Xtr_s, ytr) if lam == 0.0 else ridge_fit(Xtr_s, ytr, lam)
            curve.append(mse(yte, Xte_s @ beta))
        axs[0].plot(degrees, curve, marker='o', label=f'λ={lam:g}')
    axs[0].set_xlabel('Polynomial degree'); axs[0].set_ylabel('Test MSE'); axs[0].legend()
    axs[0].set_title('Ridge and OLS Test MSE')

    Xtr = make_design(xtr, degree); Xte = make_design(xte, degree)
    Xtr_s, Xte_s = scale_X_pair(Xtr, Xte)
    beta_ols = ols_fit(Xtr_s, ytr)
    beta_ridge = ridge_fit(Xtr_s, ytr, 0.1)

    poly = PolynomialFeatures(degree, include_bias=True)
    Xtr_poly = poly.fit_transform(xtr.reshape(-1, 1))
    lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 30),
                       fit_intercept=False, cv=5,
                       max_iter=100000, random_state=seed).fit(Xtr_poly, ytr)
    beta_lasso = lasso_cv.coef_
    idx = np.arange(degree + 1)
    axs[1].semilogy(idx, np.abs(beta_ols) + 1e-12, marker='o', label='OLS')
    axs[1].semilogy(idx, np.abs(beta_ridge) + 1e-12, marker='o', label='Ridge λ=0.1')
    axs[1].semilogy(idx, np.abs(beta_lasso) + 1e-12, marker='o', label=f'Lasso λ={lasso_cv.alpha_:.2g}')
    axs[1].set_xlabel('Coefficient index'); axs[1].set_ylabel('|coef|'); axs[1].legend()
    axs[1].set_title(f'Coefficient shrinkage degree {degree}')

    fig.tight_layout(); fig.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/Ridge_Lasso_panels.png')

def optimizers_convergence(seed=1, degree=5, iters=150, lr=0.1):
    rng_local = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 50)
    y = f_runge(x) + rng_local.normal(0, 1.0, size=x.size)
    X = make_design(x, degree)
    Xs, _ = scale_X_pair(X, X)
    n = Xs.shape[0]
    p = Xs.shape[1]

    def grad(theta):
        return (2.0 / n) * Xs.T @ (Xs @ theta - y)

    theta_gd = np.zeros(p)
    theta_mom = np.zeros(p); v_mom = np.zeros(p); gamma = 0.9
    theta_adagrad = np.zeros(p); G = np.zeros(p)
    theta_rms = np.zeros(p); Eg2 = np.zeros(p)
    theta_adam = np.zeros(p); m_ad = np.zeros(p); v_ad = np.zeros(p)

    hist = {k: [] for k in ['GD','Momentum','AdaGrad','RMSprop','ADAM']}

    for t in range(1, iters + 1):
        g_gd = grad(theta_gd)
        g_m = grad(theta_mom)
        g_adg = grad(theta_adagrad)
        g_rms = grad(theta_rms)
        g_adam = grad(theta_adam)

        hist['GD'].append(mse(y, Xs @ theta_gd))
        hist['Momentum'].append(mse(y, Xs @ theta_mom))
        hist['AdaGrad'].append(mse(y, Xs @ theta_adagrad))
        hist['RMSprop'].append(mse(y, Xs @ theta_rms))
        hist['ADAM'].append(mse(y, Xs @ theta_adam))

        theta_gd -= lr * g_gd

        v_mom = gamma * v_mom + lr * g_m
        theta_mom -= v_mom

        G += g_adg**2
        theta_adagrad -= lr * g_adg / (np.sqrt(G) + 1e-8)

        Eg2 = 0.9 * Eg2 + 0.1 * (g_rms**2)
        theta_rms -= lr * g_rms / (np.sqrt(Eg2) + 1e-8)

        m_ad = 0.9 * m_ad + 0.1 * g_adam
        v_ad = 0.999 * v_ad + 0.001 * (g_adam**2)
        m_hat = m_ad / (1 - 0.9**t)
        v_hat = v_ad / (1 - 0.999**t)
        theta_adam -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    plt.figure(figsize=(7,4))
    for k, vals in hist.items():
        plt.plot(vals, label=k)
    plt.yscale('log'); plt.xlabel('Iteration'); plt.ylabel('Training MSE')
    plt.tight_layout(); plt.legend(); plt.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/Optimizers_convergence.png')

def sgd_vs_batch(seed=1, degree=10, epochs=60, lr0=0.02, batch_size=32):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, 400); y = f_runge(x) + rng.normal(0, 1.0, x.size)
    X = make_design(x, degree); Xs, _ = scale_X_pair(X, X)
    n, p = Xs.shape

    # Batch GD
    th_b = np.zeros(p); mse_b = []
    for e in range(epochs):
        g = (2.0/n) * Xs.T @ (Xs @ th_b - y)
        th_b -= 0.05 * g
        mse_b.append(mse(y, Xs @ th_b))

    th_s = np.zeros(p); mse_s = []
    for e in range(1, epochs+1):
        lr = lr0 / np.sqrt(e)                 
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start+batch_size]
            Xi = Xs[b]; yi = y[b]
            g = (2.0/len(b)) * Xi.T @ (Xi @ th_s - yi)
            th_s -= lr * g
        mse_s.append(mse(y, Xs @ th_s))

    plt.figure(figsize=(6,4))
    plt.plot(mse_b, label='Batch GD')
    plt.plot(mse_s, label='Mini-batch SGD (decay)')
    plt.xlabel('Epoch'); plt.ylabel('Training MSE'); plt.legend(); plt.tight_layout()
    plt.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/SGD_vs_Batch.png')

def bias_variance_curve(seed=1, sigma=1.0, degrees=range(0, 16), B=200):
    rng_local = np.random.default_rng(seed)
    x_train = np.sort(rng_local.uniform(-1, 1, 50))
    y_train = f_runge(x_train) + rng_local.normal(0, sigma, size=x_train.size)
    xg = np.linspace(-1, 1, 200)
    fg = f_runge(xg)

    bias2 = []; var = []; tot = []
    for d in degrees:
        preds = np.zeros((B, xg.size))
        for b in range(B):
            idx = rng_local.integers(0, x_train.size, size=x_train.size)
            Xb = make_design(x_train[idx], d)
            yb = y_train[idx]
            Xb_s, Xg_s = scale_X_pair(Xb, make_design(xg, d))
            beta = ols_fit(Xb_s, yb)
            preds[b] = Xg_s @ beta
        mean_pred = preds.mean(axis=0)
        bias2.append(float(np.mean((fg - mean_pred)**2)))
        var.append(float(np.mean(np.var(preds, axis=0))))
        tot.append(bias2[-1] + var[-1] + sigma**2)

    plt.figure(figsize=(11,4))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(degrees, bias2, 'o-', label='Bias$^2$')
    ax1.plot(degrees, var, 'o-', label='Variance')
    ax1.set_title('Bias and variance'); ax1.set_xlabel('Polynomial degree'); ax1.set_ylabel('Error'); ax1.legend()

    ax2 = plt.subplot(1,2,2)
    ax2.plot(degrees, tot, 'o--', label='Bias$^2$ + Var + $\sigma^2$')
    ax2.set_title('Total expected error'); ax2.set_xlabel('Polynomial degree'); ax2.set_ylabel('Total error'); ax2.legend(title=f'$\sigma^2={sigma**2:.2f}$')
    plt.tight_layout(); plt.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/BiasVariance_panels.png')

def cross_validation_panels(seed=1, max_deg=15):
    rng_local = np.random.default_rng(seed)
    x = np.sort(rng_local.uniform(-1, 1, 80))
    y = f_runge(x) + rng_local.normal(0, 1.0, size=x.size)
    X = x.reshape(-1, 1)

    degs = list(range(0, max_deg + 1))
    cv_mse = []
    for d in degs:
        model = make_pipeline(PolynomialFeatures(d, include_bias=True),
                              LinearRegression(fit_intercept=False))
        scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_mse.append(np.mean(scores))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(degs, cv_mse, marker='o')
    axs[0].set_xlabel('Polynomial degree'); axs[0].set_ylabel('5-fold CV MSE')
    axs[0].set_title('Model complexity by CV')

    deg = 15
    poly = PolynomialFeatures(deg, include_bias=True)
    Xpoly = poly.fit_transform(X)
    ridge_cv = RidgeCV(alphas=np.logspace(-4, 1, 40), fit_intercept=False, cv=5).fit(Xpoly, y)
    lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 30), fit_intercept=False,
                       cv=5, max_iter=20000, random_state=seed).fit(Xpoly, y)

    text = f'Best Ridge lambda = {ridge_cv.alpha_:.6g}\nBest Lasso lambda = {lasso_cv.alpha_:.6g}'
    axs[1].axis('off'); axs[1].text(0.05, 0.7, text, fontsize=12)
    axs[1].set_title('Best hyperparameters')
    fig.tight_layout(); fig.savefig('/Users/olmaa/Desktop/Ny_mappe/Figures/CV_panels.png')

    with open('CV_best_hyperparams.txt', 'w') as fh:
        fh.write(f'Best Ridge lambda={ridge_cv.alpha_:.6g}\n')
        fh.write(f'Best Lasso lambda={lasso_cv.alpha_:.6g}\n')

if __name__ == '__main__':
    A = run_part_a()
    ols_effects_of_n_and_noise()
    ridge_and_lasso_panels()
    optimizers_convergence()
    sgd_vs_batch()
    bias_variance_curve()
    cross_validation_panels()
    plt.show()
