import unittest
import pandas as pd
import numpy as np
from scipy.stats import t, norm, kurtosis
from numpy.linalg import eigh
from numpy.random import default_rng
from scipy import stats
from scipy.integrate import quad
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

### 1.1 Covariance Missing data, skip missing rows ###
def calculate_covariance_matrix(csv_file):
    data = pd.read_csv(csv_file)
    # skip missing rows
    data_cleaned = data.dropna()
    cov_matrix = data_cleaned.cov()

    return cov_matrix


### 1.2 Correlation Missing data, skip missing rows ###
def calculate_correlation_matrix(csv_file):
    data = pd.read_csv(csv_file)
    # skip missing rows
    data_cleaned = data.dropna()
    # calculate correlation matrix
    corr_matrix = data_cleaned.corr()

    return corr_matrix


### 1.3 Covariance Missing data, Pairwise ###
def calculate_covariance_pairwise(csv_file):
    data = pd.read_csv(csv_file)
    # calculate the pairwise covariance matrix
    pairwise_cov_matrix = data.cov(min_periods=1) # minimum number of values in a row: 2

    return pairwise_cov_matrix


### 1.4 Correlation Missing data, pairwise ###
def calculate_correlation_pairwise(csv_file):
    data = pd.read_csv(csv_file)
    # calculate the pairwise covariance matrix
    pairwise_corr_matrix = data.corr(min_periods=1) # minimum number of values in a row: 2

    return pairwise_corr_matrix


### 2.1 EW Covariance, lambda=0.97 ###
def ew_cov(csv_file, lam):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    x = data_array
    m, n = x.shape
    w = np.empty(m, dtype=float)

    # Remove the mean from the series
    xm = x.mean(axis=0)
    for j in range(n):
        x[:, j] = x[:, j] - xm[j]

    # Calculate weights, going from oldest to newest
    for i in range(m):
        w[i] = (1 - lam) * lam ** (m - i - 1)  # Adjusted index for Python's 0-based indexing

    # Normalize weights to 1
    w = w / w.sum()

    # Calculate covariance matrix
    weighted_x = w[:, np.newaxis] * x  # Element-wise multiplication
    cov_matrix = weighted_x.T @ x  # Matrix multiplication

    # Convert covariance matrix numpy array to DataFrame
    cov_df = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)

    return cov_df

### 2.2 EW Correlation, lambd=0.94 ###
def ew_corr(csv_file, lam):
    # Use ew_cov function to calculate the exponentially weighted covariance matrix
    cov_matrix = ew_cov(csv_file, lam)  
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Outer product of standard deviations
    denom = np.outer(std_devs, std_devs)
    
    # Standardize covariance matrix to get the correlation matrix
    corr_matrix = cov_matrix / denom
    
    # Ensure the diagonal elements are exactly 1
    np.fill_diagonal(corr_matrix.values, 1.0)
    
    # Return correlation matrix as DataFrame
    corr_df = pd.DataFrame(corr_matrix, columns=cov_matrix.columns, index=cov_matrix.index)   
    return corr_df


### 2.3 Covariance with EW Variance (l=0.97), EW Correlation (l=0.94) ###
def ew_cov(csv_file, lam):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    x = data_array
    m, n = x.shape
    w = np.empty(m, dtype=float)

    # Remove the mean from the series
    xm = x.mean(axis=0)
    for j in range(n):
        x[:, j] = x[:, j] - xm[j]

    # Calculate weights, going from oldest to newest
    for i in range(m):
        w[i] = (1 - lam) * lam ** (m - i - 1)

    # Normalize weights to 1
    w = w / w.sum()

    # Calculate covariance matrix
    weighted_x = w[:, np.newaxis] * x
    cov_matrix = weighted_x.T @ x

    # Convert covariance matrix numpy array to DataFrame
    cov_df = pd.DataFrame(cov_matrix, columns=data.columns, index=data.columns)

    return cov_df

def calculate_adjusted_covariance(csv_file, lambda1, lambda2):
    """
    Calculate the adjusted covariance matrix using two different EW decay factors.
    
    Parameters:
    - csv_file: The path to the CSV file containing the dataset.
    - lambda1: The decay factor for extracting standard deviations.
    - lambda2: The decay factor for extracting inverse of standard deviations and calculating the final matrix.
    
    Returns:
    - final_matrix_df: The adjusted covariance matrix as a pandas DataFrame.
    """
    
    # Calculate EW covariance with lambda1 and extract standard deviations
    cout_lambda1 = ew_cov(csv_file, lambda1)
    sd1 = np.sqrt(np.diag(cout_lambda1.values))

    # Calculate EW covariance with lambda2 and extract inverse of standard deviations
    cout_lambda2 = ew_cov(csv_file, lambda2)
    sd = 1 / np.sqrt(np.diag(cout_lambda2.values))

    # Calculate the final adjusted covariance matrix
    final_matrix = np.diag(sd1) @ np.diag(sd) @ cout_lambda2.values @ np.diag(sd) @ np.diag(sd1)
    
    # Create a DataFrame from the final matrix using columns from the EW covariance calculation
    final_matrix_df = pd.DataFrame(final_matrix, columns=cout_lambda2.columns, index=cout_lambda2.index)
    
    return final_matrix_df


### 3.1 near_psd covariance ###
def near_psd_cov(csv_file, epsilon=0.0):
    data = pd.read_csv(csv_file)
    input_matrix = data.to_numpy()
    a = input_matrix  
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    # Eigen decomposition, update the eigenvalues, and scale
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs**2 @ vals[:, np.newaxis])
    T = np.diag(np.sqrt(T).flatten())
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD
    # Convert result to DataFrame
    psd_cov = pd.DataFrame(out, columns=data.columns, index=data.index)

    return psd_cov


### 3.2 near_psd correlation ###
def near_psd_corr(csv_file, epsilon=0.0):
    data = pd.read_csv(csv_file)
    input_matrix = data.to_numpy()
    a = input_matrix   
    # Eigen decomposition, update the eigenvalues, and scale
    vals, vecs = eigh(a)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs**2 @ vals[:, np.newaxis])
    T = np.diag(np.sqrt(T).flatten())
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Convert result to DataFrame
    psd_corr = pd.DataFrame(out, columns=data.columns, index=data.index)

    return psd_corr


### 3.3 Higham covariance ###
def _getPS(Rk, W):
    """Project onto the set of symmetric matrices."""
    return (Rk + Rk.T) / 2

def _getPu(Xk, W):
    """Ensure the matrix is PSD by adjusting its eigenvalues."""
    vals, vecs = np.linalg.eigh(Xk)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T

def wgtNorm(matrix, W):
    """Compute the weighted norm of a matrix."""
    return np.linalg.norm(matrix * W)

def higham_nearestPSD(csv_file, epsilon=1e-9, maxIter=100, tol=1e-9):
    data = pd.read_csv(csv_file)
    input_matrix = data.to_numpy()
    pc = input_matrix    
    n = pc.shape[0]
    W = np.diag(np.full(n, 1.0))

    deltaS = 0
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        # Ps Update
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        # Pu Update
        Yk = _getPu(Xk, W)
        # Get Norm
        norm = wgtNorm(Yk - pc, W)
        # Smallest Eigenvalue
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            # Norm converged and matrix is at least PSD
            break

        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i-1} iterations")
    
    Yk_df = pd.DataFrame(Yk, columns=data.columns, index=data.index)

    return Yk_df


### 3.4 Higham correlation ###
def _getPS(Rk, W):
    """Project onto the set of symmetric matrices."""
    return (Rk + Rk.T) / 2

def _getPu(Xk, W):
    """Ensure the matrix is PSD by adjusting its eigenvalues."""
    vals, vecs = np.linalg.eigh(Xk)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T

def wgtNorm(matrix, W):
    """Compute the weighted norm of a matrix."""
    return np.linalg.norm(matrix * W)

def higham_nearestPSD(pc, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    W = np.diag(np.full(n, 1.0))

    deltaS = 0
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        # Ps Update
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        # Pu Update
        Yk = _getPu(Xk, W)
        # Get Norm
        norm = wgtNorm(Yk - pc, W)
        # Smallest Eigenvalue
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            # Norm converged and matrix is at least PSD
            break

        norml = norm
        i += 1

    if i < maxIter:
        print(f"Converged in {i} iterations.")
    else:
        print(f"Convergence failed after {i-1} iterations")
    return Yk

def higham_nearest_psd_correlation(csv_file, epsilon=1e-9, maxIter=100, tol=1e-9):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    pc = data_array
    n = pc.shape[0]
    W = np.diag(np.full(n, 1.0))

    deltaS = 0
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        # Ps Update
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        # Pu Update
        Yk = _getPu(Xk, W)
        # Get Norm
        norm = wgtNorm(Yk - pc, W)
        # Smallest Eigenvalue
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            # Norm converged and matrix is at least PSD
            break

        norml = norm
        i += 1

    # if i < maxIter:
    #     print(f"Converged in {i} iterations.")
    # else:
    #     print(f"Convergence failed after {i-1} iterations")
    
    # Convert Yk to DataFrame
    Yk_df = pd.DataFrame(Yk, columns=data.columns, index=data.index)

    return Yk_df


### 4.1 chol_psd ###
def chol_psd(csv_file):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    a = data_array
    n = a.shape[0]
    root = np.zeros_like(a)
    
    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        temp = a[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        
        if root[j, j] == 0.0:
            root[j, (j+1):n] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j+1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    
    root = pd.DataFrame(root, columns=data.columns, index=data.index)

    return root


### 5.1 Normal Simulation PD Input 0 mean - 100,000 simulations ###
def simulate_normal_pd(csv_file):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    cov = data_array

    n_variables = cov.shape[1]
    mean_vector = np.zeros(n_variables)

    n_simulations = 100000

    rng = default_rng()

    simulated_data = rng.multivariate_normal(mean_vector, cov, n_simulations)

    sim_cov_matrix = np.cov(simulated_data, rowvar=False)

    sim_cov_matrix_df = pd.DataFrame(sim_cov_matrix, columns=data.columns, index=data.index)

    return sim_cov_matrix_df


### 5.2 Normal Simulation PSD Input 0 mean ###
def simulate_normal_psd(csv_file):
    data = pd.read_csv(csv_file)
    data_array = data.to_numpy()
    cov = data_array

    n_variables = cov.shape[1]
    mean_vector = np.zeros(n_variables)

    n_simulations = 100000

    rng = default_rng()

    simulated_data = rng.multivariate_normal(mean_vector, cov, n_simulations)

    sim_cov_matrix = np.cov(simulated_data, rowvar=False)

    sim_cov_matrix_df = pd.DataFrame(sim_cov_matrix, columns=data.columns, index=data.index)

    return sim_cov_matrix_df


### 5.3 Normal Simulation nonPSD Input, 0 mean, near_psd fix ###
def near_psd_cov_sim(input_matrix, epsilon=0.0):
    a = input_matrix  
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
    # Eigen decomposition, update the eigenvalues, and scale
    vals, vecs = eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs**2 @ vals[:, np.newaxis])
    T = np.diag(np.sqrt(T).flatten())
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T
    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def simulate_normal_near_psd(csv_file):
    data = pd.read_csv(csv_file)
    cov = data.to_numpy()

    # Ensure the covariance matrix is positive semi-definite
    psd_cov = near_psd_cov_sim(cov)

    n_variables = psd_cov.shape[1]
    mean_vector = np.zeros(n_variables)

    n_simulations = 100000

    rng = default_rng()

    simulated_data = rng.multivariate_normal(mean_vector, psd_cov, n_simulations)

    sim_cov_matrix = np.cov(simulated_data, rowvar=False)

    sim_cov_matrix_df = pd.DataFrame(sim_cov_matrix, columns=data.columns, index=data.columns)

    return sim_cov_matrix_df


### 5.4 Normal Simulation PSD Input, 0 mean, higham fix ###
def _getPS_sim(Rk, W):
    return (Rk + Rk.T) / 2

def _getPu_sim(Xk, W):
    vals, vecs = np.linalg.eigh(Xk)
    vals[vals < 0] = 0
    return vecs @ np.diag(vals) @ vecs.T

def wgtNorm_sim(matrix, W):
    return np.linalg.norm(matrix * W)

def higham_nearestPSD_sim(matrix, epsilon=1e-9, maxIter=100, tol=1e-9):
    pc = matrix
    n = pc.shape[0]
    W = np.diag(np.full(n, 1.0))

    deltaS = 0
    Yk = pc.copy()
    norml = np.inf
    i = 1

    while i <= maxIter:
        Rk = Yk - deltaS
        Xk = _getPS_sim(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu_sim(Xk, W)
        norm = wgtNorm_sim(Yk - pc, W)
        minEigVal = np.min(np.real(np.linalg.eigvals(Yk)))

        if norm - norml < tol and minEigVal > -epsilon:
            break

        norml = norm
        i += 1

    return Yk

def simulate_normal_higham_psd(csv_file):
    data = pd.read_csv(csv_file)
    cov = data.to_numpy()

    psd_cov = higham_nearestPSD_sim(cov)

    n_variables = psd_cov.shape[1]
    mean_vector = np.zeros(n_variables)

    n_simulations = 100000

    rng = default_rng()

    simulated_data = rng.multivariate_normal(mean_vector, psd_cov, n_simulations)

    sim_cov_matrix = np.cov(simulated_data, rowvar=False)

    sim_cov_matrix_df = pd.DataFrame(sim_cov_matrix, columns=data.columns, index=data.columns)

    return sim_cov_matrix_df


### 5.5 PCA Simulation, 99% explained, 0 mean - 100,000 simulations ###
def pca_simulation(input_file_path):
    # Load the input CSV file
    input_data = pd.read_csv(input_file_path)
    
    # 1. Data Standardization
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_data = scaler.fit_transform(input_data)
    
    # 2. Execute PCA
    pca = PCA(n_components=0.99)
    pca_result = pca.fit_transform(scaled_data)
    
    # 3. Inverse Transform the PCA results to attempt to reconstruct the original data
    reconstructed_data = pca.inverse_transform(pca_result)
    
    # Scale back the data to original distribution
    reconstructed_original_scale = scaler.inverse_transform(reconstructed_data)
    
    # Create a DataFrame of the reconstructed data
    reconstructed_df = pd.DataFrame(reconstructed_original_scale, columns=input_data.columns)
    
    return reconstructed_df

# def simulate_pca(csv_file, nval=None, explained_variance=0.99):
#     data = pd.read_csv(csv_file, header=None, skiprows=1)
#     a = data.to_numpy(dtype=float)
    
#     # Check if the matrix is square
#     if a.shape[0] != a.shape[1]:
#         raise ValueError(f"Input matrix must be square for eigendecomposition. Current shape: {a.shape}")

#     nsim = 100000
#     # Eigenvalue decomposition
#     vals, vecs = eigh(a)

#     # Sort eigenvalues and eigenvectors
#     idx = vals.argsort()[::-1]
#     vals = vals[idx]
#     vecs = vecs[:,idx]

#     # Calculate total variance
#     tv = np.sum(vals)

#     # Find position of eigenvalues that contribute to the specified explained variance
#     cumvals = np.cumsum(vals) / tv
#     if nval is not None:
#         posv = np.where(cumvals <= explained_variance)[0]
#         if len(posv) >= nval:
#             posv = posv[:nval]
#     else:
#         posv = np.where(cumvals <= explained_variance)[0]
#         if len(posv) < 1:
#             posv = np.array([len(cumvals) - 1])  # Ensure at least one value is chosen

#     # Select the eigenvectors and eigenvalues that explain the specified variance
#     vals = vals[posv]
#     vecs = vecs[:,posv]

#     print(f"Simulating with {len(posv)} PC Factors: {np.sum(vals)/tv*100:.2f}% total variance explained")

#     # Create matrix B
#     B = vecs @ np.diag(np.sqrt(vals))
#     r = norm.rvs(size=(len(vals), nsim))
#     simulated_data = B @ r
#     sim_cov_matrix = np.cov(simulated_data, rowvar=False)

#     # Convert the covariance matrix to a DataFrame before returning
#     cov_df = pd.DataFrame(sim_cov_matrix)

#     return cov_df


### 6.1 calculate arithmetic returns ###
def calculate_arithmetic_returns(csv_file):
    data = pd.read_csv(csv_file)

    arithmetic_returns = data.iloc[:, 1:].pct_change() 
    arithmetic_returns['Date'] = data['Date']
    arithmetic_returns = arithmetic_returns[['Date'] + [col for col in arithmetic_returns.columns if col != 'Date']]
    # Drop rows with NaN values and reset index
    arithmetic_returns = arithmetic_returns.dropna().reset_index(drop=True)
    
    return arithmetic_returns


### 6.2 calculate log returns ###
def calculate_log_returns(csv_file):
    data = pd.read_csv(csv_file)
    # Since the data structure of test6.csv is already known, we can directly calculate the log returns
    log_returns = np.log(data.iloc[:, 1:] / data.iloc[:, 1:].shift(1))

    # Add the date column back to the log returns dataframe
    log_returns['Date'] = data['Date']

    # Reorder the columns to have Date as the first column
    log_returns = log_returns[['Date'] + [col for col in log_returns.columns if col != 'Date']]
    log_returns = log_returns.dropna()

    return log_returns


### 7.1 Fit Normal Distribution ###
def fit_normal_distn(csv_file):
    data = pd.read_csv(csv_file)

    # Calculate mean and standard deviation for x1
    mean_x1 = data['x1'].mean()
    std_dev_x1 = data['x1'].std()
    
    # create DataFrame to store the results
    result_df = pd.DataFrame({'mu': [mean_x1], 'sigma': [std_dev_x1]})

    return result_df

### 7.2 Fit T Distribution ###
def fit_t_distn(csv_file):
    data = pd.read_csv(csv_file)
    # nu, mu, sigma = t.fit(data)
    # result_df = pd.DataFrame({'mu': [mu], 'sigma': [sigma], "nu":[nu]})
    df, loc, scale = stats.t.fit(data)
    result_df = pd.DataFrame({'mu': [loc], 'sigma': [scale], 'nu': [df]})

    return result_df


### 7.3 T Regression ###
class FittedModel:
    def __init__(self, beta, error_model, eval_model, errors, u):
        self.beta = beta
        self.error_model = error_model
        self.eval_model = eval_model
        self.errors = errors
        self.u = u
        
    def display_results(self):
        mu = self.error_model.mean()
        sigma = self.error_model.std()
        nu = self.error_model.df
        Alpha = self.beta[0]
        B1, B2, B3 = self.beta[1:]
        
def fit_regression_t(csvfile):
    # Load the input data
    input_data = pd.read_csv(csvfile)

    # Extract x and y values from the input data
    x = input_data[['x1', 'x2', 'x3']].values
    y = input_data['y'].values

    n = x.shape[0]

    def gtl(params):
        m, s, nu, beta = params[0], params[1], params[2], params[3:]
        xm = y - np.dot(np.hstack((np.ones((n, 1)), x)), beta)
        return -np.sum(t.logpdf(xm / s, nu) - np.log(s))

    b_start = np.linalg.inv(np.dot(np.transpose(np.hstack((np.ones((n, 1)), x))), np.hstack((np.ones((n, 1)), x)))) \
              .dot(np.transpose(np.hstack((np.ones((n, 1)), x)))).dot(y)
    e = y - np.hstack((np.ones((n, 1)), x)).dot(b_start)
    start_m = np.mean(e)
    start_nu = 6.0 / kurtosis(e, fisher=False) + 4
    start_s = np.sqrt(np.var(e) * (start_nu - 2) / start_nu)

    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))

    res = minimize(gtl, initial_params, method='Nelder-Mead')
    m, s, nu, beta = res.x[0], res.x[1], res.x[2], res.x[3:]

    error_model = t(df=nu, loc=m, scale=s)

    def eval_model(x, u):
        xm = np.dot(np.hstack((np.ones((x.shape[0], 1)), x)), beta)
        errors = error_model.ppf(u)
        return xm + errors

    errors = y - eval_model(x, np.full(x.shape[0], 0.5))
    u = error_model.cdf(errors)

    
    # Create DataFrame with the results
    results_df = pd.DataFrame({
        'mu': [m],
        'sigma': [s],
        'nu': [nu],
        'Alpha': [beta[0]],
        'B1': [beta[1]],
        'B2': [beta[2]],
        'B3': [beta[3]]
    })

    return results_df


### all work in this one function below ###
### 8.1 Var from Normal Distribution ###
### 8.2 Var from T Distribution ###
### 8.3 VaR from Simulation -- compare to 8.2 values ###
def calculate_var(csv_file, method="Normal", lambda_ewv=0.94, alpha=0.05):
    returns = pd.read_csv(csv_file)

    returns_mean, returns_std = returns.mean(), returns.std()

    if method == "Normal":
        neg_VaR = norm.ppf(alpha, returns_mean, returns_std)
        VaR_diff_from_mean = returns_mean - neg_VaR
        VaR_abs = abs(neg_VaR)
        result_df = pd.DataFrame({'VaR Absolute': [VaR_abs], 'VaR Diff from Mean': [VaR_diff_from_mean]}, dtype=float)

    elif method == "T_DIST":
#         params = t.fit(results)
#         var = t.ppf(alpha, *params)
        params = t.fit(returns)
        df, loc, scale = params[0], params[1], params[2]
        neg_VaR = loc + t.ppf(0.05, df) * scale
        VaR_diff_from_mean = returns_mean - neg_VaR
        VaR_abs = abs(neg_VaR)
        result_df = pd.DataFrame({'VaR Absolute': [VaR_abs], 'VaR Diff from Mean': [VaR_diff_from_mean]}, dtype=float)

    elif method == "HISTORICAL":
#         var = np.percentile(results, 5)
        neg_VaR = np.percentile(returns, 5)
        VaR_diff_from_mean = returns_mean - neg_VaR
        VaR_abs = abs(neg_VaR)
        result_df = pd.DataFrame({'VaR Absolute': [VaR_abs], 'VaR Diff from Mean': [VaR_diff_from_mean]}, dtype=float)

    else:
        raise ValueError(f"Unsupported method: {method}")

    return result_df


### 8.4 ES From Normal Distribution ###
def calculate_normal_es(csv_file, col, confidence_level):
    data = pd.read_csv(csv_file)
    # mean and std dev
    mean = data[col].mean()
    std_dev = data[col].std()

    VaR = norm.ppf(1 - confidence_level, mean, std_dev)
    ES = mean - std_dev * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
    ES_abs = abs(ES)
    ES_Diff_from_Mean = mean + ES_abs
    
    result_df = pd.DataFrame({'ES Absolute': [ES_abs], 'ES Diff from Mean': [ES_Diff_from_Mean]}, dtype=float)

    return result_df

### 8.5 ES from T Distribution ###
def calculate_t_es(csv_file, alpha=0.05):
    data = pd.read_csv(csv_file)

    df, loc, scale = stats.t.fit(data)
    mean = data.mean()

    t_alpha = stats.t.ppf(alpha, df, loc=loc, scale=scale)

    es, _ = quad(lambda x: x * stats.t.pdf(x, df, loc=loc, scale=scale), -np.inf, t_alpha)
    ES_abs = -es / alpha
    ES_Diff_from_Mean = abs(ES_abs + mean) 
    
    result_df = pd.DataFrame({'ES Absolute': [ES_abs], 'ES Diff from Mean': [ES_Diff_from_Mean]}, dtype=float)

    return result_df

### 8.6 ES from Simulation ###
def calculate_sim_es(file_path):
    file = pd.read_csv(file_path)
    data = file.iloc[:, 0].values

    params = t.fit(data)

    sim = t.rvs(*params, size=10000)

    def ES(data, alpha=0.05):
        sorted_losses = np.sort(data)
        cutoff_index = int(alpha * len(sorted_losses))
        return -np.mean(sorted_losses[:cutoff_index])

    es_absolute = ES(sim)
    es_diff_from_mean = ES(sim - np.mean(sim))


    result_df = pd.DataFrame({
        "ES Absolute": [es_absolute],
        "ES Diff from Mean": [es_diff_from_mean]
    })

    return result_df


### 9.1 VaR/ES on 2 levels from simulated values - Copula ###
def sim_var_es_copula(returns_f, portfolio_f):
    returns_data = pd.read_csv(returns_f)
    portfolio_data = pd.read_csv(portfolio_f)
    mean_a = np.mean(returns_data['A'])
    std_a = np.std(returns_data['A'], ddof=1)
    df_b, loc_b, scale_b = stats.t.fit(returns_data['B'])
    
    
    nsim = 100000
    corr_coeff, _ = spearmanr(returns_data['A'],returns_data['B'])
    corr_matrix = np.array([[1, corr_coeff], [corr_coeff, 1]])
    e_vals, e_vecs = eigh(corr_matrix)
    random_vars = np.random.randn(nsim,2)
    
    pca_factors = (e_vecs * np.sqrt(e_vals)).dot(random_vars.T).T
    
    corr_normals = stats.norm.ppf(stats.norm.cdf(pca_factors))
    
    sim_rtn_a = mean_a + std_a * corr_normals[:,0]
    sim_rtn_b = loc_b+scale_b * stats.t.ppf(stats.norm.cdf(corr_normals[:, 1]), df_b)
    
    sim_rtn = pd.DataFrame({'A': sim_rtn_a, 'B': sim_rtn_b})
    
    iterations = np.arange(nsim) + 1
    portfolio_data['currentValue'] = portfolio_data['Holding'] * portfolio_data['Starting Price']
    values = pd.merge(portfolio_data, pd.DataFrame({'iteration': iterations}), how='cross')
    values['simulatedValue'] = values.apply(lambda row: row['currentValue'] * (1 + sim_rtn.loc[row['iteration']-1, row['Stock']]), axis=1)
    values['pnl'] = values['simulatedValue'] - values['currentValue']
    risk = aggRisk(values, ['Stock'])
    
    return risk

def aggRisk(values, group_by_columns):
    risk_metrics_data = []
    grouped = values.groupby(group_by_columns)
    
    for name, group in grouped:
        name = name[0] if isinstance(name, tuple) and len(group_by_columns) == 1 else name
        metrics = calculate_metrics(group)
        metrics['Stock'] = name
        risk_metrics_data.append(metrics)
    
    total_val = values['currentValue'].sum()/100000
    total_pnl = values.groupby('iteration')['pnl'].sum().reset_index(name='pnl')
    total_metrics = calculate_metrics(total_pnl, is_total=True, total_val=total_val)
    total_metrics['Stock']='Total'
    risk_metrics_data.append(total_metrics)
    risk_metrics = pd.DataFrame(risk_metrics_data, columns=['Stock', 'VaR95', 'ES95', 'VaR95_Pct', 'ES95_Pct'])
    
    return risk_metrics

def calculate_metrics(group, is_total=False, total_val=None):
    sorted_pnl = group['pnl'].sort_values()
    var_95 = sorted_pnl.quantile(0.05)
    es_95 = sorted_pnl[sorted_pnl <= var_95].mean()
    
    if is_total:
        current_value = total_val
    else:
        current_value = group['currentValue'].iloc[0]
    
    var_95_pct = abs(var_95) / current_value
    es_95_pct =  abs(es_95) / current_value
    
    return{
        'VaR95': abs(var_95),
        'ES95': abs(es_95),
        'VaR95_Pct': abs(var_95_pct), 
        'ES95_Pct': abs(es_95_pct)
    }

