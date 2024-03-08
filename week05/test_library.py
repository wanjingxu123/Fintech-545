import unittest
import pandas as pd
import numpy as np
from library import *

class TestCalculateCovarianceMatrix(unittest.TestCase):
    def test_covariance_matrix(self):
        result = calculate_covariance_matrix('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test1.csv') 
        result.to_csv('test1.1.csv', index=False)
        data = pd.read_csv("test1.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.1.csv') 
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 1.1 passed: two results are identical")

    def test_calculate_correlation_matrix(self):
        result = calculate_correlation_matrix('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test1.csv') 
        result.to_csv('test1.2.csv', index=False)
        data = pd.read_csv("test1.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.2.csv') 
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 1.2 passed: two results are identical")

    def test_calculate_covariance_pairwise(self):
        result = calculate_covariance_pairwise('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test1.csv') 
        result.to_csv('test1.3.csv', index=False)
        data = pd.read_csv("test1.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.3.csv') 
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 1.3 passed: two results are identical")


    def test_calculate_correlation_pairwise(self):
        result = calculate_correlation_pairwise('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test1.csv') 
        result.to_csv('test1.4.csv', index=False)
        data = pd.read_csv("test1.4.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.4.csv') 
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 1.4 passed: two results are identical")


    def test_ew_cov(self):
        result = ew_cov('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test2.csv', lam=0.97) 
        result.to_csv('test2.1.csv', index=False)
        data = pd.read_csv("test2.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_2.1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 2.1 passed: two results are identical")


    def test_ew_corr(self):
        result = ew_corr('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test2.csv', lam=0.94) 
        result.to_csv('test2.2.csv', index=False)
        data = pd.read_csv("test2.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_2.2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 2.2 passed: two results are identical")

    def test_calculate_adjusted_covariance(self):
        result = calculate_adjusted_covariance('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test2.csv',  lambda1=0.97, lambda2=0.94)
        result.to_csv('test2.3.csv', index=False)
        data = pd.read_csv("test2.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)
        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_2.3.csv') 
        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 2.3 passed: two results are identical")

    def test_near_psd_cov(self):
        result = near_psd_cov('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.3.csv', epsilon=0.0) 
        result.to_csv('test3.1.csv', index=False)
        data = pd.read_csv("test3.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_3.1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 3.1 passed: two results are identical")

    def test_near_psd_corr(self):
        result = near_psd_corr('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.4.csv', epsilon=0.0) 
        result.to_csv('test3.2.csv', index=False)
        data = pd.read_csv("test3.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_3.2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 3.2 passed: two results are identical")

    def test_higham_near_psd_covariance(self):
        result = higham_nearest_psd_correlation('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.3.csv') 
        result.to_csv('test3.3.csv', index=False)
        data = pd.read_csv("test3.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_3.3.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-1, rtol=1e-1)

        print("test 3.3 passed: two results are identical")

    def test_higham_nearest_psd_correlation(self):
        result = higham_nearest_psd_correlation('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_1.4.csv') 
        result.to_csv('test3.4.csv', index=False)
        data = pd.read_csv("test3.4.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_3.4.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 3.4 passed: two results are identical")


    def test_chol_psd(self):
        result = chol_psd('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_3.1.csv') 
        result.to_csv('test4.1.csv', index=False)
        data = pd.read_csv("test4.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_4.1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 4.1 passed: two results are identical")

    def test_simulate_normal_pd(self):
        result = simulate_normal_pd('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test5_1.csv') 
        result.to_csv('test5.1.csv', index=False)
        data = pd.read_csv("test5.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_5.1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-2, rtol=1e-2)

        print("test 5.1 passed: two results are identical")


    def test_simulate_normal_psd(self):
        result = simulate_normal_psd('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test5_2.csv') 
        result.to_csv('test5.2.csv', index=False)
        data = pd.read_csv("test5.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_5.2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-2, rtol=1e-2)

        print("test 5.2 passed: two results are identical")


    def test_simulate_normal_near_psd(self):
        result = simulate_normal_near_psd('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test5_3.csv') 
        result.to_csv('test5.3.csv', index=False)
        data = pd.read_csv("test5.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_5.3.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-3, rtol=1e-3)

        print("test 5.3 passed: two results are identical")

    def test_simulate_normal_higham_psd(self):
        result = simulate_normal_higham_psd('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test5_3.csv') 
        result.to_csv('test5.4.csv', index=False)
        data = pd.read_csv("test5.4.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_5.4.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-1, rtol=1e-1)

        print("test 5.4 passed: two results are identical")

    def test_pca_simulation(self):
        result = pca_simulation('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test5_2.csv') 
        result.to_csv('test5.5.csv', index=False)
        data = pd.read_csv("test5.5.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout_5.5.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-3, rtol=1e-3)

        print("test 5.5 passed: two results are identical")




    def test_calculate_arithmetic_returns(self):
        result = calculate_arithmetic_returns('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test6.csv') 
        result.to_csv('test6.1.csv', index=False)
        data = pd.read_csv("test6.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test6_1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 6.1 passed: two results are identical")

    def test_calculate_log_returns(self):
        result = calculate_log_returns('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test6.csv') 
        result.to_csv('test6.2.csv', index=False)
        data = pd.read_csv("test6.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test6_2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 6.2 passed: two results are identical")

    def test_fit_normal_distn(self):
        result = fit_normal_distn('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_1.csv') 
        result.to_csv('test7.1.csv', index=False)
        data = pd.read_csv("test7.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout7_1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 7.1 passed: two results are identical")

    def test_fit_t_distn(self):
        result = fit_t_distn('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_2.csv') 
        result.to_csv('test7.2.csv', index=False)
        data = pd.read_csv("test7.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout7_2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 7.2 passed: two results are identical")

    def test_fit_regression_t(self):
        result = fit_regression_t('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_3.csv') 
        result.to_csv('test7.3.csv', index=False)
        data = pd.read_csv("test7.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout7_3.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 7.3 passed: two results are identical")

    def test_calculate_var_normal(self):
        result = calculate_var('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_1.csv', method="Normal") 
        result['VaR Absolute'] = result['VaR Absolute'].apply(lambda x: round(x[0], 10))
        result['VaR Diff from Mean'] = result['VaR Diff from Mean'].apply(lambda x: round(x[0], 10))

        result.to_csv('test8.1.csv', index=False)
        data = pd.read_csv("test8.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result, atol=1e-5, rtol=1e-5)

        print("test 8.1 passed: two results are identical")

    def test_calculate_var_t(self):
        result = calculate_var('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_2.csv', method="T_DIST") 
        result['VaR Absolute'] = result['VaR Absolute'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))
        result['VaR Diff from Mean'] = result['VaR Diff from Mean'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))

        result.to_csv('test8.2.csv', index=False)
        data = pd.read_csv("test8.2.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_2.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=1e-4, rtol=1e-4)

        print("test 8.2 passed: two results are identical")


    def test_calculate_var_h(self):
        result = calculate_var('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_2.csv', method="HISTORICAL") 
        result['VaR Absolute'] = result['VaR Absolute'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))
        result['VaR Diff from Mean'] = result['VaR Diff from Mean'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))

        result.to_csv('test8.3.csv', index=False)
        data = pd.read_csv("test8.3.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_3.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=1e-2, rtol=1e-2)

        print("test 8.3 passed: two results are identical")

    def test_calculate_normal_es(self):
        result = calculate_normal_es('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_1.csv', col='x1', confidence_level=0.95) 
        result['ES Absolute'] = result['ES Absolute'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))
        result['ES Diff from Mean'] = result['ES Diff from Mean'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))

        result.to_csv('test8.4.csv', index=False)
        data = pd.read_csv("test8.4.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_4.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=1e-4, rtol=1e-4)

        print("test 8.4 passed: two results are identical")


    def test_calculate_t_es(self):
        result = calculate_t_es('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_2.csv') 
        result['ES Absolute'] = result['ES Absolute'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))
        result['ES Diff from Mean'] = result['ES Diff from Mean'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))

        result.to_csv('test8.5.csv', index=False)
        data = pd.read_csv("test8.5.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_5.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=1e-4, rtol=1e-4)

        print("test 8.5 passed: two results are identical")


    def test_calculate_sim_es(self):
        result = calculate_sim_es('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test7_2.csv') 
        result['ES Absolute'] = result['ES Absolute'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))
        result['ES Diff from Mean'] = result['ES Diff from Mean'].apply(lambda x: round(x[0], 6) if isinstance(x, tuple) else round(x, 6))

        result.to_csv('test8.6.csv', index=False)
        data = pd.read_csv("test8.6.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout8_6.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=1e-2, rtol=1e-2)

        print("test 8.6 passed: two results are identical")
    
    def test_sim_var_es_copula(self):
        result = sim_var_es_copula('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test9_1_returns.csv', '/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/test9_1_portfolio.csv') 
        result.to_csv('test9.1.csv', index=False)
        data = pd.read_csv("test9.1.csv", header=0, index_col=0)
        data.reset_index(inplace=True)

        expected_result = pd.read_csv('/Users/qianduoduo/Documents/fintech512/fintech-512-assignments/week07_545w5/data/testout9_1.csv') 

        # Perform numerical comparison between data and expected_result
        pd.testing.assert_frame_equal(data, expected_result,  atol=2, rtol=2)

        print("test 9.1 passed: two results are identical")
    





# Run the test case
if __name__ == "__main__":
    unittest.main()