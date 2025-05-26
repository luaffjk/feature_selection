import numpy as np
import pandas as pd
from scipy.stats import chisquare, ks_2samp
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split

def detect_feature_drift(train_series, test_series, feature_type='continuous', bins=10):
    """
    Detects distribution drift between train and test for a given feature.
    
    Args:
        train_series (pd.Series): The feature values from the training set.
        test_series (pd.Series): The feature values from the test/production set.
        feature_type (str): 'continuous' or 'categorical'.
        bins (int): Number of bins to use for continuous variables (for better visualization).
        
    Returns:
        dict: Test statistic, p-value, and a drift flag.
    """
    if feature_type == 'categorical':
        train_counts = train_series.value_counts().sort_index()
        test_counts = test_series.value_counts().sort_index()
        # Align indexes
        all_categories = sorted(set(train_counts.index) | set(test_counts.index))
        train_freqs = train_counts.reindex(all_categories, fill_value=0)
        test_freqs = test_counts.reindex(all_categories, fill_value=0)
        stat, p = chisquare(f_obs=test_freqs, f_exp=train_freqs + 1e-8)  # add small value to avoid zero division
        drift = p < 0.05
        return {'test': 'chi-squared', 'statistic': stat, 'p_value': p, 'drift': drift}
    
    elif feature_type == 'continuous':
        stat, p = ks_2samp(train_series, test_series)
        drift = p < 0.05
        return {'test': 'ks', 'statistic': stat, 'p_value': p, 'drift': drift}
    else:
        raise ValueError("feature_type must be 'continuous' or 'categorical'")

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

dict_feature = {}
for i, name in enumerate(list(data.target_names)):
    # feature 1 
    train_feature = X_train[:, i]
    test_feature = X_test[:, i]

    # For continuous feature
    result = detect_feature_drift(pd.Series(train_feature), pd.Series(test_feature), feature_type='continuous')
    dict_feature[name] = result

df_drift = pd.DataFrame(dict_feature).T
df_drift
