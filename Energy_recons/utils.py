import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib 

def compute_correction_coefficients(sin_alpha, air_density, target_factor, degree=3):
    """
    Compute polynomial correction coefficients for normalized amplitude.

    The target quantity is the normalized amplitude:
        A_norm = A / (E * sin_alpha)
    which still has residual dependence on sin_alpha and air density.
    This function fits a polynomial model to correct for these second-order effects.

    Args:
        sin_alpha (np.ndarray): geomagnetic angle factor, shape (n,)
        air_density (np.ndarray): air density at the emission point Xsource, shape (n,)
        target_factor (np.ndarray): normalization scaling factor (A_norm), shape (n,)
        degree (int): degree of the polynomial for regression (default=3)

    Returns:
        poly_transformer (PolynomialFeatures): fitted polynomial transformer
        linreg_model (LinearRegression): trained linear regression model
        corrected_prediction (np.ndarray): predicted correction factor for the training set
    """
    # Combine features into a 2D array for polynomial regression
    features = np.column_stack([sin_alpha, air_density])  # shape (n, 2)

    # Create polynomial features
    poly_transformer = PolynomialFeatures(degree=degree)
    features_poly = poly_transformer.fit_transform(features)

    # Fit linear regression to predict the normalized amplitude
    linreg_model = LinearRegression()
    linreg_model.fit(features_poly, target_factor)

    # Predict correction factors on the training set
    corrected_prediction = linreg_model.predict(features_poly)

    return poly_transformer, linreg_model, corrected_prediction

def apply_correction_coefficients(sin_alpha, air_density, poly_transformer, linreg_model):
    """
    Apply precomputed polynomial correction coefficients to new data.

    This function uses the fitted polynomial transformer and linear regression model
    to compute the corrected amplitude factor for new events:
        A_corr = f(sin_alpha, air_density)
    which accounts for second-order dependencies on geomagnetic angle and air density.

    Args:
        sin_alpha (np.ndarray): geomagnetic angle factor for new events, shape (n,)
        air_density (np.ndarray): air density at the emission point, shape (n,)
        poly_transformer (PolynomialFeatures): fitted polynomial transformer
        linreg_model (LinearRegression): trained linear regression model
    
    Returns:
        corrected_prediction (np.ndarray): predicted correction factor for the new dataset
    """
    # Combine features into a 2D array for polynomial transformation
    features_new = np.column_stack([sin_alpha, air_density])

    # Transform features using the fitted polynomial transformer
    features_poly_new = poly_transformer.transform(features_new)

    # Predict corrected amplitude factor
    corrected_prediction = linreg_model.predict(features_poly_new)

    return corrected_prediction

def split_events(event_names, seed=42):
    """
    Split a list or array of event names into two random sets.
    
    Parameters
    ----------
    event_names : list or np.ndarray
        List or array of event IDs/names.
    seed : int
        Seed for reproducibility.
    
    Returns
    -------
    set1, set2 : np.ndarray
        Two arrays containing the split events.
    """
    
    # Keep only unique event names
    unique_events = np.unique(event_names)
    
    # Shuffle the unique events
    np.random.seed(seed)
    np.random.shuffle(unique_events)
    
    # Split in half
    half_size = len(unique_events) // 2
    set1 = unique_events[:half_size]
    set2 = unique_events[half_size:]
    return set1, set2

