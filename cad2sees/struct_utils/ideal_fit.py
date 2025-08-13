"""
Ideal fit module for multi-linear curve approximation.

Provides functionality for fitting trilinear curves to moment-curvature
or force-displacement data for idealised plastic hinge models.

Key features:
- Trilinear curve fitting with optimisation
- Automatic sign handling for negative curves
- Peak detection and preservation
- Post-peak behaviour modelling
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize


def multi_linear(X, Y, Flags):
    """
    Create a trilinear idealisation of a nonlinear curve.

    Fits a trilinear curve to input data for creating idealised plastic
    hinge models from moment-curvature or force-displacement relationships.

    The trilinear model consists of:
    1. Initial elastic segment
    2. Yielding/hardening segment
    3. Post-peak/softening segment
    4. Residual strength segment

    Parameters
    ----------
    X : array_like
        Independent variable data (e.g., curvature, displacement)
    Y : array_like
        Dependent variable data (e.g., moment, force)
    Flags : array_like
        Control parameters [initial yield point X, end point X]

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - IdealX: X-coordinates of idealised curve control points
        - IdealY: Y-coordinates of idealised curve control points
        
        Both arrays have 5 points: [origin, yield, peak, end, residual]

    Notes
    -----
    Automatically handles negative curves and preserves initial stiffness,
    peak strength, and post-peak behaviour through optimisation.
    """
    def min_positive_index(arr):
        """
        Find the index of the minimum non-negative value in an array.

        Parameters
        ----------
        arr : array_like
            Input array to search

        Returns
        -------
        int or None
            Index of minimum non-negative value, or None if not found
        """
        # Filter out negative values
        positive_values = arr[arr >= 0]
        # Find the minimum non-negative value
        if len(positive_values) > 0:
            min_positive_val = np.min(positive_values)
            # Find the index of the minimum non-negative value
            min_positive_idx = np.where(arr == min_positive_val)[0][0]
            return min_positive_idx
        else:
            return None

    def trilinear_curve(xs, *params):
        """
        Generate trilinear curve values for given x-coordinates.

        Parameters
        ----------
        xs : array_like
            X-coordinates for curve evaluation
        *params : tuple
            Parameters (XLim1, XLim2) defining transition points

        Returns
        -------
        np.ndarray
            Y-values of trilinear curve at given x-coordinates
        """
        XLim1, XLim2 = params
        y = np.zeros(xs.shape)

        # Define segments of the trilinear curve
        map1 = (xs <= XLim1)  # Initial elastic segment
        map2 = (xs > XLim1) & (xs <= XLim2)  # Hardening segment
        map3 = (xs > XLim2) & (xs <= X_end)  # Post-peak segment
        map4 = ~(map1 | map2 | map3)  # Beyond end point

        # Calculate y-values for each segment
        y[map1] = xs[map1] * E_iniIdeal
        y[map2] = (XLim1 * E_iniIdeal +
                   (xs[map2] - XLim1) * (YPeak - XLim1 * E_iniIdeal) /
                   (XLim2 - XLim1))
        y[map3] = (YPeak + (xs[map3] - XLim2) * (Y_end - YPeak) /
                   (X_end - XLim2))
        y[map4] = 0  # Zero beyond end point
        return y

    def trilinear_curve_obj(params, xs, ys):
        """
        Objective function for trilinear curve optimisation.

        Calculates the sum of squared errors between the trilinear
        approximation and the original data points.

        Parameters
        ----------
        params : tuple
            Parameters (XLim1, XLim2) defining transition points
        xs : array_like
            X-coordinates of data points
        ys : array_like
            Y-coordinates of data points

        Returns
        -------
        float
            Sum of squared errors between fitted and original curve
        """
        XLim1, XLim2 = params
        XLim2 = min(XLim2, X_end)  # Constrain XLim2 to end point
        y = np.zeros(xs.shape)

        # Define segments of the trilinear curve
        map1 = (xs <= XLim1)  # Initial elastic segment
        map2 = (xs > XLim1) & (xs <= XLim2)  # Hardening segment
        map3 = (xs > XLim2) & (xs <= X_end)  # Post-peak segment
        map4 = ~(map1 | map2 | map3)  # Beyond end point

        # Calculate y-values for each segment
        y[map1] = xs[map1] * E_iniIdeal
        y[map2] = (XLim1 * E_iniIdeal +
                   (xs[map2] - XLim1) * (YPeak - XLim1 * E_iniIdeal) /
                   (XLim2 - XLim1))
        y[map3] = (YPeak + (xs[map3] - XLim2) * (Y_end - YPeak) /
                   (X_end - XLim2))
        y[map4] = 0  # Zero beyond end point

        # Return sum of squared errors
        return np.sum((y - ys)**2)

    # Determine sign of Y and X data for proper handling of negative curves
    if max(Y) != max(abs(Y)):
        SignY = -1
    else:
        SignY = 1

    if max(X) != max(abs(X)):
        SignX = -1
    else:
        SignX = 1

    # Work with positive values internally
    Y *= SignY
    X *= SignX
    Flags = abs(np.array(Flags))

    # Determine end point for curve fitting
    X_end = Flags[1]
    if X_end > max(X):
        X_end = max(X)

    # Find actual end point and corresponding Y value
    end_idx = min_positive_index(X - X_end) - 1
    Y_end = Y[end_idx]
    X_end = X[end_idx]

    # Extract data up to end point for fitting
    XS = X[:end_idx]
    YS = Y[:end_idx]

    # Find peak value before collapse
    YPreCol = Y[:min_positive_index(X - X_end)]
    YPeak = max(YPreCol)

    # Cap values above peak to avoid numerical issues
    Y[Y > YPeak] = 0.1 * YPeak

    # Calculate idealized initial stiffness
    X_iniReal = Flags[0]
    real_yield_idx = np.argwhere(X == X_iniReal)[0][0]

    # Create interpolation function for pre-yield behavior
    YXCurve_PreYield = interp1d(
        Y[:real_yield_idx],
        X[:real_yield_idx],
        kind='linear',
        fill_value='extrapolate'
    )

    # Find idealized yield point and initial stiffness
    X_iniIdeal = YXCurve_PreYield(YPeak * 0.1)
    E_iniIdeal = 0.1 * YPeak / X_iniIdeal
    X_at_Peak = X[Y == YPeak][0]
    Y_end = Y[end_idx]

    # Set up optimization for trilinear curve fitting
    initial_guess = (Flags[0], Flags[1])

    # Define parameter bounds for optimization
    bounds = [
        (X_iniIdeal, YPeak / E_iniIdeal),  # XLim1 bounds
        (X_at_Peak * 0.9, 0.95 * X_end)   # XLim2 bounds
    ]

    # Perform optimization to find best trilinear fit
    res = minimize(
        trilinear_curve_obj,
        initial_guess,
        args=(XS, YS),
        bounds=bounds
    )

    # Determine post-end behavior for residual strength
    Y_PostEnd = Y[np.argwhere(Y == Y_end)[0][0]:]
    post_peak_threshold = 0.1 * YPeak

    if sum(Y_PostEnd <= post_peak_threshold) == 0:
        # No significant strength degradation found
        Y_PP = 0.3 * YPeak
        X_PP = X_end
    else:
        # Find point where strength drops to threshold
        threshold_idx = np.where(Y_PostEnd <= post_peak_threshold)[0][0]
        Y_PP = Y_PostEnd[threshold_idx]
        X_PostEnd = X[np.argwhere(Y == Y_end)[0][0]:]
        X_PP = X_PostEnd[threshold_idx]

    # Extract optimized parameters
    X_iniIdeal = res.x[0]
    Y_iniIdeal = X_iniIdeal * E_iniIdeal
    X_at_Peak = min(res.x[1], X_end)

    # Restore original signs
    X *= SignX
    Y *= SignY

    # Ensure non-zero stiffness in post-peak region
    if Y_end == YPeak:
        Y_end *= 0.9

    # Create idealized curve control points
    IdealX = np.array([
        0, X_iniIdeal, X_at_Peak, X_end, X_PP + 1e-3
    ]) * SignX
    IdealY = np.array([
        0, Y_iniIdeal, YPeak, Y_end, Y_PP
    ]) * SignY

    return IdealX, IdealY
