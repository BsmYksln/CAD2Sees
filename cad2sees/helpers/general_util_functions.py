"""
General utility functions for CAD2Sees.

This module provides general-purpose utility functions including numerical
methods, function manipulation, and data processing utilities.
"""


def find_zero(f, vars, a, b, tolerance=1e-6, max_iterations=100):
    """
    Find the zero of a function using the bisection method.
    
    Args:
        f: Function to find the zero of
        vars: Additional variables to pass to the function
        a: Lower bound of the search interval
        b: Upper bound of the search interval
        tolerance: Convergence tolerance (default: 1e-6)
        max_iterations: Maximum number of iterations (default: 100)
        
    Returns:
        float: The zero of the function within the specified tolerance
        
    Raises:
        ValueError: If function values at endpoints have same sign or
                   if method doesn't converge
    """
    avars = [a] + vars
    bvars = [b] + vars
    i = 0
    
    # Adjust bounds if function values have same sign
    while (f(*avars) * f(*bvars) > 0) and i < 1000:
        a += 1e-3
        avars = [a] + vars
        i += 1
        
    if f(*avars) * f(*bvars) > 0:
        raise ValueError("The function values at the given "
                         "interval endpoints must have opposite signs")
    
    # Bisection method
    for _ in range(max_iterations):
        Dm = (a + b) / 2
        Dmvars = [Dm] + vars
        value = f(*Dmvars)
        
        if abs(value) < tolerance:
            return Dm
            
        if f(*avars) * value < 0:
            b = Dm
        else:
            a = Dm
            
    raise ValueError("Bisection method did not converge")


def funMod(fun, fixVals, varIdx, X):
    """
    Modify a function by fixing certain variables and varying one parameter.
    
    Args:
        fun: The function to modify
        fixVals: List of fixed values for function parameters
        varIdx: Index of the variable parameter to modify
        X: New value for the variable parameter
        
    Returns:
        Result of the function call with modified parameters
    """
    values = list(fixVals)
    values[varIdx] = X
    return fun(*values)


def indices_of_unique_values(lst):
    """
    Find indices of unique values in a list.
    
    Args:
        lst: Input list to process
        
    Returns:
        list: Indices of elements that appear exactly once in the list
    """
    unique_indices = []
    seen_values = set()
    
    for i, value in enumerate(lst):
        if value not in seen_values:
            unique_indices.append(i)
            seen_values.add(value)
        elif value in seen_values and i in unique_indices:
            unique_indices.remove(i)
            
    return unique_indices
