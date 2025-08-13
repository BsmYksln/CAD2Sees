"""
Unit system definitions for CAD2Sees.

This module defines a consistent unit system based on kN-m-sec for
structural analysis calculations. All units are defined relative to
the base units: kilonewton (kN), metre (m), and second (sec).

Base Units:
    Force: kN (kilonewton)
    Length: m (metre)
    Time: sec (second)

Derived Units:
    Stress: MPa (megapascal)
    Acceleration: m/sec²
"""

import math

# Mathematical constants
pi = math.pi

# Force units (base: kN)
kN = 1.0
MN = kN * 1000
N = kN / 1000

# Length units (base: m)
m = 1.0
cm = m / 100  # centimetre
mm = m / 1000  # millimetre

# Time units (base: sec)
sec = 1.0

# Derived units
MPa = N / (mm**2)  # Stress unit: megapascal
g = 9.807 * m / (sec**2)  # Gravitational acceleration
