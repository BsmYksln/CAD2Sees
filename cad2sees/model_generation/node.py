"""
Node creation module for CAD2Sees.

Provides functions to create nodes in OpenSees models from CAD-extracted
coordinate data with automatic unit conversion.
"""

import openseespy.opensees as ops
from ..helpers import units


def node(IDs, Coordinates):
    """
    Create nodes in the OpenSees model from coordinate data.

    Creates nodes in the current OpenSees model using provided node IDs
    and 3D coordinates with automatic conversion from centimetres to
    the model's unit system.

    Parameters
    ----------
    IDs : list
        Node identifiers (integers or convertible to int), must be unique
    Coordinates : list
        Coordinate sequences [x, y, z] in centimetres, same length as IDs
    """
    for i, cs in zip(IDs, Coordinates):
        ops.node(int(i), cs[0]*units.cm, cs[1]*units.cm, cs[2]*units.cm)
