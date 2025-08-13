"""
Model initialisation module for CAD2Sees.

Provides functions to initialise OpenSees models with specific configurations
for structural analysis including model dimensions and degrees of freedom.
"""

import openseespy.opensees as ops


def do3D():
    """
    Initialise a 3D OpenSees model with 6 degrees of freedom per node.

    Sets up a basic 3D structural analysis model using OpenSees with
    6 DOF per node and defines a rigid elastic material for constraints.

    Returns
    -------
    tuple
        Number of dimensions (3) and degrees of freedom per node (6)
    """
    ndm = 3
    ndf = 6
    ops.wipe()
    ops.model('basic', '-ndm', ndm, '-ndf', ndf)
    # Create Rigid Material
    ops.uniaxialMaterial('Elastic', 999999, 1e12)
    return ndm, ndf
