"""
Restraint and boundary condition module for CAD2Sees.

Provides functions to apply boundary conditions and restraints to nodes
in OpenSees structural models including fixed supports and constraints.
"""

import openseespy.opensees as ops


def fixall(IDs):
    """
    Apply full fixity (restraint) to specified nodes.

    Applies complete restraint to all degrees of freedom for specified nodes,
    creating fixed supports that prevent all translations and rotations.

    Parameters
    ----------
    IDs : list
        Node identifiers to be fully restrained, converted to integers
        before applying restraint
    """
    for i in IDs:
        ops.fix(int(i),
                *[1, 1, 1, 1, 1, 1])
