"""
CAD2Sees module dedicated to modelling structural constraints.

Provides functionality for creating structural constraints, particularly
rigid diaphragm constraints for multi-storey building analysis with
different modelling approaches for floor diaphragms.
"""

import openseespy.opensees as ops
import numpy as np
from cad2sees.helpers import units


def rigid_diaphragm(NodeData, RDFLag):
    """
    Create rigid diaphragm constraints for multi-storey building analysis.

    Implements rigid diaphragm constraints enforcing in-plane rigidity of
    floor slabs. Nodes at each floor level move together in horizontal
    translations and vertical rotation.

    Parameters
    ----------
    NodeData : dict
        Node information including ID, Type, Coordinates, BoundryConditions,
        Mass arrays
    RDFLag : int
        Rigid diaphragm flag:
        0: No constraints
        1: Use existing node as master node
        2: Create new master nodes at mass-weighted centres

    Returns
    -------
    dict
        Diaphragm constraint information including floor elevations,
        master/slave node relationships, and restraint nodes
    """
    # Create boundary condition and BCJ node mapping arrays
    BCMap = NodeData['BoundryConditions'][:, 2] == 1  # Z-restrained nodes
    BCJMap = np.array(NodeData['Type']) == 'BCJ'      # Beam-column joints

    # Get sorted unique floor elevations (excluding restrained nodes)
    Zs = np.sort(np.unique(NodeData['Coordinates'][~BCMap, 2])) * units.cm

    # Initialize diaphragm information dictionary
    PushNodes = {'Zs': Zs, 'StoreyLF': []}
    for Z in Zs:
        Z_mask = NodeData['Coordinates'][:, 2]*units.cm == Z
        PushNodes['StoreyLF'].append(NodeData['LF'][Z_mask].sum())
    RestrainNodes = []

    # Method 1: Use existing BCJ nodes as master nodes
    if RDFLag == 1:
        for Z in Zs:
            # Find all nodes at current floor elevation
            ZsAll = NodeData['Coordinates'][:, 2] * units.cm
            ZMap = np.array(ZsAll) == Z

            # Identify column top (BCJ) nodes at this floor
            ColumnTopMap = ZMap & (BCJMap)

            # Find non-boundary, non-BCJ nodes at this floor
            NoBCNoBCJMap = (~BCMap) & (~BCJMap) & ZMap
            NoBCNoBCJNodeID = NodeData['ID'][NoBCNoBCJMap]

            # Select master node from BCJ candidates
            MasterCandidates = NodeData['ID'][ColumnTopMap]
            if len(MasterCandidates) % 2 != 0:
                rNodeIDX = len(MasterCandidates) // 2      # Middle for odd
            else:
                rNodeIDX = len(MasterCandidates) // 2 - 1  # Middle-1 for even

            # Create master node ID with prefix '1'
            rNode = int(float(f'1{MasterCandidates[rNodeIDX]}'))

            # Create slave node lists (other BCJ + non-BCJ nodes)
            cNodesDummy = np.delete(MasterCandidates, rNodeIDX)
            cNodes = ([int(float(f'1{c}')) for c in cNodesDummy] +
                      [int(i) for i in NoBCNoBCJNodeID])

            # Store constraint information and apply OpenSees constraint
            PushNodes[str(Z)] = [rNode] + cNodes
            ops.rigidDiaphragm(3, rNode, *cNodes)
            RestrainNodes.append(rNode)

        PushNodes['TopNode'] = rNode

    # Method 2: Create new master nodes at mass-weighted centers
    elif RDFLag == 2:
        for Zidx, Z in enumerate(Zs):
            # Find all nodes at current floor elevation
            ZsAll = NodeData['Coordinates'][:, 2] * units.cm
            ZMap = np.array(ZsAll) == Z

            # Create new master node with sequential numbering
            rNode = int(11111 * (Zidx + 1))
            PushNodes[str(Z)] = [rNode]

            # Calculate mass-weighted center of floor
            CurXs = NodeData['Coordinates'][ZMap, 0] * units.cm
            CurYs = NodeData['Coordinates'][ZMap, 1] * units.cm
            WCur = NodeData['Mass'][ZMap]
            XCentre = np.average(CurXs, weights=WCur)
            YCentre = np.average(CurYs, weights=WCur)

            # Create master node at center and fix appropriately
            ops.node(rNode, XCentre, YCentre, Z)
            ops.fix(rNode, 0, 0, 1, 1, 1, 0)  # Free in X, Y, RZ

            # Find BCJ nodes at this floor for constraining
            ColumnTopMap = ZMap & (BCJMap)
            cNodesDummy = NodeData['ID'][ColumnTopMap]

            # Find additional non-boundary nodes to constrain
            NoBCNoBCJMap = (~BCMap) & (~BCJMap) & ZMap
            NoBCNoBCJNodeID = NodeData['ID'][NoBCNoBCJMap]

            # Create slave node list with prefix '1'
            cNodes = [int(float(f'1{c}')) for c in cNodesDummy]

            # Apply rigid diaphragm constraint
            ops.rigidDiaphragm(3, rNode, *cNodes)
            RestrainNodes.append(rNode)

        PushNodes['TopNode'] = rNode

    # No rigid diaphragm case
    else:
        print("No Rigid Diaphragm Modelled!")

    # Store restraint nodes and return constraint information
    PushNodes['RestrainNodes'] = RestrainNodes
    return PushNodes
