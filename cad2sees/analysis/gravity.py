"""
Gravity Analysis Module

This module performs gravity analysis.

Functions:
    do: Perform gravity analysis on structural model
"""

import openseespy.opensees as ops


def do(NodeData, IterationNum=50, Tol=1e-5, TestPFlag=0):
    """
    Perform gravity analysis on the structural model.

        Parameters
    ----------
    NodeData : dict
        Node information containing 'ID', 'LoadElevation', 'BoundryConditions'
    IterationNum : int, optional
        Maximum iterations for convergence (default: 50)
    Tol : float, optional
        Convergence tolerance (default: 1e-5)
    TestPFlag : int, optional
        Print flag for convergence test (default: 0)
    """
    # Initialize gravity analysis
    print('\nGravity Analysis...')

    # Set up time series and load pattern
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)

    # Identify nodes with applied loads
    LoadedNodeMap = NodeData['LoadElevation'] != 0
    LoadedNodeIDs = NodeData['ID'][LoadedNodeMap]
    NodeLoads = NodeData['LoadElevation'][LoadedNodeMap]

    # Apply vertical loads to loaded nodes
    for i, L in enumerate(NodeLoads):
        LoadingList = [0.0, 0.0, -L, 0.0, 0.0, 0.0]  # Downward load
        NodeID = LoadedNodeIDs[i]
        # Apply load with node ID prefix convention
        ops.load(int(float(f'1{NodeID}')), *LoadingList)

    # Configure analysis strategy
    ops.constraints('Transformation')     # Constraint handler
    ops.numberer('RCM')                  # Reverse Cuthill-McKee numbering
    ops.system('UmfPack')                # Direct solver
    ops.test('NormDispIncr', Tol, IterationNum, TestPFlag)  # Convergence test
    ops.algorithm('KrylovNewton')        # Solution algorithm

    # Set up incremental loading
    nG = 5                               # Number of load increments
    ops.integrator('LoadControl', 1/nG)  # Load control integrator
    ops.analysis('Static')               # Static analysis type

    # Perform incremental analysis
    ops.analyze(nG)

    # Make loads constant for subsequent analyses
    ops.loadConst('-time', 0.0)

    # Analysis completion and verification
    print('Gravity Analysis Completed!')
    ops.reactions()

    # Calculate total base reaction for verification
    BCMap = NodeData['BoundryConditions'][:, 2] == 1  # Z-restrained nodes
    FixNodes = NodeData['ID'][BCMap]
    BR = 0.0
    for nodei in FixNodes:
        BR += ops.nodeReaction(int(nodei), 3)  # Sum Z-direction reactions

    # Print verification results
    print(f'Calculated Base Reaction: {BR},\n')
    print(f'Loaded Total Force: {NodeData["LoadElevation"].sum()},\n')
