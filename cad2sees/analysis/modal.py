"""
Modal Analysis Module

Provides functionality for performing eigenvalue analysis.

Functions:
    do: Perform modal analysis and extract results
"""

import openseespy.opensees as ops
import json
import numpy as np
import os


def do(NumModes, outsavemode=0, outdir=os.getcwd(), outMPR=False,
       OutModeOuts=False):
    """
    Perform modal analysis to extract natural frequencies and mode shapes.

    Parameters
    ----------
    NumModes : int
        Number of modes to extract from eigenvalue analysis
    outsavemode : int, optional
        0: No file output (default),
        1: Export modal properties to JSON
    outdir : str, optional
        Output directory for modal data files
        (default: current directory)
    outMPR : bool, optional
        Return periods and modes with max mass participation ratios
        (default: False)
    OutModeOuts : bool, optional
        Return periods and complete modal properties
        (default: False)

    Returns
    -------
    list or tuple
        - If outMPR=True: (periods, [MPR_X_mode, MPR_Y_mode])
        - If OutModeOuts=True: (periods, modal_properties_dict)
        - Otherwise: periods list
    """
    # Perform eigenvalue analysis
    # Lambda = ops.eigen('-fullGenLapack', NumModes)  # Alternative solver
    Lambda = ops.eigen(NumModes)
    
    # Calculate natural frequencies and periods
    # Natural frequency: f = sqrt(lambda) / (2*pi)
    freq = [(L**0.5)/(2*3.14159) for L in Lambda]
    Prd = [1/f for f in freq]  # Period = 1/frequency
    # Extract comprehensive modal properties
    ModeOuts = ops.modalProperties('-return')
    # Extract eigenvectors for all nodes and modes
    for ModeI in range(1, NumModes+1):
        NodesAll = ops.getNodeTags()
        ModeOuts[str(ModeI)] = {}
        for NodeCur in NodesAll:
            CurOut = ops.nodeEigenvector(NodeCur, ModeI)
            ModeOuts[str(ModeI)][str(NodeCur)] = CurOut
    # Handle output based on save mode
    if outsavemode == 0:
        pass  # No file output required
    elif outsavemode == 1:
        # Export modal properties to JSON file
        OutFile = os.path.join(outdir, 'ModalProps.json')
        with open(OutFile, 'w') as f:
            json.dump(ModeOuts, f, indent=4)
    # Extract mode shapes for retained nodes (currently unused)
    # This section prepares modal vectors but doesn't save them
    Ns = ops.getRetainedNodes()
    for Mode in range(NumModes):
        ModalVec = []
        for N in Ns:
            ModalVec.append(ops.nodeEigenvector(N, Mode+1))
    # Return results based on requested output format
    if outMPR is True:
        # Return periods and modes with maximum mass participation ratios
        MPR = [np.argmax(ModeOuts['partiMassRatiosMX'])+1,
               np.argmax(ModeOuts['partiMassRatiosMY'])+1]
        return Prd, MPR
    elif OutModeOuts is True:
        # Return periods and complete modal properties
        return Prd, ModeOuts
    else:
        # Return periods only
        return Prd
