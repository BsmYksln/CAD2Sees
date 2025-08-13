"""
Pushover analysis module for CAD2Sees.

This module provides functions for performing static pushover analysis
using different load patterns (modal, uniform, triangular, linear).
"""

import openseespy.opensees as ops
import numpy as np


def _WriteInfo(PatternTag, PushDir):
    """
    Extract and compile pushover information from loaded nodes.
    
    Args:
        PatternTag: OpenSees pattern tag
        PushDir: Push direction (1=X, 2=Y, 3=Z)
        
    Returns:
        list: Push information containing [Z-coordinate, mass, node,
              load factor]
    """
    LoadedNodes = ops.getNodeLoadTags(PatternTag)
    LoadsAll = ops.getNodeLoadData(PatternTag)
    PHIs = []
    Masses = []
    Zs = []
    PushInfo = []
    
    for idx, Node in enumerate(LoadedNodes):
        LoadsCurrent = LoadsAll[(6*idx):6*(idx+1)]
        PHI = LoadsCurrent[PushDir-1]
        PHIs.append(PHI)
        
        cNodes = ops.getConstrainedNodes(Node)
        TotalMass = 0
        for cN in cNodes:
            TotalMass += ops.nodeMass(cN, PushDir)
        Masses.append(TotalMass)
        
        ZCur = ops.nodeCoord(Node)[-1]
        Zs.append(ZCur)
        PushInfo.append([ZCur, TotalMass, Node, PHI])
        
    return PushInfo


def _SinglePush(dref, mu, ctrlNode, dispDir, nSteps, TestTyp='NormDispIncr',
                tolInit=1e-5, iterInit=10, algorithmTyp='KrylovNewton'):
    """
    Perform a single pushover analysis using displacement control.
    
    Args:
        dref: Reference displacement
        mu: Displacement multiplier
        ctrlNode: Control node for displacement control
        dispDir: Direction of displacement
        nSteps: Number of analysis steps
        TestTyp: Convergence test type
        tolInit: Initial tolerance
        iterInit: Initial iterations
        algorithmTyp: Solution algorithm type
        
    Returns:
        list: [displacements, base shear, load factors]
    """
    ops.test(TestTyp, tolInit, iterInit)
    ops.algorithm(algorithmTyp)
    dispTarget = dref * mu
    dU = dispTarget / nSteps
    ops.integrator('DisplacementControl', ctrlNode, dispDir, dU)
    ops.analysis('Static')

    # Initialize variables
    flag = 0
    step = 1
    loadf = 1.0
    lfs = [0]
    disps = [0]
    BaseShear = [0]
    
    while (step <= nSteps) and (flag == 0):
        flag = ops.analyze(1)
        loadf = ops.getTime()
        temp = ops.nodeDisp(ctrlNode, dispDir)
        
        # Try alternative solution strategies if convergence fails
        if flag != 0:
            print('Trying KrylovNewton...')
            ops.test(TestTyp, tolInit * 0.01, iterInit * 10)
            ops.algorithm('KrylovNewton')
            flag = ops.analyze(1)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying with smaller step...')
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU * 0.5)
            ops.test(TestTyp, tolInit * 0.01, iterInit * 10)
            ops.algorithm('KrylovNewton')
            flag = ops.analyze(1)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying with even smaller step...')
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU * 0.25)
            ops.test(TestTyp, tolInit * 0.001, iterInit * 10)
            ops.algorithm('KrylovNewton')
            flag = ops.analyze(1)
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying with even more smaller step...')
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU * 0.1)
            ops.test(TestTyp, tolInit * 0.01, iterInit * 10, 1)
            ops.algorithm('KrylovNewton')
            flag = ops.analyze(1)
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)

        if flag != 0:
            print('Trying NewtonLineSearch...')
            ops.integrator('DisplacementControl', ctrlNode, dispDir, dU * 0.5)
            ops.test(TestTyp, tolInit * 0.01, iterInit * 10)
            ops.algorithm('NewtonLineSearch')
            flag = ops.analyze(1)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying relaxed convergence...')
            ops.test(TestTyp, tolInit * 0.01, iterInit * 10)
            flag = ops.analyze(1)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying Newton with initial then current..')
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm('Newton', *['-initialThenCurrent', True])
            flag = ops.analyze(1)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        if flag != 0:
            print('Trying ModifiedNewton with initial...')
            ops.test(TestTyp, tolInit, iterInit)
            ops.algorithm('ModifiedNewton', *['-initial', True])
            flag = ops.analyze(1)
            ops.algorithm(algorithmTyp)
            ops.test(TestTyp, tolInit, iterInit)
            
        # Update displacement and load factor
        temp = ops.nodeDisp(ctrlNode, dispDir)
        loadf = ops.getTime()
        ops.reactions()
        
        # Calculate base shear
        BS = 0.0
        FixNodes = ops.getFixedNodes()
        RetNodes = ops.getRetainedNodes()
        for nodei in list(set(FixNodes).difference(RetNodes)):
            BS += ops.nodeReaction(int(nodei), dispDir)
        BaseShear.append(BS)
        
        # Check for load reversal
        if len(lfs) > 2 and (np.sign(loadf) != np.sign(lfs[1])):
            flag = 3
            
        lfs.append(loadf)
        disps.append(temp)
        step += 1

    if flag != 0:
        print('DispControl Analysis FAILED')
    else:
        print('DispControl Analysis SUCCESSFUL')
        
    return [disps, BaseShear, lfs]


def _Triangle(NodesData, PushDirection, dref=0.001, mu=400, nSteps=60):
    """
    Perform triangular load pattern pushover analysis.
    
    Args:
        NodesData: Dictionary containing node information
        PushDirection: Direction of push (1=X, 2=Y, 3=Z)
        dref: Reference displacement
        mu: Displacement multiplier
        nSteps: Number of analysis steps
        
    Returns:
        tuple: (SPO curve, SPO info)
    """
    tsTag = 10
    ptTag = 10
    ops.timeSeries('Linear', tsTag, *['-factor', 1.0])
    ops.pattern('Plain', ptTag, tsTag)
    
    LFlist = [0] * 6
    Zs = NodesData['Zs']
    PushFactors = Zs / np.amin(Zs)
    
    for zi, zz in enumerate(Zs):
        PF = PushFactors[zi]
        LFlist[PushDirection-1] = PF
        CurPushNodes = NodesData[str(zz)]
        for nn in CurPushNodes:
            ops.load(nn, *LFlist)
            
    SPOInfo = _WriteInfo(ptTag, PushDirection)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('UmfPack', '-lvalueFact', 30)
    ops.record()
    
    TopNode = NodesData['TopNode']
    SPOCurve = _SinglePush(dref, mu, TopNode, PushDirection, nSteps)
    return SPOCurve, SPOInfo


def _Modal(NodesData, PushDirection, dref=0.001, mu=250, nSteps=60):
    """
    Perform modal load pattern pushover analysis.
    
    Args:
        NodesData: Dictionary containing node information
        PushDirection: Mode number for pushover
        dref: Reference displacement
        mu: Displacement multiplier
        nSteps: Number of analysis steps
        
    Returns:
        tuple: (SPO curve, SPO info)
    """
    from cad2sees.analysis import modal
    
    Prd, ModeOuts = modal.do(3, outsavemode=0, OutModeOuts=True)
    
    # Find push direction based on modal participation
    RatMX = ModeOuts['partiMassRatiosMX'][PushDirection-1]
    RatMY = ModeOuts['partiMassRatiosMY'][PushDirection-1]
    RatMZ = ModeOuts['partiMassRatiosMZ'][PushDirection-1]
    RatRMX = ModeOuts['partiMassRatiosRMX'][PushDirection-1]
    RatRMY = ModeOuts['partiMassRatiosRMY'][PushDirection-1]
    RatRMZ = ModeOuts['partiMassRatiosRMZ'][PushDirection-1]
    Ratios = np.array([RatMX, RatMY, RatMZ, RatRMX, RatRMY, RatRMZ])
    PushDir = int(np.where(Ratios == np.amax(Ratios))[0][0] + 1)

    print(f'Modal pushover for Mode: {PushDirection} is being defined...')
    print(f'Direction {PushDir}')
    
    tsTag = 10
    ptTag = 10
    ops.timeSeries('Linear', tsTag, *['-factor', 1.0])
    ops.pattern('Plain', ptTag, tsTag)
    
    DiaphNodesAll = ops.getRetainedNodes()
    for N in DiaphNodesAll:
        ops.load(N, *list(ops.nodeEigenvector(N, PushDirection)))
        
    SPOInfo = _WriteInfo(ptTag, PushDir)
    ops.constraints('Transformation')
    
    TopNode = NodesData['TopNode']
    SPOCurve = _SinglePush(dref, mu, TopNode, PushDir, nSteps)
    return SPOCurve, SPOInfo


def _Linear(NodesData, PushDirection, dref=0.001, mu=400, nSteps=60):
    """
    Perform linear (uniform) load pattern pushover analysis.
    
    Args:
        NodesData: Dictionary containing node information
        PushDirection: Direction of push (1=X, 2=Y, 3=Z)
        dref: Reference displacement
        mu: Displacement multiplier
        nSteps: Number of analysis steps
        
    Returns:
        tuple: (SPO curve, SPO info)
    """
    tsTag = 10
    ptTag = 10
    ops.timeSeries('Linear', tsTag, *['-factor', 1.0])
    ops.pattern('Plain', ptTag, tsTag)

    LFlist = [0] * 6
    Zs = NodesData['Zs']

    for zz in Zs:
        PF = 1
        LFlist[PushDirection-1] = PF
        CurPushNodes = NodesData[str(zz)]
        for nn in CurPushNodes:
            ops.load(nn, *LFlist)
            
    SPOInfo = _WriteInfo(ptTag, PushDirection)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('UmfPack', '-lvalueFact', 30)

    TopNode = NodesData['TopNode']
    SPOCurve = _SinglePush(dref, mu, TopNode, PushDirection, nSteps)
    return SPOCurve, SPOInfo


def _Uniform(NodesData, PushDirection, dref=0.001, mu=400, nSteps=60):
    """
    Perform uniform load pattern pushover analysis based on storey forces.
    
    Args:
        NodesData: Dictionary containing node information
        PushDirection: Direction of push (1=X, 2=Y, 3=Z)
        dref: Reference displacement
        mu: Displacement multiplier
        nSteps: Number of analysis steps
        
    Returns:
        tuple: (SPO curve, SPO info)
    """
    tsTag = 10
    ptTag = 10
    ops.timeSeries('Linear', tsTag, *['-factor', 1.0])
    ops.pattern('Plain', ptTag, tsTag)

    LFlist = [0] * 6
    Zs = NodesData['Zs']
    PFs = NodesData['StoreyLF']

    for zz, PF in zip(Zs, PFs):
        LFlist[PushDirection-1] = PF
        CurPushNodes = NodesData[str(zz)]
        for nn in CurPushNodes:
            ops.load(nn, *LFlist)
            
    SPOInfo = _WriteInfo(ptTag, PushDirection)
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('UmfPack', '-lvalueFact', 30)

    TopNode = NodesData['TopNode']
    SPOCurve = _SinglePush(dref, mu, TopNode, PushDirection, nSteps)
    return SPOCurve, SPOInfo


def do(Type, Direction, NodesData, dref=0.001, mu=500, nsStep=60):
    """
    Main function to perform pushover analysis.
    
    Args:
        Type: Type of load pattern ('Modal', 'Uniform', 'Triangle', 'Linear')
        Direction: Push direction (positive or negative)
        NodesData: Dictionary containing node information
        dref: Reference displacement
        mu: Displacement multiplier
        nsStep: Number of analysis steps
        
    Returns:
        tuple: (pushover curve, pushover info)
    """
    Sign = Direction / abs(Direction)
    
    if Type == 'Modal':
        Curve, Info = _Modal(NodesData, abs(Direction),
                             dref=dref, mu=mu*Sign, nSteps=nsStep)
    elif Type == 'Uniform':
        Curve, Info = _Uniform(NodesData, abs(Direction),
                               dref=dref, mu=mu*Sign, nSteps=nsStep)
    elif Type == 'Triangle':
        Curve, Info = _Triangle(NodesData, abs(Direction),
                                dref=dref, mu=mu*Sign, nSteps=nsStep)
    elif Type == 'Linear':
        Curve, Info = _Linear(NodesData, abs(Direction),
                              dref=dref, mu=mu*Sign, nSteps=nsStep)
    else:
        raise ValueError(f"Unknown pushover type: {Type}")
        
    return Curve, Info
