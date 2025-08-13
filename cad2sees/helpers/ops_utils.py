"""
OpenSees utilities for data recording and output management within CAD2Sees.

Functions for setting up recorders for structural elements, nodes, and
response quantities in OpenSees analysis models.
"""

import openseespy.opensees as ops
import numpy as np
import json
import os


def AddBCJRecorder(OutDir, ElementList):
    """
    Set up recorders for beam-column joint (BCJ) elements.

    Args:
        OutDir (str): Output directory path
        ElementList (list): List of BCJ element IDs to record

    Output Files:
        OutBCJ.out: Element connectivity matrix
        BCJForce.out: Element force time history
        BCJDeformation.out: Element deformation time history
    """
    BCJListFile = os.path.join(OutDir, 'OutBCJ.out')
    BCJForceFile = os.path.join(OutDir, 'BCJForce.out')
    BCJDeformationFile = os.path.join(OutDir, 'BCJDeformation.out')
    INodes = []
    JNodes = []
    for E in ElementList:
        Nodes = ops.eleNodes(E)
        INodes.append(Nodes[0])
        JNodes.append(Nodes[1])
    np.savetxt(BCJListFile, np.array([ElementList, INodes, JNodes]))
    ops.recorder('Element', '-file', str(BCJForceFile),
                 '-precision', 6, '-ele', *ElementList, 'force')
    ops.recorder('Element', '-file', str(BCJDeformationFile),
                 '-precision', 6, '-ele', *ElementList, 'deformation')


def AddFrameForceRecorders(OutDir, ElementList):
    """
    Configure force recorders for frame elements.

    Args:
        OutDir (str): Output directory path
        ElementList (list): List of frame element IDs to record

    Output Files:
        OutFrames.out: Element connectivity matrix
        FrameLocalForce.out: Local force time history
        FrameGlobalForce.out: Global force time history
    """
    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameLocalForceFile = os.path.join(OutDir, 'FrameLocalForce.out')
    FrameGlobalForceFile = os.path.join(OutDir, 'FrameGlobalForce.out')
    INodes = []
    JNodes = []
    for E in ElementList:
        Nodes = ops.eleNodes(E)
        INodes.append(Nodes[0])
        JNodes.append(Nodes[1])
    np.savetxt(FrameListFile, np.array([ElementList, INodes, JNodes]))
    ops.recorder('Element', '-file', str(FrameLocalForceFile),
                 '-precision', 6, '-ele', *ElementList, 'localForces')
    ops.recorder('Element', '-file', str(FrameGlobalForceFile),
                 '-precision', 6, '-ele', *ElementList, 'globalForces')


def AddFrameInflectionPointRecorders(OutDir, ElementList):
    """
    Set up recorders for frame element inflection point data.

    Args:
        OutDir (str): Output directory path
        ElementList (list): List of frame element IDs to record

    Output Files:
        InflectionPoint.out: Inflection point location time history
    """
    InflectionPointFile = os.path.join(OutDir, 'InflectionPoint.out')
    ops.recorder('Element', '-file', str(InflectionPointFile),
                 '-precision', 6, '-ele', *ElementList, 'inflectionPoint')


def AddInfillRecorder(OutDir, ElementList):
    """
    Set up recorders for infill panel elements.

    Args:
        OutDir (str): Output directory path
        ElementList (list): List of infill element IDs to record

    Output Files:
        OutInfills.out: List of infill element IDs
        InfillForce.out: Basic force time history
        InfillDeformation.out: Deformation time history
    """
    InfillListFile = os.path.join(OutDir, 'OutInfills.out')
    InfillForceFile = os.path.join(OutDir, 'InfillForce.out')
    InfillDeformationFile = os.path.join(OutDir, 'InfillDeformation.out')
    np.savetxt(InfillListFile, np.array(ElementList))
    ops.recorder('Element', '-file', str(InfillForceFile),
                 '-precision', 6, '-ele', *ElementList, 'basicForces')
    ops.recorder('Element', '-file', str(InfillDeformationFile),
                 '-precision', 6, '-ele', *ElementList, 'deformation')


def AddChordRotationRecorders(OutDir, ElementList):
    """
    Configure chord and plastic rotation recorders for frame elements.

    Args:
        OutDir (str): Output directory path
        ElementList (list): List of frame element IDs to record

    Output Files:
        OutFrames.out: Element connectivity matrix
        ChordRotation.out: Total chord rotation time history
        PlasticChordRotation.out: Plastic rotation time history
    """
    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameRotationFile = os.path.join(OutDir, 'ChordRotation.out')
    FramePlasticRotationFile = os.path.join(OutDir, 'PlasticChordRotation.out')
    INodes = []
    JNodes = []
    for E in ElementList:
        Nodes = ops.eleNodes(E)
        INodes.append(Nodes[0])
        JNodes.append(Nodes[1])
    np.savetxt(FrameListFile, np.array([ElementList, INodes, JNodes]))
    ops.recorder('Element', '-file', str(FrameRotationFile),
                 '-precision', 6, '-ele', *ElementList, 'chordRotation')
    ops.recorder('Element', '-file', str(FramePlasticRotationFile),
                 '-precision', 6, '-ele', *ElementList, 'plasticRotation')


def AddDisplacementRecorders(OutDir, NodeList=[]):
    """
    Configure nodal displacement recorders for all degrees of freedom.

    Args:
        OutDir (str): Output directory path
        NodeList (list, optional): List of node IDs to record.
                                  If empty, records all nodes.

    Output Files:
        OutNodes.out: List of recorded node IDs
        TransversalX.out: X-direction displacement (DOF 1)
        TransversalY.out: Y-direction displacement (DOF 2)
        TransversalZ.out: Z-direction displacement (DOF 3)
        RotationalX.out: X-axis rotation (DOF 4)
        RotationalY.out: Y-axis rotation (DOF 5)
        RotationalZ.out: Z-axis rotation (DOF 6)
    """
    NodeListFile = os.path.join(OutDir, 'OutNodes.out')
    NodeTrans1File = os.path.join(OutDir, 'TransversalX.out')
    NodeTrans2File = os.path.join(OutDir, 'TransversalY.out')
    NodeTrans3File = os.path.join(OutDir, 'TransversalZ.out')
    NodeRot1File = os.path.join(OutDir, 'RotationalX.out')
    NodeRot2File = os.path.join(OutDir, 'RotationalY.out')
    NodeRot3File = os.path.join(OutDir, 'RotationalZ.out')
    if len(NodeList) == 0:
        NodeList = list(ops.getNodeTags())
    np.savetxt(NodeListFile, np.array(NodeList))
    ops.recorder('Node', '-file', str(NodeTrans1File), '-precision', 6,
                 '-node', *NodeList, '-dof', 1, 'disp')

    ops.recorder('Node', '-file', str(NodeTrans2File), '-precision', 6,
                 '-node', *NodeList, '-dof', 2, 'disp')

    ops.recorder('Node', '-file', str(NodeTrans3File), '-precision', 6,
                 '-node', *NodeList, '-dof', 3, 'disp')

    ops.recorder('Node', '-file', str(NodeRot1File), '-precision', 6,
                 '-node', *NodeList, '-dof', 4, 'disp')

    ops.recorder('Node', '-file', str(NodeRot2File), '-precision', 6,
                 '-node', *NodeList, '-dof', 5, 'disp')

    ops.recorder('Node', '-file', str(NodeRot3File), '-precision', 6,
                 '-node', *NodeList, '-dof', 6, 'disp')


def GetShape(OutDir):
    """
    Export structural geometry data for visualization.

    Args:
        OutDir (str): Output directory path

    Returns:
        dict: Element connectivity information

    Output Files:
        ShapeData.json: Complete structural geometry in JSON format
    """
    OutFile = os.path.join(OutDir, 'ShapeData.json')
    ElementsAll = ops.getEleTags()
    Elements = {}

    for CurElement in ElementsAll:
        CurNodes = ops.eleNodes(CurElement)
        NodeICoord = ops.nodeCoord(CurNodes[0])
        NodeJCoord = ops.nodeCoord(CurNodes[1])
        Elements[str(CurElement)] = {'i': {'Tag': CurNodes[0],
                                           'Coordinates': NodeICoord},
                                     'j': {'Tag': CurNodes[1],
                                           'Coordinates': NodeJCoord}}
    NodeTags = ops.getNodeTags()
    NodeCoords = []
    for CurNode in NodeTags:
        NodeCoords.append(ops.nodeCoord(CurNode))
    Nodes = {'Tags': NodeTags,
             'Coordinates': NodeCoords}
    Data = {'Elements': Elements,
            'Nodes': Nodes}
    with open(OutFile, 'w') as f:
        json.dump(Data, f, indent=4)
    return Elements


def AddAccelerationRecorders(OutDir, NodeList=[]):
    """
    Configure nodal acceleration recorders for dynamic analysis.

    Args:
        OutDir (str): Output directory path
        NodeList (list, optional): List of node IDs to record.
                                  If empty, records all nodes.

    Output Files:
        OutNodes.out: List of recorded node IDs
        AccelerationX.out: X-direction acceleration (DOF 1)
        AccelerationY.out: Y-direction acceleration (DOF 2)
        AccelerationZ.out: Z-direction acceleration (DOF 3)
    """
    NodeListFile = os.path.join(OutDir, 'OutNodes.out')
    NodeTrans1File = os.path.join(OutDir, 'AccelerationX.out')
    NodeTrans2File = os.path.join(OutDir, 'AccelerationY.out')
    NodeTrans3File = os.path.join(OutDir, 'AccelerationZ.out')
    if len(NodeList) == 0:
        NodeList = list(ops.getNodeTags())
    np.savetxt(NodeListFile, np.array(NodeList))
    ops.recorder('Node', '-file', str(NodeTrans1File), '-precision', 6,
                 '-time', '-node', *NodeList, '-dof', 1, 'accel')

    ops.recorder('Node', '-file', str(NodeTrans2File), '-precision', 6,
                 '-time', '-node', *NodeList, '-dof', 2, 'accel')

    ops.recorder('Node', '-file', str(NodeTrans3File), '-precision', 6,
                 '-time', '-node', *NodeList, '-dof', 3, 'accel')


def AddVelocityRecorders(OutDir, NodeList=[]):
    """
    Configure nodal velocity recorders for dynamic analysis.

    Args:
        OutDir (str): Output directory path
        NodeList (list, optional): List of node IDs to record.
                                  If empty, records all nodes.

    Output Files:
        OutNodes.out: List of recorded node IDs
        VelocityX.out: X-direction velocity (DOF 1)
        VelocityY.out: Y-direction velocity (DOF 2)
        VelocityZ.out: Z-direction velocity (DOF 3)
    """
    NodeListFile = os.path.join(OutDir, 'OutNodes.out')
    NodeTrans1File = os.path.join(OutDir, 'VelocityX.out')
    NodeTrans2File = os.path.join(OutDir, 'VelocityY.out')
    NodeTrans3File = os.path.join(OutDir, 'VelocityZ.out')
    if len(NodeList) == 0:
        NodeList = list(ops.getNodeTags())
    np.savetxt(NodeListFile, np.array(NodeList))
    ops.recorder('Node', '-file', str(NodeTrans1File), '-precision', 6,
                 '-node', *NodeList, '-dof', 1, 'vel')

    ops.recorder('Node', '-file', str(NodeTrans2File), '-precision', 6,
                 '-node', *NodeList, '-dof', 2, 'vel')

    ops.recorder('Node', '-file', str(NodeTrans3File), '-precision', 6,
                 '-node', *NodeList, '-dof', 3, 'vel')
