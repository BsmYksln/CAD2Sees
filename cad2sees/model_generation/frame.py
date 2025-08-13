"""
CAD2Sees module for modelling frame elements.

Provides functionality for creating and configuring frame elements in
structural analysis models with various analysis types including
fibre-based analysis using custom implementations and OpenSees integration.
"""

import numpy as np
import copy
from .component_modeller import frame_modeller as fm


def create(ModelType,
           FrameData,
           SectionData,
           PointData,
           GeometricTransforms):
    """
    Create frame elements for structural analysis models.

    Creates frame elements based on specified model type and input data,
    processing frame connectivity, section properties, nodal data, and
    geometric transformations.

    Parameters
    ----------
    ModelType : str
        Analysis model type: 'BWH_Fiber', 'BWH_FiberOPS', 'BWH_FiberOPS2',
        or 'BWH_Simple'
    FrameData : dict
        Frame element data including ID, Direction, section types,
        node IDs, and geometric transformation IDs
    SectionData : dict
        Section properties indexed by section ID, modified in-place
        to include moment-curvature outputs
    PointData : dict
        Nodal data including ID, Type, Coordinates, boundary conditions,
        and loads
    GeometricTransforms : dict
        Geometric transformation data with ID and TAG arrays

    Returns
    -------
    tuple[dict, dict]
        Frame element outputs and updated section data with
        moment-curvature results
    """
    FrameOut = {}

    # Process each frame element
    for i, ID in enumerate(FrameData['ID']):
        # Extract section information for I and J ends
        SectionIid = FrameData['SectionTypI'][i]
        SectionI = copy.deepcopy(SectionData[SectionIid])
        SectionJid = FrameData['SectionTypJ'][i]
        SectionJ = copy.deepcopy(SectionData[SectionJid])

        # Extract node information
        NodeI = FrameData['i_ID'][i]
        NIMap = np.where(PointData['ID'] == NodeI)
        NodeJ = FrameData['j_ID'][i]
        NJMap = np.where(PointData['ID'] == NodeJ)

        # Extract geometric transformation
        GTMap = np.where(GeometricTransforms['ID'] == FrameData['GTID'][i])
        GTTAG = GeometricTransforms['TAG'][GTMap]

        # Determine axial loads based on frame direction
        if FrameData['Direction'][i] == 3:
            INLoad = PointData['Load'][NIMap]
            JNLoad = PointData['Load'][NJMap]
        else:
            INLoad = 0.0
            JNLoad = 0.0

        # Prepare node data structures
        NodeDataI = {
            'ID': NodeI,
            'Type': np.array(PointData['Type'])[NIMap][0],
            'Coordinates': PointData['Coordinates'][NIMap][0],
            'u1': PointData['BoundryConditions'][NIMap][0][0],
            'NLoad': INLoad
        }
        NodeDataJ = {
            'ID': NodeJ,
            'Type': np.array(PointData['Type'])[NJMap][0],
            'Coordinates': PointData['Coordinates'][NJMap][0],
            'u1': PointData['BoundryConditions'][NJMap][0][0],
            'NLoad': JNLoad
        }

        # Create frame modeller based on specified model type
        frame_args = (ID, FrameData['Direction'][i], GTTAG,
                      SectionI, SectionJ, NodeDataI, NodeDataJ)

        if ModelType == 'BWH_Fiber':
            FCur = fm.BWH_Frame('Fiber', *frame_args)
        elif ModelType == 'BWH_FiberOPS':
            FCur = fm.BWH_Frame('FiberOPS', *frame_args)
        elif ModelType == 'BWH_FiberOPS2':
            FCur = fm.BWH_Frame('FiberOPS2', *frame_args)
        elif ModelType == 'BWH_Simple':
            FCur = fm.BWH_Frame('Simple', *frame_args)
        else:
            raise ValueError(f"Unknown ModelType: {ModelType}")

        # Build frame capacity and get moment-curvature outputs
        CurrentOut, MCOutI, MCOutJ = FCur.buildWcapacity()

        # Store moment-curvature results in section data
        # Avoid redundant calculations for identical sections with same loads
        if SectionIid == SectionJid and INLoad == JNLoad:
            # Same section at both ends with same axial load
            if 'MCOut' not in SectionData[SectionIid]:
                SectionData[SectionIid]['MCOut'] = {}
            SectionData[SectionIid]['MCOut'][str(INLoad)] = {
                **MCOutI, **MCOutJ
            }
        else:
            # Different sections or different axial loads
            if 'MCOut' not in SectionData[SectionIid]:
                SectionData[SectionIid]['MCOut'] = {}
            if 'MCOut' not in SectionData[SectionJid]:
                SectionData[SectionJid]['MCOut'] = {}
            SectionData[SectionIid]['MCOut'][str(INLoad)] = MCOutI
            SectionData[SectionJid]['MCOut'][str(JNLoad)] = MCOutJ

        # Store frame element output
        FrameOut[str(ID)] = CurrentOut

    return FrameOut, SectionData
