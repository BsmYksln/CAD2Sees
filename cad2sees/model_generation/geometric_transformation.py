"""
Geometric transformation module for CAD2Sees.

Provides functions to create geometric transformations for structural
frame elements in OpenSees models, including joint offsets for rigid
zone modelling.
"""

import numpy as np
from cad2sees.helpers import units
from cad2sees.helpers.general_util_functions import indices_of_unique_values
import openseespy.opensees as ops


def do(FramesData, SectionsData, RLFlag=1, ndm=3):
    """
    Create geometric transformations for frame elements with joint offsets.

    Generates geometric transformations for structural frame elements based
    on element orientation and creates joint offsets to model rigid zones
    at beam-column connections.

    Parameters
    ----------
    FramesData : dict
        Frame element data including Direction, section types, IDs,
        and node connections
    SectionsData : dict
        Section properties with section IDs as keys including width
        and height dimensions
    RLFlag : int, optional
        Rigid link flag (1: include joint offsets, 2: no offsets), default 1
    ndm : int, optional
        Number of model dimensions, default 3

    Returns
    -------
    tuple
        Updated frame data with GTID key and geometric transformations
        dictionary with ID, TAG, and Joint_Offset arrays
    """
    IDs = []
    JntOff = []
    GTTAGs = []
    GTIDs = []
    # Find Columns
    ColumnMap = (FramesData['Direction'] == 3)
    ColumnSecI = list(set(np.array(FramesData['SectionTypI'])[ColumnMap]))
    ColumnSecJ = list(set(np.array(FramesData['SectionTypJ'])[ColumnMap]))
    ColumnSections = list(set(ColumnSecI + ColumnSecJ))
    bcs = [SectionsData[key]['b'] for key in ColumnSections]
    hcs = [SectionsData[key]['h'] for key in ColumnSections]
    dims = list(set(bcs + hcs + [0]))
    for ci in dims:
        for cj in dims:
            GeoTransName = f'BeamX_{int(ci)}_{int(cj)}'
            IDs.append(GeoTransName)
            if RLFlag == 1:
                JntOffX = [0.5*ci*units.mm, 0, 0, -0.5*cj*units.mm, 0, 0]
            elif RLFlag == 2:
                JntOffX = 2*ndm*[0]
            JntOff.append(JntOffX)
            GeoTransName = f'BeamY_{int(ci)}_{int(cj)}'
            IDs.append(GeoTransName)
            if RLFlag == 1:
                JntOffY = [0, 0.5*ci*units.mm, 0, 0, -0.5*cj*units.mm, 0]
            elif RLFlag == 2:
                JntOffY = 2*ndm*[0]
            JntOff.append(JntOffY)
    # Find Beams
    BeamMap = ~ColumnMap
    BeamSecI = list(set(np.array(FramesData['SectionTypI'])[BeamMap]))
    BeamSecJ = list(set(np.array(FramesData['SectionTypJ'])[BeamMap]))
    BeamSections = list(set(BeamSecI + BeamSecJ))
    hbs = list(set([SectionsData[key]['h'] for key in BeamSections] + [0]))
    for ci in hbs:
        for cj in hbs:
            GeoTransName = f'Column_{int(ci)}_{int(cj)}'
            IDs.append(GeoTransName)
            if RLFlag == 1:
                # JntOffC = [0.5*ci*units.mm, 0, 0, -0.5*cj*units.mm, 0, 0]
                JntOffC = [0, 0, -0.5*ci*units.mm, 0, 0, 0.5*cj*units.mm]
            elif RLFlag == 2:
                JntOffC = 2*ndm*[0]
            JntOff.append(JntOffC)
    idxs = indices_of_unique_values(IDs)
    IDs = np.array(IDs)[idxs]
    JntOffs = np.array(JntOff)[idxs]
    for idx, ID in enumerate(IDs):
        # JntOff = [0.0] * 6
        JntOff = JntOffs[idx]
        GTTAG = idx + 101
        GTTAGs.append(GTTAG)
        if 'Column' in ID:
            # Column transformation | Z in -X
            ops.geomTransf('PDelta', GTTAG,
                           -1, 0, 0,
                           '-jntOffset', *JntOff)
        elif 'BeamX' in ID:
            # Beam X-X direct transformation | Z in +Y
            ops.geomTransf('Linear', GTTAG,
                           0, 1, 0,
                           '-jntOffset', *JntOff)
        elif 'BeamY' in ID:
            # Beam Y-Y drct transformation | Z in -X
            ops.geomTransf('Linear', GTTAG,
                           -1, 0, 0,
                           '-jntOffset', *JntOff)
    # Map ID to Frames
    for i, ID in enumerate(FramesData['ID']):
        i_ID = FramesData['i_ID'][i]
        j_ID = FramesData['j_ID'][i]
        FrameDir = FramesData['Direction'][i]
        ConnectedIMap = ((FramesData['i_ID'] == i_ID) |
                         (FramesData['j_ID'] == i_ID))
        ConnectedJMap = ((FramesData['i_ID'] == j_ID) |
                         (FramesData['j_ID'] == j_ID))
        if FrameDir == 3:
            ConnectedIMapOD = ConnectedIMap & (FramesData['Direction'] != 3)
            if sum(ConnectedIMapOD) == 0:
                ci = 0
            else:
                # No changing section
                CurrentSections = np.array(FramesData['Type'])[ConnectedIMapOD]
                hs = [SectionsData[k]['h'] for k in CurrentSections]
                ci = max(hs)

            ConnectedJMapOD = ConnectedJMap & (FramesData['Direction'] != 3)
            if sum(ConnectedJMapOD) == 0:
                cj = 0
            else:
                # No changing section
                CurrentSections = np.array(FramesData['Type'])[ConnectedJMapOD]
                hs = [SectionsData[k]['h'] for k in CurrentSections]
                cj = max(hs)
            GTID = f'Column_{int(ci)}_{int(cj)}'

        if FrameDir != 3:
            ConnectedIMapOD = ConnectedIMap & (FramesData['Direction'] == 3)
            if sum(ConnectedIMapOD) == 0:
                ci = 0
            else:
                CurrentSections = np.array(FramesData['Type'])[ConnectedIMapOD]
                if FrameDir == 1:
                    bs = [SectionsData[k]['b'] for k in CurrentSections]
                    ci = max(bs)
                elif FrameDir == 2:
                    hs = [SectionsData[k]['h'] for k in CurrentSections]
                    ci = max(hs)

            ConnectedJMapOD = ConnectedJMap & (FramesData['Direction'] == 3)
            if sum(ConnectedJMapOD) == 0:
                cj = 0
            else:
                CurrentSections = np.array(FramesData['Type'])[ConnectedJMapOD]
                if FrameDir == 1:
                    bs = [SectionsData[k]['b'] for k in CurrentSections]
                    cj = max(bs)
                elif FrameDir == 2:
                    hs = [SectionsData[k]['h'] for k in CurrentSections]
                    cj = max(hs)
            if FrameDir == 1:
                GTID = f'BeamX_{int(ci)}_{int(cj)}'
            elif FrameDir == 2:
                GTID = f'BeamY_{int(ci)}_{int(cj)}'
        GTIDs.append(GTID)
    FramesData['GTID'] = GTIDs
    GeoTransformations = {'ID': np.array(IDs),
                          'TAG': np.array(GTTAGs),
                          'Joint_Offset': np.array(JntOff)}

    return FramesData, GeoTransformations
