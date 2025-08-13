"""
Capacity data saving utilities for CAD2Sees.

Functions for saving structural element capacity data to JSON files.
"""

import os
import json


def Frame(FrameData, OutDirectory):
    """
    Save frame element capacity data to JSON file.
    
    Args:
        FrameData: Frame capacity data to save
        OutDirectory (str): Output directory path
    """
    if not os.path.exists(OutDirectory):
        os.makedirs(OutDirectory)
    
    OutFile = os.path.join(OutDirectory, 'Frames.json')
    
    with open(OutFile, 'w') as f:
        json.dump(FrameData, f, indent=4)
    
    print(f'Frames Data saved to {OutFile}')


def Infill(InfillData, OutDirectory):
    """
    Save infill panel capacity data to JSON file.
    
    Args:
        InfillData: Infill capacity data to save
        OutDirectory (str): Output directory path
    """
    if not os.path.exists(OutDirectory):
        os.makedirs(OutDirectory)

    OutFile = os.path.join(OutDirectory, 'Infills.json')

    with open(OutFile, 'w') as f:
        json.dump(InfillData, f, indent=4)

    print(f'Infills Data saved to {OutFile}')


def BCJoint(BCJData, OutDirectory):
    """
    Save beam-column joint capacity data to JSON file.
    
    Args:
        BCJData: Beam-column joint capacity data to save
        OutDirectory (str): Output directory path
    """
    if not os.path.exists(OutDirectory):
        os.makedirs(OutDirectory)

    OutFile = os.path.join(OutDirectory, 'BCJoints.json')

    with open(OutFile, 'w') as f:
        json.dump(BCJData, f, indent=4)

    print(f'Beam-Column Joints Data saved to {OutFile}')
