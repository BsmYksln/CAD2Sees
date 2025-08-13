"""
Frame failure check utilities.

Functions for extracting frame failure data from analysis output files.
"""
import os
import numpy as np


def getFrameFlexural(OutDir, step):
    """
    Get frame flexural failure data from analysis output.

    Parameters
    ----------
    OutDir : str
        Output directory containing analysis results.
    step : int
        Analysis step number to extract.

    Returns
    -------
    tuple
        Frame collapse and yield DCR data dictionaries.
    """
    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameList = np.loadtxt(FrameListFile)[0, :]

    FFDir = os.path.join(OutDir, 'FrameRotationDCR')
    
    FrameCollapse = {}
    FrameYield = {}

    for suffix in ['Y_I', 'Y_J', 'Z_I', 'Z_J']:
        FCollapseFile = os.path.join(FFDir, f'DCR_{suffix}.csv')
        FYieldFile = os.path.join(FFDir, f'DCR_Yeild_{suffix}.csv')
        DataCollapse = np.loadtxt(FCollapseFile)
        DataYield = np.loadtxt(FYieldFile)

        DataCollapse = np.maximum.accumulate(DataCollapse[:step, :], axis=0)
        DataYield = np.maximum.accumulate(DataYield[:step, :], axis=0)

        FrameCollapse[f'{suffix}'] = DataCollapse[-1]
        FrameYield[f'{suffix}'] = DataYield[-1]

    FrameCollapse = {str(int(FrameList[i])):
                     {'I': {'Y': FrameCollapse['Y_I'][i],
                            'Z': FrameCollapse['Z_I'][i]},
                      'J': {'Y': FrameCollapse['Y_J'][i],
                            'Z': FrameCollapse['Z_J'][i]}}
                     for i in range(len(FrameList))}
    FrameYield = {str(int(FrameList[i])):
                  {'I': {'Y': FrameYield['Y_I'][i],
                         'Z': FrameYield['Z_I'][i]},
                   'J': {'Y': FrameYield['Y_J'][i],
                         'Z': FrameYield['Z_J'][i]}}
                  for i in range(len(FrameList))}

    return FrameCollapse, FrameYield


def getFrameShear(OutDir, step):
    """
    Get frame shear failure data from analysis output.

    Parameters
    ----------
    OutDir : str
        Output directory containing analysis results.
    step : int
        Analysis step number to extract.

    Returns
    -------
    dict
        Frame shear failure DCR data dictionary.
    """
    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameList = np.loadtxt(FrameListFile)[0, :]

    FFDir = os.path.join(OutDir, 'FrameShearDCR')

    FFail = {}

    for suffix in ['Y_I', 'Y_J', 'Z_I', 'Z_J']:
        filename = [F for F in os.listdir(FFDir) if f'_{suffix}.csv' in F][0]
        file_path = os.path.join(FFDir, filename)
        data = np.loadtxt(file_path)
        data = np.maximum.accumulate(data[:step, :], axis=0)
        FFail[f'{suffix}'] = data[-1]

    FrameFail = {str(int(FrameList[i])):
                 {'I': {'Y': FFail['Y_I'][i],
                        'Z': FFail['Z_I'][i]},
                  'J': {'Y': FFail['Y_J'][i],
                        'Z': FFail['Z_J'][i]}}
                 for i in range(len(FrameList))}
    return FrameFail

