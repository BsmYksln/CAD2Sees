"""
Demand-capacity ratio analysis for structural components.

Calculates DCR for beam-column joints, frame elements, and infill panels
from OpenSees analysis results with multiple damage states.

Functions:
    BCJ: Beam-column joint DCR
    FrameShear: Frame shear DCR
    FrameRotation: Frame rotation DCR
    Infill: Infill panel DCR
"""

import os
import json
import numpy as np


def BCJ(OutDir, CapacityDir, step=-1, saveouts=1):
    """
    Calculate demand-capacity ratios for beam-column joints.

    Parameters
    ----------
    OutDir : str
        Output directory with analysis results.
    CapacityDir : str
        Directory with capacity data files.
    step : int, optional
        Analysis step (default: -1).
    saveouts : int, optional
        Save flag: 1=save files, 0=return arrays (default: 1).

    Returns
    -------
    list or None
        If saveouts=0: [DCRX, DCRY, DCR] arrays for 3 damage states.
        If saveouts=1: None (saves to BeamColumnJointDCR/ directory).
    """
    CapacityFile = os.path.join(CapacityDir, 'BCJoints.json')
    with open(CapacityFile, 'r') as f:
        BCJCapacity = json.load(f)

    BCJListFile = os.path.join(OutDir, 'OutBCJ.out')
    # BCJForceFile = os.path.join(OutDir, 'BCJForce.out')
    BCJDeformationFile = os.path.join(OutDir, 'BCJDeformation.out')

    BCJList = np.loadtxt(BCJListFile)[0, :]
    # BCJForce = np.loadtxt(BCJForceFile)
    try:
        BCJDeformation = np.loadtxt(BCJDeformationFile)
    except IndexError:
        print(OutDir)
    BCJoutLen = BCJDeformation.shape[1]

    Rotation_X_Def_idx = np.where(np.arange(BCJoutLen) % 6 == 3)[0]
    Rotation_X_Deformation = BCJDeformation[:, Rotation_X_Def_idx]

    Rotation_X_Def_CapDS1 = np.full(Rotation_X_Def_idx.shape, 1e4)
    Rotation_X_Def_CapDS2 = np.full(Rotation_X_Def_idx.shape, 1e4)
    Rotation_X_Def_CapDS3 = np.full(Rotation_X_Def_idx.shape, 1e4)

    Rotation_Y_Def_idx = np.where(np.arange(BCJoutLen) % 6 == 4)[0]
    Rotation_Y_Deformation = BCJDeformation[:, Rotation_Y_Def_idx]

    Rotation_Y_Def_CapDS1 = np.full(Rotation_Y_Def_idx.shape, 1e4)
    Rotation_Y_Def_CapDS2 = np.full(Rotation_Y_Def_idx.shape, 1e4)
    Rotation_Y_Def_CapDS3 = np.full(Rotation_Y_Def_idx.shape, 1e4)

    for k in BCJCapacity.keys():
        idx = BCJList == int(k)
        Rotation_X_Def_CapDS1[idx] = BCJCapacity[k]['Rot_Around_X'][1][0]
        Rotation_X_Def_CapDS2[idx] = BCJCapacity[k]['Rot_Around_X'][1][1]
        Rotation_X_Def_CapDS3[idx] = BCJCapacity[k]['Rot_Around_X'][1][2]

        Rotation_Y_Def_CapDS1[idx] = BCJCapacity[k]['Rot_Around_Y'][1][0]
        Rotation_Y_Def_CapDS2[idx] = BCJCapacity[k]['Rot_Around_Y'][1][1]
        Rotation_Y_Def_CapDS3[idx] = BCJCapacity[k]['Rot_Around_Y'][1][2]

    DCR_DS1Rotation_X = abs(Rotation_X_Deformation/Rotation_X_Def_CapDS1)
    DCR_DS2Rotation_X = abs(Rotation_X_Deformation/Rotation_X_Def_CapDS2)
    DCR_DS3Rotation_X = abs(Rotation_X_Deformation/Rotation_X_Def_CapDS3)

    DCR_DS1Rotation_Y = abs(Rotation_Y_Deformation/Rotation_Y_Def_CapDS1)
    DCR_DS2Rotation_Y = abs(Rotation_Y_Deformation/Rotation_Y_Def_CapDS2)
    DCR_DS3Rotation_Y = abs(Rotation_Y_Deformation/Rotation_Y_Def_CapDS3)

    DCR_DS1Rotation = np.maximum(DCR_DS1Rotation_X, DCR_DS1Rotation_Y)
    DCR_DS2Rotation = np.maximum(DCR_DS2Rotation_X, DCR_DS2Rotation_Y)
    DCR_DS3Rotation = np.maximum(DCR_DS3Rotation_X, DCR_DS3Rotation_Y)

    if saveouts == 1:
        DCROutDir = os.path.join(OutDir, 'BeamColumnJointDCR')
        if not os.path.exists(DCROutDir):
            os.makedirs(DCROutDir)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS1Rotation_X.csv'),
                   DCR_DS1Rotation_X)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS2Rotation_X.csv'),
                   DCR_DS2Rotation_X)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS3Rotation_X.csv'),
                   DCR_DS3Rotation_X)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS1Rotation_Y.csv'),
                   DCR_DS1Rotation_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS2Rotation_Y.csv'),
                   DCR_DS2Rotation_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS3Rotation_Y.csv'),
                   DCR_DS3Rotation_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS1Rotation.csv'),
                   DCR_DS1Rotation)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS2Rotation.csv'),
                   DCR_DS2Rotation)
        np.savetxt(os.path.join(DCROutDir, 'DCR_DS3Rotation.csv'),
                   DCR_DS3Rotation)
    else:
        DCRX = [DCR_DS1Rotation_X,
                DCR_DS2Rotation_X,
                DCR_DS3Rotation_X]
        DCRY = [DCR_DS1Rotation_Y,
                DCR_DS2Rotation_Y,
                DCR_DS3Rotation_Y]
        DCR = [DCR_DS1Rotation,
               DCR_DS2Rotation,
               DCR_DS3Rotation]
        return [DCRX, DCRY, DCR]


def FrameShear(OutDir, CapacityDir, step=-1, saveouts=1, addInfill=1,
               priestleyFlag=0, postflag=1):
    """
    Calculate demand-capacity ratios for frame shear forces.

    Parameters
    ----------
    OutDir : str
        Output directory with analysis results.
    CapacityDir : str
        Directory with capacity data files.
    step : int, optional
        Analysis step (default: -1).
    saveouts : int, optional
        Save flag: 1=save files, 0=return arrays (default: 1).
    addInfill : int, optional
        Include infill effects: 1=yes, 0=no (default: 1).
    priestleyFlag : int, optional
        Method: 0=EC8, 1=Priestley (default: 0).
    postflag : int, optional
        Calculation mode: 1=real-time, 0=pre-computed (default: 1).

    Returns
    -------
    list or None
        If saveouts=0: DCR arrays. If saveouts=1: None (saves to files).
    """

    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameList = np.loadtxt(FrameListFile)
    CapacityFile = os.path.join(CapacityDir, 'Frames.json')
    with open(CapacityFile, 'r') as f:
        FrameCapacity = json.load(f)

    if priestleyFlag == 1:
        FrameYeildDCRFile = os.path.join(OutDir, 'FrameRotationDCR',
                                         'DCRYeild.csv')
        FrameYeildDCR = np.loadtxt(FrameYeildDCRFile)
        FrameYeildDCR[FrameYeildDCR < 1] = 0
        FrameYeildDCR[FrameYeildDCR > 1] = 1

        ShearCapacity_Y_I = np.zeros(FrameYeildDCR.shape)
        ShearCapacity_Y_J = np.zeros(FrameYeildDCR.shape)
        ShearCapacity_Z_I = np.zeros(FrameYeildDCR.shape)
        ShearCapacity_Z_J = np.zeros(FrameYeildDCR.shape)

        ShearCapacity_Yeilded_Y_I = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Unyeilded_Y_I = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Yeilded_Z_I = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Unyeilded_Z_I = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Yeilded_Y_J = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Unyeilded_Y_J = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Yeilded_Z_J = np.zeros(len(FrameCapacity.keys()))
        ShearCapacity_Unyeilded_Z_J = np.zeros(len(FrameCapacity.keys()))
        for k in FrameCapacity.keys():
            idx = FrameList[0, :] == int(k)
            shear_data = FrameCapacity[k]['Shear']

            # Yielded capacities
            ShearCapacity_Yeilded_Y_I[idx] = (
                shear_data['Priestley_Shear_Yeild']['Iyy'])
            ShearCapacity_Unyeilded_Y_I[idx] = (
                shear_data['Priestley_Shear_Unyeild']['Iyy'])

            ShearCapacity_Yeilded_Z_I[idx] = (
                shear_data['Priestley_Shear_Yeild']['Izz'])
            ShearCapacity_Unyeilded_Z_I[idx] = (
                shear_data['Priestley_Shear_Unyeild']['Izz'])

            ShearCapacity_Yeilded_Y_J[idx] = (
                shear_data['Priestley_Shear_Yeild']['Jyy'])
            ShearCapacity_Unyeilded_Y_J[idx] = (
                shear_data['Priestley_Shear_Unyeild']['Jyy'])

            ShearCapacity_Yeilded_Z_J[idx] = (
                shear_data['Priestley_Shear_Yeild']['Jzz'])
            ShearCapacity_Unyeilded_Z_J[idx] = (
                shear_data['Priestley_Shear_Unyeild']['Jzz'])
        # Apply yielded/unyielded capacities based on DCR state
        yielded_mask = (FrameYeildDCR == 1)
        unyielded_mask = (FrameYeildDCR == 0)

        ShearCapacity_Y_I[yielded_mask] = np.broadcast_to(
            ShearCapacity_Yeilded_Y_I, FrameYeildDCR.shape)[yielded_mask]
        ShearCapacity_Y_I[unyielded_mask] = np.broadcast_to(
            ShearCapacity_Unyeilded_Y_I, FrameYeildDCR.shape)[unyielded_mask]

        ShearCapacity_Z_I[yielded_mask] = np.broadcast_to(
            ShearCapacity_Yeilded_Z_I, FrameYeildDCR.shape)[yielded_mask]
        ShearCapacity_Z_I[unyielded_mask] = np.broadcast_to(
            ShearCapacity_Unyeilded_Z_I, FrameYeildDCR.shape)[unyielded_mask]

        ShearCapacity_Y_J[yielded_mask] = np.broadcast_to(
            ShearCapacity_Yeilded_Y_J, FrameYeildDCR.shape)[yielded_mask]
        ShearCapacity_Y_J[unyielded_mask] = np.broadcast_to(
            ShearCapacity_Unyeilded_Y_J, FrameYeildDCR.shape)[unyielded_mask]

        ShearCapacity_Z_J[yielded_mask] = np.broadcast_to(
            ShearCapacity_Yeilded_Z_J, FrameYeildDCR.shape)[yielded_mask]
        ShearCapacity_Z_J[unyielded_mask] = np.broadcast_to(
            ShearCapacity_Unyeilded_Z_J, FrameYeildDCR.shape)[unyielded_mask]
    else:
        if postflag == 1:
            from cad2sees.helpers import units
            import pickle as pkl
            import copy
            # Get Section Properties
            # Push
            pkl_dir = os.path.normpath(os.path.join(OutDir, '..'))
            # THA
            # pkl_dir = os.path.normpath(OutDir)
            SectionFile = os.path.join(pkl_dir, 'Sections.pkl')
            with open(SectionFile, 'rb') as f:
                Sections = pkl.load(f)

            # Get Frame Properties
            FramesFile = os.path.join(pkl_dir, 'Frames.pkl')
            with open(FramesFile, 'rb') as f:
                Frames = pkl.load(f)

            # Get Nodes Properties
            NodeFile = os.path.join(pkl_dir, 'Nodes.pkl')
            with open(NodeFile, 'rb') as f:
                Nodes = pkl.load(f)

            dirs = ['y', 'z']
            ends = ['i', 'j']

            # Load Frame Local Forces
            FrameForceFile = os.path.join(OutDir, 'FrameLocalForce.out')
            FrameForce = np.loadtxt(FrameForceFile)
            FrameForceoutLen = FrameForce.shape[1]

            # Load Frame Inflection Points
            FrameInflectionFile = os.path.join(OutDir, 'InflectionPoint.out')
            FrameInflection = np.loadtxt(FrameInflectionFile)

            # Load Frame Chord Rotations
            FrameRotationFile = os.path.join(OutDir, 'ChordRotation.out')
            FrameRotation = np.loadtxt(FrameRotationFile)
            FrameRotationoutLen = FrameRotation.shape[1]

            # Frame Data Fetching
            frame_indices = np.arange(FrameForceoutLen) % 12
            rotation_indices = np.arange(FrameRotationoutLen) % 6

            FrameDetails = {
                'b': {'y_i': [], 'y_j': [], 'z_i': [], 'z_j': []},
                'Bw': {'y_i': [], 'y_j': [], 'z_i': [], 'z_j': []},
                'h': {'y_i': [], 'y_j': [], 'z_i': [], 'z_j': []},
                'fc': [],
                'fyw': [],
                'rho_w': {'y_i': [], 'y_j': [], 'z_i': [], 'z_j': []},
                'rho_L': {'i': [], 'j': []},
                'AG': {'i': [], 'j': []},
                'Vw': {'y_i': [], 'y_j': [], 'z_i': [], 'z_j': []},
                'Vn': {},
                'Vc': {},
                'Vcs': {},
                'R_Yeild': {'IY_P': [], 'IZ_P': [], 'JY_P': [], 'JZ_P': [],
                            'IY_N': [], 'IZ_N': [], 'JY_N': [], 'JZ_N': []},
                'D': {'y_i': [], 'z_i': [], 'y_j': [], 'z_j': []},
                'L': [],
                'is_column': [],
                'Lv': {}
            }

            # Get Axial Force
            N_I_idx = np.where(frame_indices == 0)[0]
            N_J_idx = np.where(frame_indices == 6)[0]
            # Possitive == Compression
            N_I = FrameForce[:, N_I_idx]
            N_J = -FrameForce[:, N_J_idx]
            FrameDetails['N'] = {'i': N_I, 'j': N_J}

            # Get Shear
            Shear_Z_I_idx = np.where(frame_indices == 2)[0]
            Shear_Z_J_idx = np.where(frame_indices == 8)[0]
            Shear_Y_I_idx = np.where(frame_indices == 1)[0]
            Shear_Y_J_idx = np.where(frame_indices == 7)[0]

            Shear_Z_I = FrameForce[:, Shear_Z_I_idx]
            Shear_Z_J = FrameForce[:, Shear_Z_J_idx]
            Shear_Y_I = FrameForce[:, Shear_Y_I_idx]
            Shear_Y_J = FrameForce[:, Shear_Y_J_idx]
            for f_idx in FrameList[0, :]:
                f_idx_in_frames = np.where(Frames['ID'] == f_idx)[0][0]
                SectionName_I = Frames['SectionTypI'][f_idx_in_frames]
                if f_idx == 100118:
                    print('Frame 100118')
                I_id = Frames['i_ID'][f_idx_in_frames]
                J_id = Frames['j_ID'][f_idx_in_frames]
                Node_idx_i = np.where(Nodes['ID'] == I_id)[0][0]
                Node_idx_j = np.where(Nodes['ID'] == J_id)[0][0]
                Coordinate_I = Nodes['Coordinates'][Node_idx_i, :]
                Coordinate_J = Nodes['Coordinates'][Node_idx_j, :]  #
                Length = np.linalg.norm(Coordinate_I - Coordinate_J)*units.cm
                FrameDetails['L'].append(Length)

                if 'Column' in SectionName_I:
                    FrameDetails['is_column'].append(1)
                else:
                    FrameDetails['is_column'].append(0)

                fc_cur = Sections[SectionName_I]['fc0']
                fyw_cur = Sections[SectionName_I]['fyw']
                FrameDetails['fc'].append(fc_cur)
                FrameDetails['fyw'].append(fyw_cur)

                for end in ends:
                    SectionName = (
                        (Frames[f'SectionTyp{end.upper()}']
                         [f_idx_in_frames]))
                    CurSec = Sections[SectionName]
                    ReinfsAll = CurSec['ReinfL'][:, -1]
                    As_L = np.sum(0.25*units.pi*(ReinfsAll*units.mm)**2)
                    cur_sec_b = CurSec['b']*units.mm
                    cur_sec_h = CurSec['h']*units.mm
                    cur_sec_s = CurSec['s']*units.mm
                    cv_cur = CurSec['Cover']*units.mm
                    dbl_cur = np.mean(ReinfsAll**2)**0.5*units.mm
                    phit_cur = CurSec['phi_T']*units.mm
                    for dir in dirs:
                        # Get Yeild Rotation
                        rotdir = end.upper() + dir + dir
                        frame_cap = FrameCapacity[str(int(f_idx))]['Flexural']
                        R_Yeild_P_cur = frame_cap[rotdir]['thetaYp']
                        R_Yeild_N_cur = frame_cap[rotdir]['thetaYn']
                        key_p = f'{end.upper()}{dir.upper()}_P'
                        key_n = f'{end.upper()}{dir.upper()}_N'
                        FrameDetails['R_Yeild'][key_p].append(R_Yeild_P_cur)
                        FrameDetails['R_Yeild'][key_n].append(R_Yeild_N_cur)
                        # Get Transversal Reinforcement Contribution
                        if dir == 'y':
                            bcur = copy.deepcopy(cur_sec_b)
                            hcur = copy.deepcopy(cur_sec_h)
                        else:
                            bcur = copy.deepcopy(cur_sec_h)
                            hcur = copy.deepcopy(cur_sec_b)
                        FrameDetails['b'][dir+'_'+end].append(bcur)
                        FrameDetails['h'][dir+'_'+end].append(hcur)
                        numofstr = CurSec[f'NumofStr{dir.upper()}Dir']
                        d_cur = 0.9*(hcur-dbl_cur*0.5-cv_cur-phit_cur)
                        bw_cur = bcur-cv_cur*2
                        Av_cur = numofstr*0.25*units.pi*phit_cur**2
                        rhow_cur = Av_cur/(bw_cur*cur_sec_s)
                        FrameDetails['rho_w'][dir+'_'+end].append(rhow_cur)
                        FrameDetails['D'][dir+'_'+end].append(d_cur)
                        VwCur = rhow_cur*bw_cur*d_cur*fyw_cur
                        FrameDetails['Vw'][dir+'_'+end].append(VwCur)
                    AgCur = bcur*hcur
                    FrameDetails['AG'][end].append(AgCur)
                    FrameDetails['rho_L'][end].append(As_L/AgCur)

            Lv_Y_I = FrameInflection[:, ::2]
            Lv_Y_I[Lv_Y_I < 0] = 0
            Ls = np.tile(np.array(FrameDetails['L']), (Lv_Y_I.shape[0], 1))
            Lv_Y_I[Lv_Y_I > Ls] = Ls[Lv_Y_I > Ls]
            Lv_Z_I = FrameInflection[:, 1::2]
            Lv_Z_I[Lv_Z_I < 0] = 0
            Lv_Z_I[Lv_Z_I > Ls] = Ls[Lv_Z_I > Ls]
            Lv_Y_J = Ls - Lv_Y_I
            Lv_Z_J = Ls - Lv_Z_I
            FrameDetails['Lv']['y_i'] = Lv_Y_I
            FrameDetails['Lv']['z_i'] = Lv_Z_I
            FrameDetails['Lv']['y_j'] = Lv_Y_J
            FrameDetails['Lv']['z_j'] = Lv_Z_J

            # Get Rotation
            mu = {
                'y': {'i': [], 'j': []},
                'z': {'i': [], 'j': []}
            }
            mu_pl = mu.copy()
            index_keys_rotation = {'z_i': 1, 'z_j': 2, 'y_i': 3, 'y_j': 4}
            for dir in dirs:
                for end in ends:
                    rotation_idx = np.where(rotation_indices == index_keys_rotation[f'{dir}_{end}'])[0]
                    Rotation = FrameRotation[:, rotation_idx]
                    R_Yeild_P = FrameDetails['R_Yeild'][f'{end.upper()}{dir.upper()}_P']
                    R_Yeild_N = FrameDetails['R_Yeild'][f'{end.upper()}{dir.upper()}_N']
                    R_Yeild_P = np.tile(np.array(R_Yeild_P), (Rotation.shape[0], 1))
                    R_Yeild_N = np.tile(np.array(R_Yeild_N), (Rotation.shape[0], 1))
                    R_Yeild = np.zeros(Rotation.shape)
                    R_Yeild[Rotation >= 0] = R_Yeild_P[Rotation >= 0]
                    R_Yeild[Rotation < 0] = R_Yeild_N[Rotation < 0]
                    mu[dir][end] = abs(Rotation)/R_Yeild
                    mu_pl[dir][end] = np.maximum((abs(Rotation)/R_Yeild) - 1, 0)

            # Normal Force Term
            def getNormalForceTerm(N, Lv, h, b, fc):
                hcur = np.tile(h, (N.shape[0], 1))
                bcur = np.tile(b, (N.shape[0], 1))
                fc_cur = np.tile(fc, (N.shape[0], 1))
                nu = (N/units.N)/(hcur*bcur*fc_cur/(units.mm**2))
                x = h*np.minimum((0.25+0.85*nu), 1)
                Vn = np.zeros(N.shape)
                non_zero_Lv = Lv != 0
                # Split complex calculation for readability
                term1 = ((hcur-x)[non_zero_Lv])/(2*Lv[non_zero_Lv])
                term2 = np.minimum(np.abs(N)/units.MN, 0.55*bcur*hcur*fc)
                Vn[non_zero_Lv] = term1 * term2[non_zero_Lv]
                return Vn

            # Term Concrete Contribution
            def getConcreteTerm(rho_L, Lv, h, b, fc):
                Vc = 0.16*np.maximum(0.5, 100*rho_L)*(1-0.16*np.minimum(5, Lv/h))*b*h*fc**0.5
                return Vc

            for end in ends:
                for dir in dirs:
                    Vn_cur = getNormalForceTerm(
                        FrameDetails['N'][end],
                        FrameDetails['Lv'][f'{dir}_{end}'],
                        np.array(FrameDetails['h'][f'{dir}_{end}']),
                        np.array(FrameDetails['b'][f'{dir}_{end}']),
                        np.array(FrameDetails['fc']))
                    FrameDetails['Vn'][f'{dir}_{end}'] = Vn_cur
                    Vc_cur = getConcreteTerm(
                        np.array(FrameDetails['rho_L'][end]),
                        FrameDetails['Lv'][f'{dir}_{end}'],
                        np.array(FrameDetails['h'][f'{dir}_{end}']),
                        np.array(FrameDetails['b'][f'{dir}_{end}']),
                        np.array(FrameDetails['fc']))
                    FrameDetails['Vc'][f'{dir}_{end}'] = Vc_cur
                    Vw_cur = np.tile(
                        np.array(FrameDetails['Vw'][f'{dir}_{end}']),
                        (Vc_cur.shape[0], 1))
                    Vcs_cur = ((1-0.05*np.minimum(5, mu_pl[dir][end])) *
                               (Vw_cur + Vc_cur))
                    FrameDetails['Vcs'][f'{dir}_{end}'] = Vcs_cur
            gamma_el = 1.15
            ShearCapacity_Y_I = (FrameDetails['Vn']['y_i'] +
                                 FrameDetails['Vcs']['y_i'])*units.MN/gamma_el
            ShearCapacity_Z_I = (FrameDetails['Vn']['z_i'] +
                                 FrameDetails['Vcs']['z_i'])*units.MN/gamma_el
            ShearCapacity_Y_J = (FrameDetails['Vn']['y_j'] +
                                 FrameDetails['Vcs']['y_j'])*units.MN/gamma_el
            ShearCapacity_Z_J = (FrameDetails['Vn']['z_j'] +
                                 FrameDetails['Vcs']['z_j'])*units.MN/gamma_el
        else:
            ShearCapacity_Y_I = np.zeros(len(FrameCapacity.keys()))
            ShearCapacity_Y_J = np.zeros(len(FrameCapacity.keys()))
            ShearCapacity_Z_I = np.zeros(len(FrameCapacity.keys()))
            ShearCapacity_Z_J = np.zeros(len(FrameCapacity.keys()))

            for k in FrameCapacity.keys():
                idx = FrameList[0, :] == int(k)
                ShearCapacity_Y_I[idx] = FrameCapacity[k]['Shear']['EC8_Shear']['Iyy']
                ShearCapacity_Y_J[idx] = FrameCapacity[k]['Shear']['EC8_Shear']['Jyy']
                ShearCapacity_Z_I[idx] = FrameCapacity[k]['Shear']['EC8_Shear']['Izz']
                ShearCapacity_Z_J[idx] = FrameCapacity[k]['Shear']['EC8_Shear']['Jzz']
    FrameForceFile = os.path.join(OutDir, 'FrameLocalForce.out')
    FrameForce = np.loadtxt(FrameForceFile)
    FrameoutLen = FrameForce.shape[1]

    Shear_Z_I_idx = np.where(np.arange(FrameoutLen) % 12 == 2)[0]
    Shear_Z_J_idx = np.where(np.arange(FrameoutLen) % 12 == 8)[0]
    Shear_Y_I_idx = np.where(np.arange(FrameoutLen) % 12 == 1)[0]
    Shear_Y_J_idx = np.where(np.arange(FrameoutLen) % 12 == 7)[0]

    Shear_Z_I = FrameForce[:, Shear_Z_I_idx]
    Shear_Z_J = FrameForce[:, Shear_Z_J_idx]
    Shear_Y_I = FrameForce[:, Shear_Y_I_idx]
    Shear_Y_J = FrameForce[:, Shear_Y_J_idx]

    infinfo = ''
    if addInfill == 1:
        infinfo = 'WithInfill'
        InfillOutFile = os.path.join(CapacityDir, 'Infills.json')
        with open(InfillOutFile, 'r') as f:
            InfillData = json.load(f)

        ShearInfill_Z_I = np.zeros(Shear_Z_I.shape)
        ShearInfill_Z_J = np.zeros(Shear_Z_J.shape)
        ShearInfill_Y_I = np.zeros(Shear_Y_I.shape)
        ShearInfill_Y_J = np.zeros(Shear_Y_J.shape)

        InfillListFile = os.path.join(OutDir, 'OutInfills.out')
        InfillForceFile = os.path.join(OutDir, 'InfillForce.out')
        InfillList = np.loadtxt(InfillListFile)
        InfillForce = np.loadtxt(InfillForceFile)

        for k in InfillData.keys():
            idx = np.where(InfillList == int(k))[0][0]
            CurInfillForce = InfillForce[:, idx]
            for i in range(4):
                RelatedFrame = InfillData[k]['Related_Frames'][i]
                if RelatedFrame != 0:
                    FrameIdx = np.where(FrameList[0, :] == RelatedFrame)[0][0]
                    FrameData = FrameList[:, FrameIdx]
                    RelatedNode = InfillData[k]['Related_Nodes'][i]

                    Alpha = InfillData[k]['Alphas'][i]
                    CurShearInfill2Frame = abs(CurInfillForce*Alpha)

                    # Find Frame Direction
                    Frame_Direction = (
                        Frames['Direction']
                        [np.where(FrameData == RelatedNode)[0][0]])
                    if Frame_Direction != 3:
                        if np.where(FrameData == RelatedNode)[0][0] == 2:
                            ShearInfill_Y_J[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Y_J[:, FrameIdx])
                        elif np.where(FrameData == RelatedNode)[0][0] == 1:
                            ShearInfill_Y_I[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Y_I[:, FrameIdx])
                    else:
                        # Direction -> 5 Shear Y
                        # Direction -> 4 Shear Z
                        if InfillData[k]['Direction'] == 5:
                            if np.where(FrameData == RelatedNode)[0][0] == 2:
                                ShearInfill_Y_J[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Y_J[:, FrameIdx])
                            elif np.where(FrameData == RelatedNode)[0][0] == 1:
                                ShearInfill_Y_I[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Y_I[:, FrameIdx])
                        elif InfillData[k]['Direction'] == 4:
                            if np.where(FrameData == RelatedNode)[0][0] == 2:
                                ShearInfill_Z_J[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Z_J[:, FrameIdx])
                            elif np.where(FrameData == RelatedNode)[0][0] == 1:
                                ShearInfill_Z_I[:, FrameIdx] = np.maximum(CurShearInfill2Frame, ShearInfill_Z_I[:, FrameIdx])

        Shear_Z_I = np.maximum(abs(Shear_Z_I), ShearInfill_Z_I)
        Shear_Z_J = np.maximum(abs(Shear_Z_J), ShearInfill_Z_J)
        Shear_Y_I = np.maximum(abs(Shear_Y_I), ShearInfill_Y_I)
        Shear_Y_J = np.maximum(abs(Shear_Y_J), ShearInfill_Y_J)

    Shear_Z = np.maximum(abs(Shear_Z_I), abs(Shear_Z_J))
    Shear_Y = np.maximum(abs(Shear_Y_I), abs(Shear_Y_J))

    DCRShear_Y_I = abs(Shear_Y_I/ShearCapacity_Y_I)
    DCRShear_Y_J = abs(Shear_Y_J/ShearCapacity_Y_J)
    DCRShear_Z_I = abs(Shear_Z_I/ShearCapacity_Z_I)
    DCRShear_Z_J = abs(Shear_Z_J/ShearCapacity_Z_J)

    DCRShear_Z = np.maximum(DCRShear_Z_I, DCRShear_Z_J)
    DCRShear_Y = np.maximum(DCRShear_Y_I, DCRShear_Y_J)
    DCRShear = np.maximum(DCRShear_Y, DCRShear_Z)

    if saveouts == 1:
        DCROutDir = os.path.join(OutDir, 'FrameShearDCR')
        if not os.path.exists(DCROutDir):
            os.makedirs(DCROutDir)
        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Y_I.csv'),
                   DCRShear_Y_I)
        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Y_J.csv'),
                   DCRShear_Y_J)

        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Z_I.csv'),
                   DCRShear_Z_I)
        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Z_J.csv'),
                   DCRShear_Z_J)

        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Z.csv'),
                   DCRShear_Z)
        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}_Y.csv'),
                   DCRShear_Y)

        np.savetxt(os.path.join(DCROutDir, f'DCRShear{infinfo}.csv'),
                   DCRShear)
    else:
        DCRShearZ = [DCRShear_Z_I,
                     DCRShear_Z_J]

        DCRShearY = [DCRShear_Y_I,
                     DCRShear_Y_J]

        DCRShears = [DCRShear_Z,
                     DCRShear_Y]

        return DCRShearZ, DCRShearY, DCRShears, DCRShear


def FrameRotation(OutDir, CapacityDir, step=-1, saveouts=1):
    """
    Calculate demand-capacity ratios for frame rotations.

    Analyses chord rotation demands and compares against yield and ultimate
    capacity limits for rotations about Y and Z axes at element ends.

    Parameters
    ----------
    OutDir : str
        Output directory with OpenSees analysis results.
    CapacityDir : str
        Directory with capacity data files (Frames.json).
    step : int, optional
        Analysis step (default: -1 for all steps).
    saveouts : int, optional
        Output save flag:
        - 0: Return DCR arrays without saving
        - 1: Save DCR results to CSV files
        - 2: Save DCR results and return arrays

    Returns
    -------
    list or None
        Returns [DCRZ, DCRY, DCRYZs, DCRIJs, DCRRotation] where:
        - DCRZ: Z-direction DCR arrays [I-end, J-end]
        - DCRY: Y-direction DCR arrays [I-end, J-end]
        - DCRYZs: Combined direction DCR arrays [Y, Z]
        - DCRIJs: Element end DCR arrays [I-end, J-end]
        - DCRRotation: Overall rotation DCR array
        Returns None if saveouts=1.

    Notes
    -----
    Evaluates both yield and ultimate capacity levels:
    - Yield Capacity: First yielding of reinforcement
    - Ultimate Capacity: Maximum rotation before failure
    
    Files saved to FrameRotationDCR/ directory with ultimate and yield DCR
    results for all direction/end combinations.
    """
    CapacityFile = os.path.join(CapacityDir, 'Frames.json')
    with open(CapacityFile, 'r') as f:
        FrameCapacity = json.load(f)

    FrameListFile = os.path.join(OutDir, 'OutFrames.out')
    FrameList = np.loadtxt(FrameListFile)
    FrameRotationFile = os.path.join(OutDir, 'ChordRotation.out')
    FrameRotation = np.loadtxt(FrameRotationFile)

    RotationCapacity_Y_I_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Y_J_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Z_I_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Z_J_P = np.zeros(len(FrameCapacity.keys()))

    RotationCapacity_Y_I_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Y_J_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Z_I_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Z_J_N = np.zeros(len(FrameCapacity.keys()))

    RotationCapacity_Yeild_IY_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_IZ_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_JY_N = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_JZ_N = np.zeros(len(FrameCapacity.keys()))

    RotationCapacity_Yeild_IY_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_IZ_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_JY_P = np.zeros(len(FrameCapacity.keys()))
    RotationCapacity_Yeild_JZ_P = np.zeros(len(FrameCapacity.keys()))

    FrameoutLen = FrameRotation.shape[1]

    Rotation_Z_I_idx = np.where(np.arange(FrameoutLen) % 6 == 1)[0]
    Rotation_Z_J_idx = np.where(np.arange(FrameoutLen) % 6 == 2)[0]
    Rotation_Y_I_idx = np.where(np.arange(FrameoutLen) % 6 == 3)[0]
    Rotation_Y_J_idx = np.where(np.arange(FrameoutLen) % 6 == 4)[0]
    
    Rotation_Z_I = FrameRotation[:, Rotation_Z_I_idx]
    Rotation_Z_J = FrameRotation[:, Rotation_Z_J_idx]
    Rotation_Y_I = FrameRotation[:, Rotation_Y_I_idx]
    Rotation_Y_J = FrameRotation[:, Rotation_Y_J_idx]

    for k in FrameCapacity.keys():
        idx = FrameList[0, :] == int(k)
        RotationCapacity_Y_I_P[idx] = FrameCapacity[k]['Flexural']['Iyy']['thetaUmP']
        RotationCapacity_Y_J_P[idx] = FrameCapacity[k]['Flexural']['Jyy']['thetaUmP']
        RotationCapacity_Z_I_P[idx] = FrameCapacity[k]['Flexural']['Izz']['thetaUmP']
        RotationCapacity_Z_J_P[idx] = FrameCapacity[k]['Flexural']['Jzz']['thetaUmP']

        RotationCapacity_Y_I_N[idx] = FrameCapacity[k]['Flexural']['Iyy']['thetaUmN']
        RotationCapacity_Y_J_N[idx] = FrameCapacity[k]['Flexural']['Jyy']['thetaUmN']
        RotationCapacity_Z_I_N[idx] = FrameCapacity[k]['Flexural']['Izz']['thetaUmN']
        RotationCapacity_Z_J_N[idx] = FrameCapacity[k]['Flexural']['Jzz']['thetaUmN']

        RotationCapacity_Yeild_IY_P[idx] = FrameCapacity[k]['Flexural']['Iyy']['thetaYp']
        RotationCapacity_Yeild_IZ_P[idx] = FrameCapacity[k]['Flexural']['Izz']['thetaYp']
        RotationCapacity_Yeild_JY_P[idx] = FrameCapacity[k]['Flexural']['Jyy']['thetaYp']
        RotationCapacity_Yeild_JZ_P[idx] = FrameCapacity[k]['Flexural']['Jzz']['thetaYp']

        RotationCapacity_Yeild_IY_N[idx] = FrameCapacity[k]['Flexural']['Iyy']['thetaYn']
        RotationCapacity_Yeild_IZ_N[idx] = FrameCapacity[k]['Flexural']['Izz']['thetaYn']
        RotationCapacity_Yeild_JY_N[idx] = FrameCapacity[k]['Flexural']['Jyy']['thetaYn']
        RotationCapacity_Yeild_JZ_N[idx] = FrameCapacity[k]['Flexural']['Jzz']['thetaYn']

    DCR_Z_I = Rotation_Z_I/RotationCapacity_Z_I_P
    DCR_Z_I[DCR_Z_I < 0] = abs(Rotation_Z_I/RotationCapacity_Z_I_N)[DCR_Z_I < 0]

    DCR_Y_I = Rotation_Y_I/RotationCapacity_Y_I_P
    DCR_Y_I[DCR_Y_I < 0] = abs(Rotation_Y_I/RotationCapacity_Y_I_N)[DCR_Y_I < 0]

    DCR_Z_J = Rotation_Z_J/RotationCapacity_Z_J_P
    DCR_Z_J[DCR_Z_J < 0] = abs(Rotation_Z_J/RotationCapacity_Z_J_N)[DCR_Z_J < 0]

    DCR_Y_J = Rotation_Y_J/RotationCapacity_Y_J_P
    DCR_Y_J[DCR_Y_J < 0] = abs(Rotation_Y_J/RotationCapacity_Y_J_N)[DCR_Y_J < 0]

    DCR_I = np.maximum(DCR_Z_I, DCR_Y_I)
    DCR_J = np.maximum(DCR_Z_J, DCR_Y_J)

    DCR_Z = np.maximum(DCR_Z_I, DCR_Z_J)
    DCR_Y = np.maximum(DCR_Y_I, DCR_Y_J)

    DCR_Yeild_Y_I = Rotation_Y_I/RotationCapacity_Yeild_IY_P
    DCR_Yeild_Y_I[DCR_Yeild_Y_I < 0] = abs(Rotation_Y_I/RotationCapacity_Yeild_IY_N)[DCR_Yeild_Y_I < 0]

    DCR_Yeild_Z_I = Rotation_Z_I/RotationCapacity_Yeild_IZ_P
    DCR_Yeild_Z_I[DCR_Yeild_Z_I < 0] = abs(Rotation_Z_I/RotationCapacity_Yeild_IZ_N)[DCR_Yeild_Z_I < 0]

    DCR_Yeild_Y_J = Rotation_Y_J/RotationCapacity_Yeild_JY_P
    DCR_Yeild_Y_J[DCR_Yeild_Y_J < 0] = abs(Rotation_Y_J/RotationCapacity_Yeild_JY_N)[DCR_Yeild_Y_J < 0]

    DCR_Yeild_Z_J = Rotation_Z_J/RotationCapacity_Yeild_JZ_P
    DCR_Yeild_Z_J[DCR_Yeild_Z_J < 0] = abs(Rotation_Z_J/RotationCapacity_Yeild_JZ_N)[DCR_Yeild_Z_J < 0]

    DCR_Yeild_I = np.maximum(DCR_Yeild_Y_I, DCR_Yeild_Z_I)
    DCR_Yeild_J = np.maximum(DCR_Yeild_Y_J, DCR_Yeild_Z_J)

    DCR_Yeild_Y = np.maximum(DCR_Yeild_Y_I, DCR_Yeild_Y_J)
    DCR_Yeild_Z = np.maximum(DCR_Yeild_Z_I, DCR_Yeild_Z_J)

    DCRRotation = np.maximum(DCR_Z, DCR_Y)
    DCRYeild = np.maximum(DCR_Yeild_Y, DCR_Yeild_Z)

    if saveouts == 1:
        DCROutDir = os.path.join(OutDir, 'FrameRotationDCR')
        if not os.path.exists(DCROutDir):
            os.makedirs(DCROutDir)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z_I.csv'),
                   DCR_Z_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z_J.csv'),
                   DCR_Z_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y_I.csv'),
                   DCR_Y_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y_J.csv'),
                   DCR_Y_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y.csv'),
                   DCR_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z.csv'),
                   DCR_Z)
        
        np.savetxt(os.path.join(DCROutDir, 'DCR_I.csv'),
                   DCR_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_J.csv'),
                   DCR_J)
        
        np.savetxt(os.path.join(DCROutDir, 'DCR.csv'),
                   DCRRotation)

        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z_I.csv'),
                   DCR_Yeild_Z_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z_J.csv'),
                   DCR_Yeild_Z_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y_I.csv'),
                   DCR_Yeild_Y_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y_J.csv'),
                   DCR_Yeild_Y_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y.csv'),
                   DCR_Yeild_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z.csv'),
                   DCR_Yeild_Z)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_I.csv'),
                   DCR_Yeild_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_J.csv'),
                   DCR_Yeild_J)
        np.savetxt(os.path.join(DCROutDir, 'DCRYeild.csv'),
                   DCRYeild)
    elif saveouts == 2:
        DCROutDir = os.path.join(OutDir, 'FrameRotationDCR')
        if not os.path.exists(DCROutDir):
            os.makedirs(DCROutDir)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z_I.csv'),
                   DCR_Z_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z_J.csv'),
                   DCR_Z_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y_I.csv'),
                   DCR_Y_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y_J.csv'),
                   DCR_Y_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Y.csv'),
                   DCR_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Z.csv'),
                   DCR_Z)
        
        np.savetxt(os.path.join(DCROutDir, 'DCR_I.csv'),
                   DCR_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_J.csv'),
                   DCR_J)
        
        np.savetxt(os.path.join(DCROutDir, 'DCR.csv'),
                   DCRRotation)

        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z_I.csv'),
                   DCR_Yeild_Z_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z_J.csv'),
                   DCR_Yeild_Z_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y_I.csv'),
                   DCR_Yeild_Y_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y_J.csv'),
                   DCR_Yeild_Y_J)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Y.csv'),
                   DCR_Yeild_Y)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_Z.csv'),
                   DCR_Yeild_Z)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_I.csv'),
                   DCR_Yeild_I)
        np.savetxt(os.path.join(DCROutDir, 'DCR_Yeild_J.csv'),
                   DCR_Yeild_J)
        np.savetxt(os.path.join(DCROutDir, 'DCRYeild.csv'),
                   DCRYeild)
        DCRZ = [DCR_Z_I,
                DCR_Z_J]

        DCRY = [DCR_Y_I,
                DCR_Y_J]

        DCRYZs = [DCR_Y,
                  DCR_Z]

        DCRIJs = [DCR_I,
                  DCR_J]

        return [DCRZ, DCRY, DCRYZs, DCRIJs, DCRRotation]
    elif saveouts == 0:
        DCRZ = [DCR_Z_I,
                DCR_Z_J]

        DCRY = [DCR_Y_I,
                DCR_Y_J]

        DCRYZs = [DCR_Y,
                  DCR_Z]

        DCRIJs = [DCR_I,
                  DCR_J]

        return [DCRZ, DCRY, DCRYZs, DCRIJs, DCRRotation]


def Infill(OutDir, CapacityDir, step=-1, saveouts=1):
    """
    Calculate demand-capacity ratios for infill panels.

    Analyses deformation demands in masonry infill panels against capacity
    limits for four progressive damage states.

    Parameters
    ----------
    OutDir : str
        Output directory with OpenSees analysis results.
    CapacityDir : str
        Directory with capacity data files (Infills.json).
    step : int, optional
        Analysis step (default: -1 for all steps).
    saveouts : int, optional
        Output save flag:
        - 1: Save DCR results to CSV files (default)
        - 0: Return DCR arrays without saving

    Returns
    -------
    list or None
        If saveouts=0, returns [DCR_DS1, DCR_DS2, DCR_DS3, DCR_DS4]
        for each damage state. If saveouts=1, returns None and saves
        results to InfillDCR/ directory.

    Notes
    -----
    Evaluates four progressive damage states:
    - DS1: Light cracking
    - DS2: Moderate cracking
    - DS3: Severe cracking
    - DS4: Partial collapse
    
    Capacity limits are defined in Infills.json based on experimental
    and analytical studies of masonry infill behaviour.
    """
    InfillOutFile = os.path.join(CapacityDir, 'Infills.json')
    with open(InfillOutFile, 'r') as f:
        InfillData = json.load(f)
    InfillListFile = os.path.join(OutDir, 'OutInfills.out')
    InfillList = np.loadtxt(InfillListFile)
    InfillDeformationFile = os.path.join(OutDir, 'InfillDeformation.out')
    InfillDeformation = np.loadtxt(InfillDeformationFile)
    num_DS = 4
    for i in range(1, num_DS+1):
        vars()[f'InfillCapacity_DS{i}'] = np.zeros(InfillDeformation.shape[1])
    for k in InfillData.keys():
        idx = np.where(InfillList == int(k))[0][0]
        for i in range(1, num_DS+1):
            vars()[f'InfillCapacity_DS{i}'][idx] = InfillData[k]['ULimits'][i-1]
    for i in range(1, num_DS+1):
        vars()[f'DCR_DS{i}'] = InfillDeformation/vars()[f'InfillCapacity_DS{i}']
    if saveouts == 1:
        DCROutDir = os.path.join(OutDir, 'InfillDCR')
        if not os.path.exists(DCROutDir):
            os.makedirs(DCROutDir)
        for i in range(1, num_DS+1):
            np.savetxt(os.path.join(DCROutDir, f'DCR_DS{i}.csv'),
                       vars()[f'DCR_DS{i}'])
    else:
        return [vars()[f'DCR_DS{i}'] for i in range(1,num_DS+1)]
