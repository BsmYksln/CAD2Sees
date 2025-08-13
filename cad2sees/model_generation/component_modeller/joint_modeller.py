"""
CAD2Sees module dedicated to modeling beam-column joints in RC frame structures.

Provides classes for modelling beam-column joints in RC frame structures
using calibrated joint models with zero-length elements.

References
----------
O'Reilly, G. J., and T. J. Sullivan. 2019. "Modeling Techniques for the
Seismic Assessment of the Existing Italian RC Frame Structures." Journal
of Earthquake Engineering 23 (8): 1262–1296.
https://doi.org/10.1080/13632469.2017.1360224
"""

import openseespy.opensees as ops
from cad2sees.helpers import units
from cad2sees.struct_utils import capacity as C
import numpy as np


class Joint:
    """
    Models reinforced concrete beam-column joints using zero-length elements.

    Creates joint models for seismic analysis considering location
    (exterior/interior), seismic detailing level, material properties,
    and hysteretic behaviour calibrated from experimental data.

    Parameters
    ----------
    JointType : int
        Joint type identifier (0 = no seismic detail)
    JointData : dict
        Joint data including coordinates, loads, location
    ColumnSection : dict
        Column section properties
    BeamXSections : list
        Beam sections in X-direction
    BeamYSections : list
        Beam sections in Y-direction

    Methods
    -------
    build()
        Create the joint model in OpenSees
    capacity()
        Calculate joint moment capacity
    buildWcapacity()
        Build model and calculate capacity
    """
    def __init__(self,
                 JointType,
                 JointData,
                 ColumnSection,
                 BeamXSections,
                 BeamYSections):
        # Beam Column Joint Properties
        #
        # !! Important Joint1 model do not consider the variation of sections
        # !! for continous elements [L-R Beam or T-B Col]
        # !! Thus, in here we assume the min section dimensions
        #
        # (These figures come from the calibration done by
        #                            O'Reilly and Sullivan [2019])
        self.Out = {}

        self.JT = JointType
        self.JD = JointData
        self.CS = ColumnSection
        self.BXS = BeamXSections
        self.BYS = BeamYSections

        self.X = self.JD['X']
        self.Y = self.JD['Y']
        self.Z = self.JD['Z']
        self.NLoad = self.JD['NLoad']
        self.NMass = self.JD['NMass']

        # Column dimensions
        self.hcY = self.CS['h']*units.mm
        self.hcX = self.CS['b']*units.mm
        self.Ac = self.hcX*self.hcY

        # Beams dimensions
        if self.JD['JointLocX'] != 3:  # If BC-Joint Exists at X-X dir
            hbXs = [self.BXS[k]['h'] for k in range(len(self.BXS))]
            self.hbX = min(hbXs)*units.mm
            bbXs = [self.BXS[k]['b'] for k in range(len(self.BXS))]
            self.bbX = min(bbXs)*units.mm
            # Bar information from Beams
            dbLXs = [(self.BXS[k]['ReinfL'][:, -1]**2).mean()**0.5
                     for k in range(len(self.BXS))]
            self.dbLX = np.mean(dbLXs)*units.mm
            dbVXs = [self.BXS[k]['phi_T'] for k in range(len(self.BXS))]
            self.dbVX = np.mean(dbVXs)*units.mm
            cvXs = [self.BXS[k]['Cover'] for k in range(len(self.BXS))]
            self.cv = np.mean(cvXs)*units.mm
        else:
            self.hbX = 0
            self.bbX = 0
            self.dbLX = 0
            self.dbVX = 0

        if self.JD['JointLocY'] != 3:  # If BC-Joint Exists at Y-Y dir
            hbYs = [self.BYS[k]['h'] for k in range(len(self.BYS))]
            self.hbY = min(hbYs)*units.mm
            bbYs = [self.BYS[k]['b'] for k in range(len(self.BYS))]
            self.bbY = min(bbYs)*units.mm
            # Bar information from Beams
            dbLYs = [(self.BYS[k]['ReinfL'][:, -1]**2).mean()**0.5
                     for k in range(len(self.BYS))]
            self.dbLY = np.mean(dbLYs)*units.mm
            dbVYs = [self.BYS[k]['phi_T'] for k in range(len(self.BYS))]
            self.dbVY = np.mean(dbVYs)*units.mm
            cvYs = [self.BYS[k]['Cover'] for k in range(len(self.BYS))]
            self.cv = np.mean(cvYs)*units.mm
        else:
            self.hbY = 0
            self.bbY = 0
            self.dbLY = 0
            self.dbVY = 0

        self.hb = max([self.hbX, self.hbY])

        # Concrete material props
        self.fc = self.CS['fc0']*units.MPa
        self.Ec = self.CS['Ec']*units.MPa

        # Detirmine connection witdh
        # Around X-X Axis
        self.bjX = (min(self.hcX, self.bbY+0.5*self.hcY) if
                    self.hcX >= self.bbY else min(self.bbY,
                                                  self.hcX+0.5*self.hcY))
        # Around Y-Y Axis
        self.bjY = (min(self.hcY, self.bbX+0.5*self.hcX) if
                    self.hcY >= self.bbX else min(self.bbX,
                                                  self.hcY+0.5*self.hcX))

    def build(self):
        """
        Create the beam-column joint model in OpenSees.

        Builds appropriate joint model based on joint type using zero-length
        elements with hysteretic materials for moment-rotation behaviour.
        """
        if self.JT == 0:
            self._NoSeismicDetail()
        else:
            raise ValueError(f'Joint Type: {self.JT} not defined')

    def capacity(self):
        """
        Calculate joint moment capacity using EC8 and NTC building codes.

        Computes joint capacity for both directions using EC8-3 and NTC-2018
        approaches. Results stored in self.Out dictionary.

        Returns
        -------
        dict
            Joint capacity results for both directions and code methods
        """
        JCur = C.Joint(self.JT,
                       self.JD,
                       self.CS,
                       self.BXS,
                       self.BYS)
        JCur.EC8()
        JCur.NTC()
        self.Out.update(JCur.Out)
        return JCur.Out

    def buildWcapacity(self):
        """
        Build joint model and calculate capacity in one operation.

        Convenience method combining model building and capacity calculation.

        Returns
        -------
        dict
            Complete joint analysis results including OpenSees model
            information and capacity results
        """
        self.build()
        _ = self.capacity()
        return self.Out

    def _NoSeismicDetail(self):
        """
        Create joint model for non-seismic detailed construction.

        Implements joint behaviour for structures without seismic detailing
        based on calibrated parameters. Creates zero-length element with
        moment-rotation relationships for both directions and appropriate
        DOF assignments based on joint location.
        """

        if self.JD['JointLocX'] in (1, 10):
            #                  k_cr, k_pk, k_ult
            ptcX = np.asarray([0.132, 0.132, 0.053,
                               0.132, 0.132, 0.053])
            #                gamma_cr, gamma_pk, gamma_ult
            gammaX = np.asarray([0.0002, 0.0132, 0.0270,
                                 0.0002, 0.0132, 0.0270])
            hystX = np.asarray([0.6, 0.2, 0.0, 0.0, 0.3])
        elif self.JD['JointLocX'] in (2, 20):
            #                  k_cr, k_pk, k_ult
            ptcX = np.asarray([0.29, 0.42, 0.42,
                               0.29, 0.42, 0.42])
            #                gamma_cr, gamma_pk, gamma_ult
            gammaX = np.asarray([0.0002, 0.0090, 0.0200,
                                 0.0002, 0.0090, 0.0200])
            hystX = np.asarray([0.6, 0.2, 0.0, 0.01, 0.3])

        if self.JD['JointLocY'] in (1, 10):
            #                  k_cr, k_pk, k_ult
            ptcY = np.asarray([0.132, 0.132, 0.053,
                               0.132, 0.132, 0.053])
            #                gamma_cr, gamma_pk, gamma_ult
            gammaY = np.asarray([0.0002, 0.0132, 0.0270,
                                 0.0002, 0.0132, 0.0270])
            hystY = np.asarray([0.6, 0.2, 0.0, 0.0, 0.3])
        elif self.JD['JointLocY'] in (2, 20):
            #                  k_cr, k_pk, k_ult
            ptcY = np.asarray([0.29, 0.42, 0.42,
                               0.29, 0.42, 0.42])
            #                gamma_cr, gamma_pk, gamma_ult
            gammaY = np.asarray([0.0002, 0.0090, 0.0200,
                                 0.0002, 0.0090, 0.0200])
            hystY = np.asarray([0.6, 0.2, 0.0, 0.01, 0.3])

        # Create Flexural Materials
        # Around X-X
        # Exterior joint
        if self.JD['JointLocY'] != 3:
            self.jX = 0.9*(self.hbY-self.cv-self.dbVY-self.dbLY*0.5)
            pt_x = ptcY*((self.fc/units.MPa)**0.5)*units.MPa
        if self.JD['JointLocY'] == 1:
            MjX = ((pt_x*self.bjX*self.hcY) *
                   ((self.JD['Hint']*self.jX/(self.JD['Hint']-self.jX))) *
                   ((0.5*self.hbY/self.hcY) +
                    (((0.5*self.hbY/self.hcY)**2) + 1 +
                     (self.NLoad/(pt_x*self.bjX*self.hcY)))**0.5))
        # Interior joint
        elif self.JD['JointLocY'] == 2:
            MjX = ((pt_x*self.bjX*self.hcY) *
                   ((self.JD['Hint']*self.jX/(self.JD['Hint']-self.jX))) *
                   ((1+(self.NLoad/(pt_x*self.bjX*self.hcY)))**0.5))
        # Exterior Elastic joint
        elif self.JD['JointLocY'] == 10:
            MjX = (((self.JD['Hint']*self.jX/(self.JD['Hint']-self.jX)) *
                    self.bjX*self.hcY*pt_x) *
                   ((0.5*self.hbY/self.hcY) +
                    ((0.5*self.hbY/self.hcY)**2 + 1 +
                     (self.NLoad/(pt_x*self.bjX*self.hcY)))**0.5))
        # Interior Elastic joint
        elif self.JD['JointLocY'] == 20:
            MjX = (((self.JD['Hint']*self.jX/(self.JD['Hint']-self.jX)) *
                    self.bjX*self.hcY*pt_x) *
                   ((0.5*self.hbY/self.hcY) +
                    ((0.5*self.hbY/self.hcY)**2 + 1 +
                     (self.NLoad/(pt_x*self.bjX*self.hcY)))**0.5))

        # Around Y-Y
        if self.JD['JointLocX'] != 3:
            self.jY = 0.9*(self.hbX-self.cv-self.dbVX-self.dbLX*0.5)
            pt_y = ptcX*((self.fc/units.MPa)**0.5)*units.MPa
        # Exterior joint
        if self.JD['JointLocX'] == 1:
            MjY = ((pt_y*self.bjY*self.hcX) *
                   ((self.JD['Hint']*self.jY/(self.JD['Hint']-self.jY))) *
                   ((0.5*self.hbX/self.hcX) +
                    (((0.5*self.hbX/self.hcX)**2) + 1 +
                     (self.NLoad/(pt_y*self.bjY*self.hcX)))**0.5))
        # Interior joint
        elif self.JD['JointLocX'] == 2:
            MjY = ((pt_y*self.bjY*self.hcX) *
                   ((self.JD['Hint']*self.jY/(self.JD['Hint']-self.jY))) *
                   ((1+(self.NLoad/(pt_y*self.bjY*self.hcX)))**0.5))
        # Exterior Elastic joint
        elif self.JD['JointLocX'] == 10:
            MjY = (((self.JD['Hint']*self.jY/(self.JD['Hint']-self.jY)) *
                    self.bjY*self.hcX*pt_y) *
                   ((0.5*self.hbX/self.hcX) +
                    ((0.5*self.hbX/self.hcX)**2 + 1 +
                     (self.NLoad/(pt_y*self.bjY*self.hcX)))**0.5))
        # Interior Elastic joint
        elif self.JD['JointLocX'] == 20:
            MjY = (((self.JD['Hint']*self.jY/(self.JD['Hint']-self.jY)) *
                    self.bjY*self.hcX*pt_y) *
                   ((0.5*self.hbX/self.hcX) +
                    ((0.5*self.hbX/self.hcX)**2 + 1 +
                     (self.NLoad/(pt_y*self.bjY*self.hcX)))**0.5))

        rigM = int(999999)
        axM = int(float(f'2{self.JD["ID"]}'))
        flXMdummy = int(float(f'3{self.JD["ID"]}'))
        flYMdummy = int(float(f'4{self.JD["ID"]}'))
        flXM = int(float(f'5{self.JD["ID"]}'))
        flYM = int(float(f'6{self.JD["ID"]}'))

        # X-X Axis
        # Use a higher limit for now, needs to be updated later
        # !! This note from O'Reily??
        gamm_max = 0.1
        if self.JD['JointLocY'] in (1, 2):
            ops.uniaxialMaterial('Hysteretic', flXMdummy,
                                 *[MjX[0], gammaY[0]],
                                 *[MjX[1], gammaY[1]],
                                 *[MjX[2], gammaY[2]],
                                 *[-MjX[3], -gammaY[3]],
                                 *[-MjX[4], -gammaY[4]],
                                 *[-MjX[5], -gammaY[5]],
                                 hystY[0],
                                 hystY[1],
                                 hystY[2],
                                 hystY[3],
                                 hystY[4])
        elif self.JD['JointLocY'] in (10, 20):
            ops.uniaxialMaterial('Elastic', flXMdummy,
                                 MjX[0]/gammaY[0])
        if self.JD['JointLocY'] != 3:
            RotX = [list(MjX), list(gammaY)]
            ops.uniaxialMaterial('MinMax', flXM, flXMdummy,
                                 '-min', -gamm_max,
                                 '-max', gamm_max)
        else:
            RotX = [[1e8]*6, [1e8]*6]

        # Y-Y Axis
        if self.JD['JointLocX'] in (1, 2):
            ops.uniaxialMaterial('Hysteretic', flYMdummy,
                                 *[MjY[0], gammaX[0]],
                                 *[MjY[1], gammaX[1]],
                                 *[MjY[2], gammaX[2]],
                                 *[-MjY[3], -gammaX[3]],
                                 *[-MjY[4], -gammaX[4]],
                                 *[-MjY[5], -gammaX[5]],
                                 hystX[0],
                                 hystX[1],
                                 hystX[2],
                                 hystX[3],
                                 hystX[4])
        elif self.JD['JointLocX'] in (10, 20):
            ops.uniaxialMaterial('Elastic', flYMdummy,
                                 MjY[0]/gammaX[0])
        if self.JD['JointLocX'] != 3:
            RotY = [list(MjY), list(gammaX)]
            ops.uniaxialMaterial('MinMax', flYM, flYMdummy,
                                 '-min', -gamm_max,
                                 '-max', gamm_max)
        else:
            RotY = [[1e8]*6, [1e8]*6]

        # Create Axial Material
        Kspr = (2*self.Ec*self.Ac)/self.hb
        ops.uniaxialMaterial('Elastic', axM, Kspr)

        # Create Zero-Length Element
        # Create Nodes
        ops.node(int(float(f'1{self.JD["ID"]}')),
                 self.X, self.Y, self.Z,
                 '-mass', *self.NMass)
        ops.node(int(float(f'6{self.JD["ID"]}')),
                 self.X, self.Y, self.Z,
                 '-mass', *len(self.NMass)*[0.0])

        ET = int(float(f'9{self.JD["ID"]}'))

        if self.JD['JointLocX'] != 3 and self.JD['JointLocY'] != 3:
            ops.element('zeroLength', ET,
                        *[int(float(f'1{self.JD["ID"]}')),
                          int(float(f'6{self.JD["ID"]}'))],
                        '-mat', *[rigM, rigM, axM, flXM, flYM, rigM],
                        '-dir', *[1, 2, 3, 4, 5, 6],
                        '-doRayleigh', 1)
        elif self.JD['JointLocX'] == 3:
            ops.element('zeroLength', ET,
                        *[int(float(f'1{self.JD["ID"]}')),
                          int(float(f'6{self.JD["ID"]}'))],
                        '-mat', *[rigM, rigM, axM, flXM, rigM, rigM],
                        '-dir', *[1, 2, 3, 4, 5, 6],
                        '-doRayleigh', 1)
        elif self.JD['JointLocY'] == 3:
            ops.element('zeroLength', ET,
                        *[int(float(f'1{self.JD["ID"]}')),
                          int(float(f'6{self.JD["ID"]}'))],
                        '-mat', *[rigM, rigM, axM, rigM, flYM, rigM],
                        '-dir', *[1, 2, 3, 4, 5, 6],
                        '-doRayleigh', 1)
        if self.JD['ID'] == 138:
            print('here')

        # NTC X [Out3], NTC Y [Out4],
        # EC8 X [Out5], EC8 Y [Out6]
        self.Out['ElementTag'] = ET
        self.Out['Rot_Around_X'] = RotX
        self.Out['Rot_Around_Y'] = RotY
