"""
Structural capacity analysis module.

Calculates structural capacity of reinforced concrete joints and frame elements
according to European (EC8) and Italian (NTC) building codes.

Classes:
    Joint: Calculates moment capacity of beam-column joints
    Frame: Calculates flexural and shear capacity of frame elements
"""

from cad2sees.helpers import units
import numpy as np


class Joint:
    """
    Calculates moment capacity of reinforced concrete beam-column joints.

    Analyses joint behaviour according to EC8 and NTC building codes.

    Attributes:
        JT: Joint type identifier
        JD: Joint data containing location and load information
        CS: Column section properties
        BXS: Beam sections in X-direction
        BYS: Beam sections in Y-direction
        Out: Output containing calculated capacities

    Methods:
        EC8(): Calculate joint capacity per EC8 code
        NTC(): Calculate joint capacity per NTC-2018 code
    """
    def __init__(self,
                 JointType,
                 JointData,
                 ColumnSection,
                 BeamXSections,
                 BeamYSections):
        self.JT = JointType
        self.JD = JointData
        self.CS = ColumnSection
        self.BXS = BeamXSections
        self.BYS = BeamYSections
        self.Out = {}

        self.fc = self.CS['fc0']
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
        self.jX = 0.9*(self.hbY-self.cv-self.dbVY-self.dbLY*0.5)
        self.bjX = (min(self.hcX, self.bbY+0.5*self.hcY) if
                    self.hcX >= self.bbY else min(self.bbY,
                                                  self.hcX+0.5*self.hcY))
        # Around Y-Y Axis
        self.jY = 0.9*(self.hbX-self.cv-self.dbVX-self.dbLX*0.5)
        self.bjY = (min(self.hcY, self.bbX+0.5*self.hcX) if
                    self.hcY >= self.bbX else min(self.bbX,
                                                  self.hcY+0.5*self.hcX))

    def EC8(self):
        """
        Calculate joint moment capacity per EC8-3 (2005).

        Computes moment capacity considering joint shear stress limits,
        normalised axial load ratio effects, and joint geometry.

        Updates:
            self.Out['MomentCapacityX_EC8']: X-direction moment capacity
            self.Out['MomentCapacityY_EC8']: Y-direction moment capacity
        """
        # EC8
        Eta = 0.6*(1-self.fc/(250*units.MPa))
        vd = self.JD['NLoad']/(self.hcX*self.hcY*self.fc)
        Tao_jh_t_extEC8 = 0.8*Eta*self.fc*(1-(vd/Eta))**0.5
        Tao_jh_t_intEC8 = Eta*self.fc*(1-(vd/Eta))**0.5
        Tao_jh_c_extEC8 = 0.8*Eta*self.fc*(1-(vd/Eta))**0.5
        Tao_jh_c_intEC8 = Eta*self.fc*(1-(vd/Eta))**0.5
        # Around X-X
        if self.JD['JointLocY'] != 3:
            if self.JD['JointLocY'] in (1, 10):  # External
                # EC8
                MjXtEC8 = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jh_t_extEC8))
                MjXcEC8 = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jh_c_extEC8))
            elif self.JD['JointLocY'] in (2, 20):  # Internal
                # EC8
                MjXtEC8 = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jh_t_intEC8))
                MjXcEC8 = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jh_c_intEC8))
            MomentCapXEC8 = [MjXtEC8, MjXcEC8]
        else:
            # If Does not exist put very high value
            MomentCapXEC8 = [1e8, 1e8]

        # Around Y-Y
        if self.JD['JointLocX'] != 3:
            if self.JD['JointLocX'] in (1, 10):  # External
                # EC8
                MjYtEC8 = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jh_t_extEC8))
                MjYcEC8 = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jh_c_extEC8))
            elif self.JD['JointLocX'] in (2, 20):
                # EC8
                MjYtEC8 = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jh_t_intEC8))
                MjYcEC8 = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jh_c_intEC8))
            MomentCapYEC8 = [MjYtEC8, MjYcEC8]
        else:
            MomentCapYEC8 = [1e8, 1e8]
        self.Out['MomentCapacityX_EC8'] = MomentCapXEC8
        self.Out['MomentCapacityY_EC8'] = MomentCapYEC8

    def NTC(self):
        """
        Calculate joint moment capacity per NTC-2018 (Italian code).

        Computes moment capacity using tensile strength limits and
        beam-to-column width ratio effects.

        Updates:
            self.Out['MomentCapacityX_NTC']: X-direction moment capacity
            self.Out['MomentCapacityY_NTC']: Y-direction moment capacity
        """
        # NTC-2018
        sigt_t = (0.3*(self.fc/units.MPa)**0.5)*units.MPa
        sigt_c = (0.5*(self.fc/units.MPa)**0.5)*units.MPa

        # Around X-X
        if self.JD['JointLocY'] != 3:
            if self.JD['JointLocY'] in (1, 10):  # External
                # NTC-2018
                Tao_jhY_t_NTC = (sigt_t *
                                 ((0.5*self.hbX/self.hcX) +
                                  ((0.5*self.hbX/self.hcX)**2 +
                                   (self.JD['NLoad'] /
                                   (sigt_t*self.bjX*self.hcY))+1)**0.5))
                Tao_jhY_c_NTC = (sigt_c *
                                 ((0.5*self.hbX/self.hcX) +
                                  ((0.5*self.hbX/self.hcX)**2 +
                                   (self.JD['NLoad'] /
                                   (sigt_c*self.bjX*self.hcY))+1)**0.5))

                MjXtNTC = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jhY_t_NTC))
                MjXcNTC = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jhY_c_NTC))
            elif self.JD['JointLocY'] in (2, 20):  # Internal
                # NTC-2018
                Tao_jhY_t_NTC = sigt_t*((self.JD['NLoad'] /
                                        (sigt_t*self.bjX*self.hcY))+1)**0.5
                Tao_jhY_c_NTC = sigt_c*((self.JD['NLoad'] /
                                        (sigt_c*self.bjX*self.hcY))+1)**0.5
                MjXtNTC = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jhY_t_NTC))
                MjXcNTC = (self.bjX*self.hcY *
                           (self.JD['Hint']*self.jX /
                            (self.JD['Hint']-self.jX)) *
                           (Tao_jhY_c_NTC))
            MomentCapXNTC = [MjXtNTC, MjXcNTC]
        else:
            # If Does not exist put very high value
            MomentCapXNTC = [1e8, 1e8]
        # Around Y-Y
        if self.JD['JointLocX'] != 3:
            if self.JD['JointLocX'] in (1, 10):  # External
                # NTC-2018
                Tao_jhX_t_NTC = (sigt_t *
                                 ((0.5*self.hbY/self.hcY) +
                                  ((0.5*self.hbY/self.hcY)**2 +
                                   (self.JD['NLoad'] /
                                   (sigt_t*self.bjY*self.hcX))+1)**0.5))
                Tao_jhX_c_NTC = (sigt_c *
                                 ((0.5*self.hbY/self.hcY) +
                                  ((0.5*self.hbY/self.hcY)**2 +
                                   (self.JD['NLoad'] /
                                   (sigt_c*self.bjY*self.hcX))+1)**0.5))

                MjYtNTC = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jhX_t_NTC))
                MjYcNTC = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jhX_c_NTC))
            elif self.JD['JointLocX'] in (2, 20):
                # NTC-2018
                Tao_jhX_t_NTC = sigt_t*((self.JD['NLoad'] /
                                        (sigt_t*self.bjY*self.hcX))+1)**0.5
                Tao_jhX_c_NTC = sigt_c*((self.JD['NLoad'] /
                                        (sigt_c*self.bjY*self.hcX))+1)**0.5
                MjYtNTC = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jhX_t_NTC))
                MjYcNTC = (self.bjY*self.hcX *
                           (self.JD['Hint']*self.jY /
                            (self.JD['Hint']-self.jY)) *
                           (Tao_jhX_c_NTC))
            MomentCapYNTC = [MjYtNTC, MjYcNTC]
        else:
            MomentCapYNTC = [1e8, 1e8]
        self.Out['MomentCapacityX_NTC'] = MomentCapXNTC
        self.Out['MomentCapacityY_NTC'] = MomentCapYNTC


class Frame:
    """
    Calculates flexural and shear capacity of reinforced concrete frames.

    Performs capacity calculations for RC frame elements (beams/columns)
    per EC8-3 and Priestley approaches.

    Attributes:
        ISP: I-end section properties
        JSP: J-end section properties
        IND: I-node data (coordinates, loads)
        JND: J-node data (coordinates, loads)
        FlexuralOut: Flexural capacity results
        ShearOut: Shear capacity results

    Methods:
        Flexural(): Calculate flexural capacity and rotation demands
        Shear(): Calculate shear capacity using multiple approaches
    """
    def __init__(self,
                 ISectionProps,
                 JSectionProps,
                 INodeData,
                 JNodeData):
        self.ISP = ISectionProps
        self.JSP = JSectionProps
        self.IND = INodeData
        self.JND = JNodeData

        # Material Properties
        self.fc = self.ISP['fc0']*units.MPa
        self.fy = self.ISP['fy']*units.MPa
        self.fyw = self.ISP['fyw']*units.MPa
        self.Es = self.ISP['Es']*units.MPa
        self.epsyL = self.fy/self.Es

        Xi = INodeData['Coordinates'][0]*units.cm
        Yi = INodeData['Coordinates'][1]*units.cm
        Zi = INodeData['Coordinates'][2]*units.cm

        Xj = JNodeData['Coordinates'][0]*units.cm
        Yj = JNodeData['Coordinates'][1]*units.cm
        Zj = JNodeData['Coordinates'][2]*units.cm

        self.Ls = (((Xi-Xj)**2 + (Yi-Yj)**2 + (Zi-Zj)**2)**0.5)*0.5

        # Section properties for I-end
        self.bi = self.ISP['b']*units.mm
        self.hi = self.ISP['h']*units.mm
        self.si = self.ISP['s']*units.mm
        self.cvi = self.ISP['Cover']*units.mm
        self.dbli = np.mean(self.ISP['ReinfL'][:, -1]**2)**0.5*units.mm
        self.dbwi = self.ISP['phi_T']*units.mm
        AsLi = np.sum((self.ISP['ReinfL'][:, -1]*units.mm)**2)*0.25*units.pi
        self.rho_Li = AsLi/(self.bi*self.hi)
        numYdiri = self.ISP['NumofStrYDir']
        numZdiri = self.ISP['NumofStrZDir']

        TopSecReinfLMap = self.ISP['ReinfL'][:, 1] >= self.ISP['h']/6
        self.rhoTopi = (0.25*units.pi *
                        np.sum((self.ISP['ReinfL'][TopSecReinfLMap, -1] *
                                units.mm)**2)/(self.bi*self.hi))

        BotSecReinfLMap = self.ISP['ReinfL'][:, 1] <= -self.ISP['h']/6
        self.rhoBoti = (0.25*units.pi *
                        np.sum((self.ISP['ReinfL'][BotSecReinfLMap, -1] *
                                units.mm)**2)/(self.bi*self.hi))

        LeftSecReinfLMap = self.ISP['ReinfL'][:, 0] <= -self.ISP['b']/6
        self.rhoLefti = (0.25*units.pi *
                         np.sum((self.ISP['ReinfL'][LeftSecReinfLMap, -1] *
                                 units.mm)**2)/(self.bi*self.hi))

        RightSecReinfLMap = self.ISP['ReinfL'][:, 0] >= self.ISP['b']/6
        self.rhoRighti = (0.25*units.pi *
                          np.sum((self.ISP['ReinfL'][RightSecReinfLMap, -1] *
                                  units.mm)**2)/(self.bi*self.hi))

        botBars = self.ISP['ReinfL'][BotSecReinfLMap, :]
        topBars = self.ISP['ReinfL'][TopSecReinfLMap, :]
        self.BarsTopBoti = [botBars, topBars]
        leftBars = self.ISP['ReinfL'][LeftSecReinfLMap, :]
        rightBars = self.ISP['ReinfL'][RightSecReinfLMap, :]
        self.BarsRightLefti = [rightBars, leftBars]

        AsvYI = numYdiri*0.25*units.pi*(self.dbwi**2)
        AsvZI = numZdiri*0.25*units.pi*(self.dbwi**2)
        self.rho_syi = AsvYI/(self.si*self.bi)
        self.rho_szi = AsvZI/(self.si*self.hi)

        # Section properties for J-end
        self.bj = self.JSP['b']*units.mm
        self.hj = self.JSP['h']*units.mm
        self.sj = self.JSP['s']*units.mm
        self.cvj = self.JSP['Cover']*units.mm

        AsLj = np.sum((self.JSP['ReinfL'][:, -1]*units.mm)**2)*0.25*units.pi
        self.rho_Lj = AsLj/(self.bj*self.hj)

        self.dblj = np.mean(self.JSP['ReinfL'][:, -1]**2)**0.5*units.mm
        self.dbwj = self.JSP['phi_T']*units.mm

        numYdirj = self.JSP['NumofStrYDir']
        numZdirj = self.JSP['NumofStrZDir']

        TopSecReinfLMap = self.JSP['ReinfL'][:, 1] >= self.JSP['h']/6
        self.rhoTopj = (0.25*units.pi *
                        np.sum((self.JSP['ReinfL'][TopSecReinfLMap, -1] *
                                units.mm)**2)/(self.bj*self.hj))

        BotSecReinfLMap = self.JSP['ReinfL'][:, 1] <= -self.JSP['h']/6
        self.rhoBotj = (0.25*units.pi *
                        np.sum((self.JSP['ReinfL'][BotSecReinfLMap, -1] *
                                units.mm)**2)/(self.bj*self.hj))

        LeftSecReinfLMap = self.JSP['ReinfL'][:, 0] <= -self.JSP['b']/6
        self.rhoLeftj = (0.25*units.pi *
                         np.sum((self.JSP['ReinfL'][LeftSecReinfLMap, -1] *
                                 units.mm)**2)/(self.bj*self.hj))

        RightSecReinfLMap = self.JSP['ReinfL'][:, 0] >= self.JSP['b']/6
        self.rhoRightj = (0.25*units.pi *
                          np.sum((self.JSP['ReinfL'][RightSecReinfLMap, -1] *
                                  units.mm)**2)/(self.bj*self.hj))

        botBars = self.JSP['ReinfL'][BotSecReinfLMap, :]
        topBars = self.JSP['ReinfL'][TopSecReinfLMap, :]
        self.BarsTopBotj = [topBars, botBars]
        leftBars = self.JSP['ReinfL'][LeftSecReinfLMap, :]
        rightBars = self.JSP['ReinfL'][RightSecReinfLMap, :]
        self.BarsRightLeftj = [rightBars, leftBars]

        AsvYJ = numYdirj*0.25*units.pi*(self.dbwj**2)
        AsvZJ = numZdirj*0.25*units.pi*(self.dbwj**2)
        self.rho_syj = AsvYJ/(self.sj*self.bj)
        self.rho_szj = AsvZJ/(self.sj*self.hj)
        # Normalised axial load ratio (-)
        self.nui = abs(INodeData['NLoad'])*units.kN/(self.bi*self.hi*self.fc)
        self.nuj = abs(JNodeData['NLoad'])*units.kN/(self.bj*self.hj*self.fc)

    def Flexural(self):
        """
        Calculate flexural capacity and rotation demands for both ends.

        Performs flexural analysis using EC8-3 provisions for ultimate,
        yield, and plastic rotation capacities in both local axes.

        Returns:
            dict: Flexural capacity results for I-end and J-end:
                - 'Izz': I-end strong axis results
                - 'Iyy': I-end weak axis results
                - 'Jzz': J-end strong axis results
                - 'Jyy': J-end weak axis results
        """
        phiYIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][1])

        phiUIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][-1])

        MYIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]['MIdealPZ'][1])
        MYIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]['MIdealNZ'][1])
        OutIzz = self._EC8_flexural2(self.bi,
                                     self.hi,
                                     self.dbwi,
                                     self.dbli,
                                     self.cvi,
                                     self.BarsTopBoti,
                                     self.rhoTopi,
                                     self.rhoBoti,
                                     self.nui,
                                     phiYp=phiYIzzP,
                                     phiYn=phiYIzzN,
                                     Myp=MYIzzP,
                                     Myn=MYIzzN,
                                     phiUp=phiUIzzP,
                                     phiUn=phiUIzzN)

        phiYIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPY'][1])
        phiYIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNY'][1])
        phiUIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPY'][-1])
        phiUIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNY'][-1])
        MYIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealPY'][1])
        MYIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealNY'][1])

        IyyRigthLeftBars = []
        for Bars in self.BarsRightLefti:
            Bars0 = Bars[:, 0].copy()
            Bars1 = Bars[:, 1].copy()
            Bars[:, 0] = Bars1
            Bars[:, 1] = Bars0
            IyyRigthLeftBars.append(Bars)

        OutIyy = self._EC8_flexural2(self.hi,
                                     self.bi,
                                     self.dbwi,
                                     self.dbli,
                                     self.cvi,
                                     IyyRigthLeftBars,
                                     self.rhoRighti,
                                     self.rhoLefti,
                                     self.nui,
                                     phiYp=phiYIyyP,
                                     phiYn=phiYIyyN,
                                     Myp=MYIyyP,
                                     Myn=MYIyyN,
                                     phiUp=phiUIyyP,
                                     phiUn=phiUIyyN)

        phiYJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][1])
        phiUJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][-1])
        MYJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealPZ'][1])
        MYJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealNZ'][1])
        OutJzz = self._EC8_flexural2(self.bj,
                                     self.hj,
                                     self.dbwj,
                                     self.dblj,
                                     self.cvj,
                                     self.BarsTopBotj,
                                     self.rhoTopj,
                                     self.rhoBotj,
                                     self.nuj,
                                     phiYp=phiYJzzP,
                                     phiYn=phiYJzzN,
                                     Myp=MYJzzP,
                                     Myn=MYJzzN,
                                     phiUp=phiUJzzP,
                                     phiUn=phiUJzzN)

        phiYJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPY'][1])
        phiYJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNY'][1])
        phiUJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPY'][-1])
        phiUJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNY'][-1])
        MYJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealPY'][1])
        MYJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealNY'][1])

        JyyRigthLeftBars = []
        for Bars in self.BarsRightLeftj:
            Bars0 = Bars[:, 0].copy()
            Bars1 = Bars[:, 1].copy()
            Bars[:, 0] = Bars1
            Bars[:, 1] = Bars0
            JyyRigthLeftBars.append(Bars)

        OutJyy = self._EC8_flexural2(self.hj,
                                     self.bj,
                                     self.dbwj,
                                     self.dblj,
                                     self.cvj,
                                     JyyRigthLeftBars,
                                     self.rhoRightj,
                                     self.rhoLeftj,
                                     self.nuj,
                                     phiYp=phiYJyyP,
                                     phiYn=phiYJyyN,
                                     Myp=MYJyyP,
                                     Myn=MYJyyN,
                                     phiUp=phiUJyyP,
                                     phiUn=phiUJyyN)

        self.FlexuralOut = {'Izz': OutIzz,
                            'Iyy': OutIyy,
                            'Jzz': OutJzz,
                            'Jyy': OutJyy}
        return self.FlexuralOut

    def _EC8_flexural2(self,
                       b, h, dbw, dbl, cv,
                       BarsTopBot,
                       rho1, rho3,
                       nu,
                       phiYp=0, phiYn=0,
                       Myp=0, Myn=0,
                       phiUp=0, phiUn=0):
        """
        Calculate flexural capacity per EC8-3 Annex A.

        Updated EC8 flexural capacity calculation using provided
        curvature and moment values from moment-curvature analysis.

        Args:
            b: Section width (mm)
            h: Section height (mm)
            dbw: Transverse reinforcement diameter (mm)
            dbl: Longitudinal reinforcement diameter (mm)
            cv: Concrete cover (mm)
            BarsTopBot: Top and bottom reinforcement coordinates
            rho1: Bottom reinforcement ratio
            rho3: Top reinforcement ratio
            nu: Normalised axial load ratio
            phiYp: Positive yield curvature (optional)
            phiYn: Negative yield curvature (optional)
            Myp: Positive yield moment (optional)
            Myn: Negative yield moment (optional)
            phiUp: Positive ultimate curvature (optional)
            phiUn: Negative ultimate curvature (optional)

        Returns:
            dict: Rotation capacities including ultimate and plastic rotations
        """

        # EC8-3 -2005
        # Annex A - A-1
        gamma_el = 1.5
        # Geometrical Props

        b0 = b - 2*cv-2*dbw-dbl
        h0 = h - 2*cv-2*dbw-dbl
        z = 0.9*(h-cv-dbw-0.5*dbl)

        if phiYp == 0:
            phiYp = 2.1*self.epsyL/h
        if phiYn == 0:
            phiYn = 2.1*self.epsyL/h

        gamma_c = 1.2
        CRd_c = 0.18/gamma_c
        k = min(2, 1+(0.2/h0)**0.5)
        fck = (self.fc/units.MPa)-8
        k1 = 0.15
        sigma_cp = min(0.2*self.fc/units.MPa, nu)
        VR_cp = ((CRd_c*k*(100*rho3*fck)**0.33)+k1*sigma_cp)*b0*h0*1000
        VR_cn = ((CRd_c*k*(100*rho1*fck)**0.33)+k1*sigma_cp)*b0*h0*1000

        if Myp < VR_cp*self.Ls:
            avp = 0
        else:
            avp = 1

        if Myn < VR_cn*self.Ls:
            avn = 0
        else:
            avn = 1

        thetaYTerm1p = (self.Ls+avp*z)/3
        thetaYTerm1n = (self.Ls+avn*z)/3
        thetaYTerm2 = 0.0014*(1+1.5*h/self.Ls)

        dblp = np.mean(BarsTopBot[1][:, -1]**2)**0.5
        dbln = np.mean(BarsTopBot[0][:, -1]**2)**0.5

        thetaYTerm3p = ((phiYp*dblp*units.mm*(self.fy/units.MPa)) /
                        (8*((self.fc/units.MPa)**0.5)))

        thetaYTerm3n = ((phiYn*dbln*units.mm*(self.fy/units.MPa)) /
                        (8*((self.fc/units.MPa)**0.5)))

        thetaYp = phiYp*thetaYTerm1p + thetaYTerm2 + thetaYTerm3p
        thetaYn = phiYn*thetaYTerm1n + thetaYTerm2 + thetaYTerm3n

        Lp = 0.08*self.Ls + 0.022*dbl*self.ISP['fy']

        thetaUmpos = (thetaYp + (phiUp-phiYp)*Lp*(1-0.5*Lp/self.Ls))/gamma_el
        thetaUmneg = (thetaYn + (phiUn-phiYn)*Lp*(1-0.5*Lp/self.Ls))/gamma_el

        thetaUmpospl = thetaUmpos - thetaYp
        thetaUmnegpl = thetaUmneg - thetaYn
        if thetaUmpospl < 0:
            print('thetaUmpospl is negative')
        if thetaUmnegpl < 0:
            print('thetaUmnegpl is negative')

        Out = {'thetaUmP': float(thetaUmpos),
               'thetaUmN': float(thetaUmneg),
               'thetaUmPpl': float(thetaUmpospl),
               'thetaUmNpl': float(thetaUmnegpl),
               'thetaYp': float(thetaYp),
               'thetaYn': float(thetaYn)}
        return Out

    def _EC8_flexural(self,
                      b, h, s, dbw, dbl, cv,
                      BarsTopBot, BarsRightLeft,
                      rho1, rho3, rhos, rhod,
                      nu,
                      phiYp=0, phiYn=0,
                      Myp=0, Myn=0,
                      phiUp=0, phiUn=0):

        # EC8-3 -2005
        # Annex A - A-1
        gamma_el = 1.5
        # Geometrical Props

        b0 = b - 2*cv-2*dbw-dbl
        h0 = h - 2*cv-2*dbw-dbl
        z = 0.9*(h-cv-dbw-0.5*dbl)

        if phiYp == 0:
            phiYp = 2.1*self.epsyL/h

        if phiYn == 0:
            phiYn = 2.1*self.epsyL/h

        gamma_c = 1.2
        CRd_c = 0.18/gamma_c
        k = min(2, 1+(0.2/h0)**0.5)
        fck = (self.fc/units.MPa)-8
        k1 = 0.15
        sigma_cp = min(0.2*self.fc/units.MPa, nu)
        VR_c = ((CRd_c*k*(100*rho3*fck)**0.33)+k1*sigma_cp)*b0*h0*1000

        if Myp < VR_c*self.Ls:
            avp = 0
        else:
            avp = 1

        if Myn < VR_c*self.Ls:
            avn = 0
        else:
            avn = 1

        thetaYTerm1p = (self.Ls+avp*z)/3
        thetaYTerm1n = (self.Ls+avn*z)/3
        thetaYTerm2 = 0.0014*(1+1.5*h/self.Ls)

        dblp = np.mean(BarsTopBot[1][:, -1]**2)**0.5
        dbln = np.mean(BarsTopBot[0][:, -1]**2)**0.5

        thetaYTerm3p = ((phiYp*dblp*(self.fy/units.MPa)) /
                        (8*((self.fc/units.MPa)**0.5)))

        thetaYTerm3n = ((phiYn*dbln*(self.fy/units.MPa)) /
                        (8*((self.fc/units.MPa)**0.5)))

        thetaYp = phiYp*thetaYTerm1p + thetaYTerm2 + thetaYTerm3p
        thetaYn = phiYn*thetaYTerm1n + thetaYTerm2 + thetaYTerm3n

        Aineff = 0
        for Bars in BarsRightLeft:
            Bars = Bars[Bars[:, 0].argsort()]
            for i in range(len(Bars)-1):
                z0 = Bars[i, 0]
                z1 = Bars[i+1, 0]
                wi = z1-z0
                Aineff += (wi**2)/6

        for Bars in BarsTopBot:
            Bars = Bars[Bars[:, 1].argsort()]
            for i in range(len(Bars)-1):
                y0 = Bars[i, 1]
                y1 = Bars[i+1, 1]
                wi = y1-y0
                Aineff += (wi**2)/6

        alpha = ((1-min(0.5*s/b0, 1)) *
                 (1-min(0.5*s/h0, 1)) *
                 (1-((Aineff*(units.mm**2))/(h0*b0))))

        # ThetaUm
        thetaUm1 = 0.016*(0.3**nu)/gamma_el
        thetaUm1pl = 0.0145*(0.25**nu)/gamma_el

        thetaUm2pos = (max(0.01, rho1)*(self.fc/units.MPa) /
                       max(0.01, rho3))**0.225
        thetaUm2neg = (max(0.01, rho3)*(self.fc/units.MPa) /
                       max(0.01, rho1))**0.225

        thetaUm2pospl = (((max(0.01, rho1)/max(0.01, rho3))**0.3) *
                         ((self.fc/units.MPa)**0.2))
        thetaUm2negpl = (((max(0.01, rho3)/max(0.01, rho1))**0.3) *
                         ((self.fc/units.MPa)**0.2))

        thetaUm3 = (min(9, self.Ls/h))**0.35

        thetaUm4 = 25**(alpha*rhos*self.fyw/self.fc)
        thetaUm5 = 1.25**(100*rhod)

        thetaUmpos = thetaUm1*thetaUm2pos*thetaUm3*thetaUm4*thetaUm5
        thetaUmneg = thetaUm1*thetaUm2neg*thetaUm3*thetaUm4*thetaUm5

        thetaUmpospl = thetaUm1pl*thetaUm2pospl*thetaUm3*thetaUm4*thetaUm5
        thetaUmnegpl = thetaUm1pl*thetaUm2negpl*thetaUm3*thetaUm4*thetaUm5

        Out = {'thetaUmP': float(thetaUmpos),
               'thetaUmN': float(thetaUmneg),
               'thetaUmPpl': float(thetaUmpospl),
               'thetaUmNpl': float(thetaUmnegpl),
               'thetaYp': float(thetaYp),
               'thetaYn': float(thetaYn)}
        return Out

    def Shear(self):
        """
        Calculate shear capacity using multiple approaches.

        Computes shear capacity using EC8-3 provisions and Priestley
        approach for both yielded and unyielded conditions.

        Returns:
            dict: Shear capacity results:
                - 'EC8_Shear': EC8-based shear capacity
                - 'Priestley_Shear_Unyeild': Priestley (unyielded)
                - 'Priestley_Shear_Yeild': Priestley (yielded)
        """
        phiYIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][1])
        phiUIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][-1])
        MYIzzP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealPZ'][1])
        MYIzzN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealNZ'][1])

        RotationOutsIzz = [[[phiYIzzP, MYIzzP], [phiYIzzN, MYIzzN]],
                           [phiUIzzP, phiUIzzN]]

        OutIzz = self._EC8_Shear(self.hi, self.bi,
                                 self.si, self.cvi,
                                 self.dbwi, self.dbli,
                                 self.BarsRightLefti,
                                 self.BarsTopBoti,
                                 self.rhoRighti,
                                 self.rhoLefti,
                                 self.rho_szi,
                                 0, self.rho_Li,
                                 self.nui,
                                 RotationOutsIzz)

        phiYIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][1])
        phiUIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                       ['CurvIdealNZ'][-1])
        MYIyyP = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealPZ'][1])
        MYIyyN = abs(self.ISP['MCOut'][str(self.IND['NLoad'])]
                     ['MIdealNZ'][1])

        RotationOutsIyy = [[[phiYIyyP, MYIyyP],
                            [phiYIyyN, MYIyyN]],
                           [phiUIyyP, phiUIyyN]]

        OutIyy = self._EC8_Shear(self.bi, self.hi,
                                 self.si, self.cvi,
                                 self.dbwi, self.dbli,
                                 self.BarsTopBoti,
                                 self.BarsRightLefti,
                                 self.rhoBoti,
                                 self.rhoTopi,
                                 self.rho_syi,
                                 0, self.rho_Li,
                                 self.nui,
                                 RotationOutsIyy)

        phiYJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][1])
        phiUJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][-1])
        MYJzzP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealPZ'][1])
        MYJzzN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealNZ'][1])

        RotationOutsJzz = [[[phiYJzzP, MYJzzP], [phiYJzzN, MYJzzN]],
                           [phiUJzzP, phiUJzzN]]

        OutJzz = self._EC8_Shear(self.hj, self.bj,
                                 self.sj, self.cvj,
                                 self.dbwj, self.dblj,
                                 self.BarsRightLeftj,
                                 self.BarsTopBotj,
                                 self.rhoRightj,
                                 self.rhoLeftj,
                                 self.rho_szj,
                                 0, self.rho_Lj,
                                 self.nuj,
                                 RotationOutsJzz)

        phiYJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][1])
        phiYJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][1])
        phiUJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealPZ'][-1])
        phiUJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                       ['CurvIdealNZ'][-1])
        MYJyyP = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealPZ'][1])
        MYJyyN = abs(self.JSP['MCOut'][str(self.JND['NLoad'])]
                     ['MIdealNZ'][1])

        RotationOutsJyy = [[[phiYJyyP, MYJyyP], [phiYJyyN, MYJyyN]],
                           [phiUJyyP, phiUJyyN]]

        OutJyy = self._EC8_Shear(self.bj, self.hj,
                                 self.sj, self.cvj,
                                 self.dbwj, self.dblj,
                                 self.BarsTopBotj,
                                 self.BarsRightLeftj,
                                 self.rhoBotj,
                                 self.rhoTopj,
                                 self.rho_syj,
                                 0, self.rho_Lj,
                                 self.nuj,
                                 RotationOutsJyy)

        EC8_Shear = {'Izz': OutIzz,
                     'Iyy': OutIyy,
                     'Jzz': OutJzz,
                     'Jyy': OutJyy}

        Priestley_Shear_Unyeild = {}
        Priestley_Shear_Yeild = {}

        Cases = [('Izz', (self.hi, self.bi, self.si, self.cvi),
                  abs(self.IND['NLoad']), self.rho_szi),
                 ('Iyy', (self.bi, self.hi, self.si, self.cvi),
                  abs(self.IND['NLoad']), self.rho_syi),
                 ('Jzz', (self.hj, self.bj, self.sj, self.cvj),
                  abs(self.JND['NLoad']), self.rho_szj),
                 ('Jyy', (self.bj, self.hj, self.sj, self.cvj),
                  abs(self.JND['NLoad']), self.rho_syj)]

        for label, dims, load, rho in Cases:
            Priestley_Shear_Unyeild[label] = self._Priestley_Shear(*dims, load,
                                                                   rho, 0.29)
            Priestley_Shear_Yeild[label] = self._Priestley_Shear(*dims, load,
                                                                 rho, 0.1)

        self.ShearOut = {'EC8_Shear': EC8_Shear,
                         'Priestley_Shear_Unyeild': Priestley_Shear_Unyeild,
                         'Priestley_Shear_Yeild': Priestley_Shear_Yeild}
        return self.ShearOut

    def _Priestley_Shear(self,
                         b, h, s, cv,
                         P, rho_s,
                         k):
        """
        Calculate shear capacity using Priestley approach.

        Computes shear capacity from three components: axial load,
        concrete contribution, and steel reinforcement.

        Args:
            b: Section width (mm)
            h: Section height (mm)
            s: Transverse reinforcement spacing (mm)
            cv: Concrete cover (mm)
            P: Axial load (kN)
            rho_s: Transverse reinforcement ratio
            k: Concrete contribution factor (0.29 unyielded, 0.1 yielded)

        Returns:
            float: Total shear capacity (N)
        """
        # Arrange the Units for dimensions
        b /= units.mm
        h /= units.mm
        s /= units.mm
        cv /= units.mm
        # Axial Load Component
        Vp = P*np.tan(0.5*(h*units.mm)/self.Ls)
        # Concrete Shear
        Vc = k * np.sqrt(self.fc/units.MPa) * \
            b * h * 0.8 * units.N
        # Steel Rebar
        theta = 45
        Av = rho_s * b * s
        Vs = ((Av*(self.fy/units.MPa)*(h-2*cv)/s) /
              np.tan(np.deg2rad(theta)) * units.N)
        Vr = Vp + Vc + Vs
        return float(Vr)

    def _EC8_Shear(self,
                   b, h, s, cv,
                   dbw, dbl,
                   BarsTopBot, BarsRightLeft,
                   rhoTop, rhoBot, rho_s, rho_d, rho_L,
                   nu, RotationOuts):
        """
        Calculate shear capacity per EC8-3 provisions.

        Implements EC8-3 Section A3.3.1 considering axial load effects,
        ductility demand, and reinforcement contributions.

        Args:
            b: Section width (mm)
            h: Section height (mm)
            s: Transverse reinforcement spacing (mm)
            cv: Concrete cover (mm)
            dbw: Transverse bar diameter (mm)
            dbl: Longitudinal bar diameter (mm)
            BarsTopBot: Top/bottom bar coordinates
            BarsRightLeft: Right/left bar coordinates
            rhoTop: Top reinforcement ratio
            rhoBot: Bottom reinforcement ratio
            rho_s: Transverse reinforcement ratio
            rho_d: Diagonal reinforcement ratio
            rho_L: Longitudinal reinforcement ratio
            nu: Normalised axial load ratio
            RotationOuts: Rotation outputs from flexural analysis

        Returns:
            float: Shear capacity per EC8-3 (limited by compression)
        """

        # EC8-P3 A3.3.1
        # Important Units are MN and m
        gamma_el = 1.15
        bw = b-0.5*dbl-2*(cv+dbw)
        hw = h-0.5*dbl-2*(cv+dbw)

        z = 0.9*(h-0.5*dbl-cv-dbw)

        x = h*min((0.25+0.85*nu), 1)
        Ac = bw*hw
        Acg = b*h
        N = nu*Acg*self.fc

        flexOut = self._EC8_flexural2(b, h,
                                      dbw, dbl,
                                      cv,
                                      BarsTopBot,
                                      rhoTop, rhoBot,
                                      nu,
                                      phiYp=RotationOuts[0][0][0],
                                      Myp=RotationOuts[0][0][1],
                                      phiYn=RotationOuts[0][1][0],
                                      Myn=RotationOuts[0][1][0],
                                      phiUp=RotationOuts[1][0],
                                      phiUn=RotationOuts[1][1])

        muPlP = (flexOut['thetaUmP']/flexOut['thetaYp'])-1
        muPlN = (flexOut['thetaUmN']/flexOut['thetaYn'])-1

        muPl = min(muPlN, muPlP)
        # Trans bar ratio
        Vw = rho_s*b*self.fyw*z*0.001

        Term1 = (min(N/units.MN, 0.55*Ac*self.fc/units.MPa))*(h-x)/(2*self.Ls)
        Term21 = 1-0.05*min(5, muPl)
        Term221 = 0.16*max(0.5, 100*rho_L)
        Term222 = 1-0.16*min(5, self.Ls/h)
        Term223 = (b*z*(self.fc/units.MPa)**0.5)
        Term224 = Vw
        Term22 = Term221*Term222*Term223 + Term224
        Term2 = Term21*Term22
        VR = ((1/gamma_el)*(Term1+Term2))*units.MN
        VRMax = self._EC8_ShearMax(b, h, cv, dbw, dbl)
        return min(float(VR), float(VRMax))

    def _EC8_ShearMax(self,
                      b, h, cv,
                      dbw, dbl):
        """
        Calculate maximum shear capacity due to diagonal compression.

        Implements EC2 Part 1 Section 6.2.3 for diagonal compression
        limit to prevent crushing before reinforcement yielding.

        Args:
            b: Section width (mm)
            h: Section height (mm)
            cv: Concrete cover (mm)
            dbw: Transverse bar diameter (mm)
            dbl: Longitudinal bar diameter (mm)

        Returns:
            float: Maximum shear capacity due to concrete compression
        """
        # Diagonal compression in the web
        # Calculated as per EC2 P1 - 6.2.3
        v1 = 0.6*(1-(self.fc/units.MPa)/250)
        # Non-Prestressed Section
        alpha_cw = 1.0
        # Assumed as 45 degrees
        thetaRad = np.deg2rad(45)

        z = 0.9*(h-cv-dbw-0.5*dbl)
        bw = b-cv-dbw-0.5*dbl
        VR_Max = (alpha_cw*bw*z*v1*self.fc /
                  (1.15*((np.tan(thetaRad) + 1/np.tan(thetaRad)))))
        return VR_Max
