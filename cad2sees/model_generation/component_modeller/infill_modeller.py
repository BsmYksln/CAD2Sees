"""
CAD2Sees module dedicated to modeling masonry infills.

Provides modelling capabilities for masonry infills with various
models for properties, capacity, backbone curves, and hysteretic
behaviour for nonlinear analysis.

References
----------
Mucedero, G., D. Perrone, and R. Monteiro. 2021. "Nonlinear Static
Characterisation of Masonry-Infilled RC Building Portfolios Accounting for
Variability of Infill Properties." Bulletin of Earthquake Engineering 19 (6):
2597–2641. https://doi.org/10.1007/s10518-021-01068-2
"""

import openseespy.opensees as ops
from cad2sees.helpers import units
import numpy as np


class In_Plane:
    """
    In-plane infill panel modelling for masonry-infilled RC frames.

    Models masonry infill panels with various empirical approaches for
    properties, capacity, backbone curves, and hysteretic behaviour.

    Parameters
    ----------
    InfillData : dict
        Infill geometry and loading data including panel dimensions,
        node IDs, loads, and frame element properties
    InfillMechanicalProps : dict or int or str
        Mechanical properties - either custom dict or predefined set
        (1: Weak, 2: Medium, 3: Strong, 4: Morandi, 5: Clay, 6: Calcarenite)
    StiffnessReductionType : int or str
        Opening reduction model (0: None, 1: DaweSeah, 2: ImaiMiyamoto, etc.)
    StrutWidthType : int or str
        Strut width method (1: Bertoldi, 2: PaulayPriestley,
        5: Mainstone, etc.)
    CriticalStressFlag : int or str
        Capacity calculation method (1-7: Various failure mechanisms)
    BackboneType : int or str
        Backbone curve model (1: Bertoldi, 2: PanagiotakosFardis, etc.)
    HysteresisType : int or str
        Hysteretic model (1: NoSD, 2: Nohetal)
    alphaPost : float, optional
        Post-peak stiffness ratio (default -1: auto)
    betaPost : float, optional
        Post-peak strength ratio (default -1: auto)
    EpsThetaFlag : int, optional
        Strain calculation flag for Sassun model (default 1)
    RedcutionEffectiveness : float, optional
        Opening reduction factor (default 1)

    Methods
    -------
    Capacity()
        Calculate infill capacity
    BackBone()
        Generate backbone curve
    Hysteresis()
        Define hysteretic parameters
    Frame_Interaction()
        Calculate interaction coefficients
    Build()
        Create OpenSees model
    BuildWInteraction()
        Build with interaction effects
    """
    def __init__(self,
                 InfillData,
                 InfillMechanicalProps,
                 StiffnessReductionType,
                 StrutWidthType,
                 CriticalStressFlag,
                 BackboneType,
                 HysteresisType,
                 alphaPost=-1,
                 betaPost=-1,
                 EpsThetaFlag=1,
                 RedcutionEffectiveness=1):
        """Initialise infill panel model with specified parameters."""

        def _Haketal2012Weak():
            """Weak infill properties as per Hak et al. 2012."""
            self.tw = 80.00 * units.mm
            self.Ewv = 1873.00 * units.MPa
            self.Ewh = 991.00 * units.MPa
            self.Gw = 1089.00 * units.MPa
            self.v = 0.250
            self.fwv = 2.020 * units.MPa
            self.fwu = 0.22 * units.MPa
            self.fws = 0.55 * units.MPa
            self.flat = 1.18 * units.MPa
            self.sig_v = 0.0 * units.MPa
            self.sigm_cr = 0.0 * units.MPa

        def _Haketal2012Medium():
            """Medium infill properties as per Hak et al. 2012."""
            self.tw = 240.00 * units.mm
            self.Ewv = 1873.00 * units.MPa
            self.Ewh = 991.00 * units.MPa
            self.Gw = 1089.00 * units.MPa
            self.v = 0.250
            self.fwv = 1.500 * units.MPa
            self.fwu = 0.250 * units.MPa
            self.fws = 0.310 * units.MPa
            self.flat = 1.110 * units.MPa
            self.sig_v = 0.0 * units.MPa
            self.sigm_cr = 0.0 * units.MPa

        def _Haketal2012Strong():
            """Strong infill properties as per Hak et al. 2012."""
            self.tw = 300.00 * units.mm
            self.Ewv = 3240.00 * units.MPa
            self.Ewh = 1050.00 * units.MPa
            self.Gw = 1296.00 * units.MPa
            self.v = 0.250
            self.fwv = 3.510 * units.MPa
            self.flat = 1.500 * units.MPa
            self.fwu = 0.300 * units.MPa
            self.fws = 0.360 * units.MPa
            self.sig_v = 0.0 * units.MPa
            self.sigm_cr = 0.0 * units.MPa

        def _Morandietal2012():
            self.tw = 350.00*units.mm
            self.Ewv = 5299.00*units.MPa
            self.Ewh = 494.00*units.MPa
            self.Gw = 2120.00*units.MPa
            self.v = 0.25
            self.fwv = 4.640*units.MPa
            self.flat = 1.080*units.MPa
            self.fwu = 0.359*units.MPa
            self.fws = 0.0*units.MPa
            self.sig_v = 0.0*units.MPa
            self.sigm_cr = 0.0*units.MPa

        def _Cavalierietal2014Clay():
            self.tw = 150.00*units.mm
            self.Ewv = 6401.00*units.MPa
            self.Ewh = 5038.00*units.MPa
            self.Gw = 2547.00*units.MPa
            self.v = 0.25
            self.fwv = 8.660*units.MPa
            self.flat = 4.180*units.MPa
            self.fwu = 1.07*units.MPa
            self.fws = 0.0*units.MPa
            self.sig_v = 0.0*units.MPa
            self.sigm_cr = 0.0*units.MPa

        def _Cavalierietal2014Calcarenite():
            self.tw = 210.00*units.mm
            self.Ewv = 7106.00*units.MPa
            self.Ewh = 9528.00*units.MPa
            self.Gw = 2937.00*units.MPa
            self.v = 0.250
            self.fwv = 4.570*units.MPa
            self.flat = 3.920*units.MPa
            self.fwu = 0.89*units.MPa
            self.fws = 0.0*units.MPa
            self.sig_v = 0.0*units.MPa
            self.sigm_cr = 0.0*units.MPa

        self.InfD = InfillData
        self.IMP = InfillMechanicalProps
        self.SRT = StiffnessReductionType
        self.SWT = StrutWidthType
        self.CSF = CriticalStressFlag
        self.IPBBT = BackboneType
        self.alphapost = alphaPost
        self.betapost = betaPost
        self.epsthetaflag = EpsThetaFlag
        self.RE = RedcutionEffectiveness
        self.HT = HysteresisType

        if isinstance(self.IMP, dict):
            self.tw = self.IMP['tw']
            self.Ewv = self.IMP['Ewv']
            self.Ewh = self.IMP['Ewh']
            self.Gw = self.IMP['Gw']
            self.v = self.IMP['v']
            self.fwv = self.IMP['fwv']
            self.fwu = self.IMP['fwu']
            self.fws = self.IMP['fws']
            self.flat = self.IMP['flat']
            self.sig_v = self.IMP['sig_v']
            self.sigm_cr = self.IMP['sigm_cr']
        elif self.IMP in (1, '1', 'Haketal2012Weak', 'Weak'):
            _Haketal2012Weak()
        elif self.IMP in (2, '2', 'Haketal2012Medium', 'Medium'):
            _Haketal2012Medium()
        elif self.IMP in (3, '3', 'Haketal2012Strong', 'Strong'):
            _Haketal2012Strong()
        elif self.IMP in (4, '4', 'Morandietal2018'):
            _Morandietal2012()
        elif self.IMP in (5, '5', 'Cavalierietal2014Clay'):
            _Cavalierietal2014Clay()
        elif self.IMP in (6, '6', 'Cavalierietal2014Calcarenite'):
            _Cavalierietal2014Calcarenite()

        self.Fv = self.InfD['Fv']
        self.Ec = self.InfD['Ec']

        self.lp = self.InfD['lp']
        self.hp = self.InfD['hp']

        self.B = self.InfD['B']
        self.H = self.InfD['H']

        self.hc = self.InfD['hc']
        self.bc = self.InfD['bc']
        self.Ac = self.bc*self.hc
        self.Ic = self.bc*(self.hc**3)/12

        self.hb = self.InfD['hb']
        self.bb = self.InfD['bb']

        self.Bw = self.B - self.bc
        self.Hw = self.H - self.hb
        self.thetarad = np.arctan2(self.Hw, self.Bw)
        self.Ewtheta = (((np.sin(self.thetarad)**4)/(self.Ewv)) +
                        ((np.cos(self.thetarad)**4)/(self.Ewh)) +
                        ((np.sin(self.thetarad)**2) *
                         (np.cos(self.thetarad)**2) *
                         (1/self.Gw - 2*self.v/self.Ewh)))**-1
        self.dw = (self.Bw**2 + self.Hw**2)**0.5
        self.lambdaH = self.Hw*((self.Ewtheta*self.tw *
                                np.sin(2*self.thetarad)) /
                                (4*self.Ec*self.Ic*self.Hw))**0.25

    def Capacity(self):
        """
        Calculate infill panel capacity based on selected failure mechanism.

        Returns critical force capacity considering compression, shear, and
        tension failures with opening effects and equivalent strut width.

        Returns
        -------
        float
            Critical force capacity of the infill panel (N)
        """

        def CriticalForce():

            def _DecaniniFantini_Bertoldietal1987():
                if self.lambdaH < 3.14:
                    K1, K2 = 1.300, -0.178
                elif 3.14 < self.lambdaH < 7.85:
                    K1, K2 = 0.707, 0.010
                elif self.lambdaH > 7.85:
                    K1, K2 = 0.470, 0.040
                # Compression in centre
                sigw1 = (1.16*self.fwv*np.tan(self.thetarad) /
                         (K1+K2*self.lambdaH))
                # Compression at corners
                sigw2 = (1.12*self.fwv*np.sin(self.thetarad) *
                         np.cos(self.thetarad) /
                         ((K1*self.lambdaH**(-0.12))+(K2*self.lambdaH**0.88)))
                # Shear sliding
                sigw3 = ((self.fwu*(1.2*np.sin(self.thetarad) +
                                    0.45*np.cos(self.thetarad)) +
                          0.3*self.sig_v)*self.dw/self.bw)
                if self.fws > 0:
                    # Diagonal Tension
                    sigw4 = (0.6*self.fws+0.3*self.sig_v)*self.dw/self.bw
                # Failure Mechanism
                if self.fws > 0:
                    sigw = min(sigw1, sigw2, sigw3, sigw4)
                else:
                    sigw = min(sigw1, sigw2, sigw3)
                self.force = sigw*self.Aw
                self.CSFtxt = 'Decanini Fantini-Bertoldi et al. 1987'

            def _DecaniniFantini_Morandietal2018():
                fwuCur = 0.5*self.fwu
                if self.lambdaH < 3.14:
                    K1, K2 = 1.300, -0.178
                elif 3.14 < self.lambdaH < 7.85:
                    K1, K2 = 0.707, 0.010
                elif self.lambdaH > 7.85:
                    K1, K2 = 0.470, 0.040

                # Compression in centre
                sigw1 = (1.16*self.fwv*np.tan(self.thetarad) /
                         (K1+K2*self.lambdaH))
                # Compression at corners
                sigw2 = (1.12*self.fwv*np.sin(self.thetarad) *
                         np.cos(self.thetarad) /
                         ((K1*self.lambdaH**(-0.12))+(K2*self.lambdaH**0.88)))
                # Shear sliding
                sigw3 = ((fwuCur*(1.2*np.sin(self.thetarad) +
                                  0.45*np.cos(self.thetarad)) +
                          0.3*self.sig_v)*self.dw/self.bw)
                if self.fws > 0:
                    # Diagonal Tension
                    sigw4 = (0.6*self.fws+0.3*self.sig_v)*self.dw/self.bw
                # Failure Mechanism
                if self.fws > 0:
                    sigw = min(sigw1, sigw2, sigw3, sigw4)
                else:
                    sigw = min(sigw1, sigw2, sigw3)

                self.force = sigw*self.Aw
                self.CSFtxt = ('Decanini Fantini with reduction of fwu as per '
                               'Morandi et al. 2018')

            def _PaulayPriestley1992():
                # Shear sliding failure of the infill
                Vs = (self.fwu/(1-0.3*self.H*self.B))*self.Bw*self.tw
                # Compression failure of diagonal strut
                Vc = 2*self.z*self.tw*self.flat/3
                Vt = self.tw*self.dw*self.fwv*units.pi*0.5
                self.force = min(Vs, Vc, Vt)
                self.CSFtxt = 'Paulay Priestley 1992'

            def _PaulayPriestley1992WithfwuReductionMorandietal2018():
                fwu = 0.5*self.fwu
                # Shear sliding failure of the infill
                Vs = (fwu/(1-0.3*self.H*self.B))*self.Bw*self.tw
                # Compression failure of diagonal strut
                Vc = 2*self.z*self.tw*self.flat/3
                Vt = self.tw*self.dw*self.fwv*units.pi*0.5
                self.force = min(Vs, Vc, Vt)
                self.CSFtxt = ('Paulay Priestley 1992 reduction of fwu as per '
                               'Morandi et al. 2018')

            def _FEMA306():
                # equivalent width of the strut
                bw = 0.175*self.dw*(self.lambdaH**(-0.4))
                # Shear sliding failure of the infill
                Vs = (self.fwu+0.4*self.sig_v)*self.Bw*self.tw
                # Compression failure
                Vc = bw*self.tw*self.flat*np.cos(self.thetarad)
                if self.sigm_cr > 0:
                    # Diagonal cracking failure
                    Vcr = (2*(2**0.5) *
                           self.tw*self.Bw*self.sigm_cr*self.Hw*self.Bw /
                           (self.Bw**2+self.Hw**2))
                # Limit
                Vmi = (2 *
                       ((0.0069*self.fwv/units.MPa)**0.5)*units.MPa *
                       self.tw*self.Bw)
                # Limit
                Vmf = 0.3*Vmi
                if self.sigm_cr > 0:
                    force = min(Vs, Vc, Vcr)
                else:
                    force = min(Vs, Vc)
                self.force = max(min(Vmi, force), Vmf)
                self.CSFtxt = 'FEMA 306'

            def _EC8P1_EC6_WithoutReductionofFvo():
                # Shear sliding failure of the infill
                self.force = (self.fwu+0.4*self.sig_v)*self.Bw*self.tw
                self.CSFtxt = ('Eurocode 8 - Part1 /Eurocode 6 '
                               'without reduction of Fvo')

            def _EC8P1_EC6_WithReductionofFvo():
                # Shear sliding failure of the infill
                self.force = (0.5*self.fwu+0.4*self.sig_v)*self.Bw*self.tw
                self.CSFtxt = ('Eurocode 8 - Part1 /Eurocode 6 '
                               'with reduction of Fvo')

            if self.CSF in (1, '1', 'DecaniniFantini-Bertoldietal1987'):
                _DecaniniFantini_Bertoldietal1987()
            elif self.CSF in (2, '2', 'DecaniniFantini-Morandietal2018'):
                _DecaniniFantini_Morandietal2018()
            elif self.CSF in (3, '3', 'PaulayPriestley1992'):
                _PaulayPriestley1992()
            elif self.CSF in (4, '4',
                              'PaulayPriestley1992_fwuRedMorandietal2018'):
                _PaulayPriestley1992WithfwuReductionMorandietal2018()
            elif self.CSF in (5, '5', 'FEMA306'):
                _FEMA306()
            elif self.CSF in (6, '6', 'EC8-P1/EC6woRedFvo'):
                _EC8P1_EC6_WithoutReductionofFvo()
            elif self.CSF in (7, '7', 'EC8-P1/EC6wRedFvo'):
                _EC8P1_EC6_WithReductionofFvo

        def StrutWidth():

            def _Bertoldi1993():
                if self.lambdaH <= 3.14:
                    K1, K2 = 1.3, -0.178
                elif 3.14 < self.lambdaH <= 7.85:
                    K1, K2 = 0.707, 0.010
                elif self.lambdaH > 7.85:
                    K1, K2 = 0.470, 0.040
                bwdwrat = K2 + K1/self.lambdaH
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            def _PaulayPriestley1992():
                self.bw = 0.25*self.dw
                self.Aw = self.bw*self.tw

            def _Holmes1961():
                self.bw = 0.33*self.dw
                self.Aw = self.bw*self.tw

            def _LiauwKwan1984():
                bwdwrat = 0.95*np.sin(2*self.thetarad)/(2*(self.lambdaH**0.5))
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            def _Mainstone1974():
                bwdwrat = 0.175*self.lambdaH**(-0.4)
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            def _StaffordSmith1961():
                self.bw = self.dw*0.1
                self.Aw = self.bw*self.tw

            def _DecaniniFantin1987Uncracked():
                if self.lambdaH <= 7.85:
                    bwdwrat = 0.085 + 0.748*self.H/self.lambdaH  # !!!
                else:
                    bwdwrat = 0.130 + 0.393*self.H/self.lambdaH  # !!!
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            def _DecaniniFantin1987Cracked():
                if self.lambdaH <= 7.85:
                    bwdwrat = 0.010 + 0.707*self.H/self.lambdaH  # !!!
                else:
                    bwdwrat = 0.040 + 0.470*self.H/self.lambdaH  # !!!
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            def _Cavalerietal2005():
                epsilonv = 0.5*self.Fv/(self.Ac*self.Ec)
                k = 1+(18*self.lambdaH+200)*epsilonv
                a1 = (np.sin(self.thetarad)**4) + (np.cos(self.thetarad)**4)
                a2 = (np.sin(self.thetarad)*np.cos(self.thetarad))**2
                a3 = (1 / self.Ewh + 1 / self.Ewv - 1 / self.Gw) * a2
                a4 = a1*self.v/self.Ewh
                a5 = a4-a3
                vd = a5*self.Ewtheta
                c = 0.249-0.0116*vd+0.567*vd**2
                zeta = 0.250*self.B/self.H-0.250+1.00
                lambdastr = (((self.Ewtheta*self.tw*self.H) /
                             (self.Ec*self.Ac)) *
                             (((self.H/self.B)**2) +
                              (0.25*self.Ac*self.B/(self.Ab*self.H))))
                beta = 0.146+0.0073*vd+0.126*vd**2
                bwdwrat = k*c/(zeta*lambdastr**beta)
                self.bw = self.dw*bwdwrat
                self.Aw = self.bw*self.tw

            if self.SWT in (1, '1', 'Bertoldi1993'):
                _Bertoldi1993()
            elif self.SWT in (2, '2', 'PaulayPriestley1992'):
                _PaulayPriestley1992()
            elif self.SWT in (3, '3', 'Holmes1961'):
                _Holmes1961()
            elif self.SWT in (4, '4', 'LiauwKwan1984'):
                _LiauwKwan1984()
            elif self.SWT in (5, '5', 'Mainstone1974', 'FEMA306'):
                _Mainstone1974()
            elif self.SWT in (6, '6', 'StaffordSmith1961'):
                _StaffordSmith1961()
            elif self.SWT in (7, '7', 'DecaniniFantin1987Uncracked'):
                _DecaniniFantin1987Uncracked()
            elif self.SWT in (8, '8', 'DecaniniFantin1987Cracked'):
                _DecaniniFantin1987Cracked()
            elif self.SWT in (9, '9', 'Cavalerietal2005'):
                _Cavalerietal2005()

        def StiffnessReduction():

            def _DaweSeah(alphaL):
                self.rp = 1-1.5*alphaL*0.01
                if alphaL > 66:
                    print('Warning: Dawe and Seah 1988' +
                          'opening reduction is limited for %66 of alphaL ' +
                          f'[{alphaL}] !!!')
                self.SRTtxt = 'Dawe and Seah 1988'

            def _ImaiMiyamoto(alphaL, alphaA):
                rp1 = 1-0.01*alphaL
                rp2 = 1-0.1*(alphaA)**0.5
                self.rp = min(rp1, rp2)
                self.SRTtxt = 'Imai and Miyamoto 1989'

            def _TasnimiMohebkhah(alphaA):
                self.rp = 1-2.238*0.01*alphaA + 1.49*(0.01*alphaA)**2
                if alphaA > 40:
                    print('Warning: Tasnimi and Mohebkhah 2011 opening' +
                          'reduction is limited for %40 of alphaA '
                          f'[{alphaA}]  !!!')
                self.SRTtxt = 'Tasnimi and Mohebkhah 2011'

            def _Decaninietal2014(alphaL, alphaA):
                self.rp = (0.55*np.exp(-0.035*alphaA) +
                           0.44*np.exp(-0.025*alphaL))
                self.SRTtxt = 'Decanini et al. 2014 (Median value)'

            def _Asterisetal2011(alphaA):
                self.rp = 1-2*(alphaA**0.54) + alphaA**1.14
                self.SRTtxt = 'Asteris et al. 2011'

            alphaL = 100*self.lp/self.Bw
            alphaA = 100*(self.lp*self.hp)/(self.Bw*self.Hw)

            if self.SRT in (0, '0', 'No', 'no'):
                self.rp = 1.0
                self.SRTtxt = 'No Opening'
            elif self.SRT in (1, '1', 'DaweSeah'):
                _DaweSeah(alphaL)
            elif self.SRT in (2, '2', 'ImaiMiyamoto'):
                _ImaiMiyamoto(alphaL, alphaA)
            elif self.SRT in (3, '3', 'TasnimiMohebkhah'):
                _TasnimiMohebkhah(alphaA)
            elif self.SRT in (4, '4', 'Decaninietal2014'):
                _Decaninietal2014(alphaL, alphaA)
            elif self.SRT in (5, '5', 'Asterisetal2011'):
                _Asterisetal2011(alphaA)

        StiffnessReduction()
        StrutWidth()
        CriticalForce()

        return self.force

    def BackBone(self):
        """
        Generate backbone curve parameters for infill panel.

        Defines force-displacement backbone curve with key points for
        cracking, peak, and residual strengths based on selected model.
        """

        def _Bertoldietal1993():
            # Post Stiffness ratio
            if self.alphapost == -1:
                self.alphapost = 0.02
            # Stiffness
            # Ksec as per Mainstone
            Ksec = (self.Ewtheta*self.bw*self.tw*(np.cos(self.thetarad)**2) /
                    self.dw)
            Kel = 4*Ksec
            Kpost = self.alphapost*Ksec
            force1 = 0.8*self.force
            force2 = self.force
            force3 = 0.35*self.force
            u_cr = force1/Kel
            u_max = force2/Ksec
            u_res = u_max + (force2 - force3)/Kpost

            self.ULimits = [u_cr,
                            u_max,
                            u_res,
                            u_res]

            self.Forces = [force1,
                           force2,
                           force3,
                           force3]

        def _PanagiotakosFardis():
            if self.alphapost == -1:
                self.alphapost = 0.01
            if self.betapost == -1:
                self.betapost = 0.01
            Kel = self.Gw*self.Bw*self.tw/self.Hw
            # Ksec as per Mainstone
            Ksec = (self.Ewtheta*self.bw*self.tw*(np.cos(self.thetarad)**2) /
                    self.dw)
            Ksoft = Kel*self.alphapost

            force1 = self.force
            force2 = force1*1.3
            force3 = self.force*self.betapost

            u_cr = force1/Kel
            u_max = force2/Ksec
            u_res = u_max + (force2 - force3)/Ksoft
            self.ULimits = [u_cr,
                            u_max,
                            u_res,
                            u_res]

            self.Forces = [force1,
                           force2,
                           force3,
                           force3]

        def _DeRisietal2018():
            self.alphapost = 0.1
            Kms = (self.Ewtheta*self.bw*self.tw*(np.cos(self.thetarad)**2) /
                   self.dw)
            Ksec = 0.8*Kms
            Kel = 2.8*Kms
            Kpost = self.alphapost*Kms

            force1 = 0.7*self.force
            force2 = self.force
            force3 = 0.0

            u_cr = force1/Kel
            u_max = force2/Ksec
            u_res = u_max + (force2 - force3)/Kpost

            self.ULimits = [u_cr,
                            u_max,
                            u_res,
                            u_res]

            self.Forces = [force1,
                           force2,
                           force3,
                           force3]

        def _Sassunetal2015():
            force1 = 0.8*self.force
            force2 = self.force
            force3 = self.force*0.35

            thetaDS1 = 0.0018
            thetaDS2 = 0.0046
            thetaDS3 = 0.0105
            thetaDS4 = 0.0188

            if self.epsthetaflag == 1:
                u_cr = 0.0006*self.dw
                u_max = 0.0013*self.dw
                u_res = 0.0045*self.dw
            elif self.epsthetaflag == 2:
                u_cr = ((1-((1+(self.Bw/self.Hw-thetaDS1)**2) /
                            (1+(self.Bw/self.Hw)**2))**0.5)*self.dw)
                u_max = ((1-((1+(self.Bw/self.Hw-thetaDS2)**2) /
                             (1+(self.Bw/self.Hw)**2))**0.5)*self.dw)
                u_res = ((1-((1+(self.Bw/self.Hw-thetaDS4)**2) /
                             (1+(self.Bw/self.Hw)**2))**0.5)*self.dw)
                u_DS3 = ((1-((1+(self.Bw/self.Hw-thetaDS3)**2) /
                             (1+(self.Bw/self.Hw)**2))**0.5)*self.dw)

            self.ULimits = [u_cr,
                            u_max,
                            u_DS3,
                            u_res]

            self.Forces = [force1,
                           force2,
                           force3,
                           force3]

        if self.IPBBT in (1, '1', 'Bertoldietal1993'):
            _Bertoldietal1993()
        elif self.IPBBT in (2, '2', 'PanagiotakosFardis'):
            _PanagiotakosFardis()
        elif self.IPBBT in (3, '3', 'DeRisietal2018'):
            _DeRisietal2018()
        elif self.IPBBT in (4, '4', 'Sassunetal2015'):
            _Sassunetal2015()

    def Hysteresis(self):
        """
        Define hysteretic material parameters for infill panel.

        Establishes hysteretic behaviour parameters for cyclic loading
        including stress-strain envelope and degradation characteristics.
        """
        if self.RE == 1:
            sigma1 = self.rp*self.Forces[0]/self.Aw
            sigma2 = self.rp*self.Forces[1]/self.Aw
            sigma3 = self.rp*self.Forces[2]/self.Aw
            sigma4 = self.rp*self.Forces[2]/self.Aw
        else:
            sigma1 = self.Forces[0]/self.Aw
            sigma2 = self.Forces[1]/self.Aw
            sigma3 = self.Forces[2]/self.Aw
            sigma4 = self.Forces[2]/self.Aw

        epsilon1 = self.ULimits[0]/self.dw
        epsilon2 = self.ULimits[1]/self.dw
        epsilon3 = self.ULimits[2]/self.dw
        epsilon4 = self.ULimits[2]/self.dw
        sigma1_p = 0.1*sigma1
        sigma2_p = 0.1*sigma2
        sigma3_p = 0.1*sigma3
        sigma4_p = 0.1*sigma4
        epsilon1_p = epsilon1 * 0.1
        epsilon2_p = epsilon2 * 0.1
        epsilon3_p = epsilon3 * 0.1
        epsilon4_p = epsilon4 * 0.1

        # Hysteresis Type

        # Coefficients for Unloading/Reloading Stiffness degradation and
        # Coefficients for Strength degradation ==0
        if self.HT in (1, 'NoSD'):
            pF = [sigma1_p, sigma2_p, sigma3_p, sigma4_p]
            nF = [-sigma1, -sigma2, -sigma3, -sigma4]
            pD = [epsilon1_p, epsilon2_p, epsilon3_p, epsilon4_p]
            nD = [-epsilon1, -epsilon2, -epsilon3, -epsilon4]
            rDisp = [0.8, 0.8]
            rForce = [0.1, 0.1]
            uForce = [0.0, 0.0]
            gammaK = [0.0, 0.0, 0.0, 0.0, 0.0]
            gammaD = [0.0, 0.0, 0.0, 0.0, 0.0]
            gammaF = [0.0, 0.0, 0.0, 0.0, 0.0]
            gammaE = 0.0
            damage = 'energy'
        # Coefficients for Unloading/Reloading Stiffness degradation and
        # Coefficients for Strength degradation based on
        # N. Mohammad Noh et al. / Engineering Structures 150 (2017) 599–621
        elif self.HT in (2, 'Nohetal'):
            pF = [sigma1_p, sigma2_p, sigma3_p, sigma4_p]
            nF = [-sigma1, -sigma2, -sigma3, -sigma4]
            pD = [epsilon1_p, epsilon2_p, epsilon3_p, epsilon4_p]
            nD = [-epsilon1, -epsilon2, -epsilon3, -epsilon4]
            rDisp = [0.8, 0.8]
            rForce = [0.1, 0.1]
            uForce = [0.0, 0.0]
            gammaK = [0.8, 0.7, 0.7, 0.7, 0.8]
            gammaD = [0.0, 0.0, 0.0, 0.0, 0.1]
            gammaF = [1.0, 0.0, 1.0, 1.0, 0.1]
            gammaE = 10
            damage = 'cycle'

        self.ModellingMaterialInfo = {'pF': pF,
                                      'nF': nF,
                                      'pD': pD,
                                      'nD': nD,
                                      'rDisp': rDisp,
                                      'rForce': rForce,
                                      'uForce': uForce,
                                      'gammaK': gammaK,
                                      'gammaD': gammaD,
                                      'gammaF': gammaF,
                                      'gammaE': gammaE,
                                      'damage': damage}

    def Frame_Interaction(self):
        """
        Calculate frame-infill interaction coefficients.

        Computes interaction coefficients for frame element capacity
        modifications to represent composite infilled frame behaviour.
        """
        ar = self.Bw/self.Hw
        lambdastar = ((self.Ewtheta/self.Ec) *
                      (self.tw*self.H/(self.hc*self.bc)) *
                      (((self.H/self.B)**2) +
                       0.25*(self.B*self.hc*self.bc) /
                       (self.H*self.bb*self.hb)))
        Bstar = min(1, max(0.7, (2-ar)*0.3+0.7))
        sigma2 = -self.ModellingMaterialInfo['nF'][1]
        # fvom = sigma2*self.bw*Bstar/units.MPa  # This should be checked??
        fvom = sigma2*self.Aw/(self.tw*self.dw*Bstar)/units.MPa
        xistar = self.hb/self.hc  # I would say hb/bc but ok?
        psi = lambdastar * xistar * fvom

        aBNO1 = 0.98*(psi**(-0.33))
        aBNO2 = 0.60*(psi**(-0.39))
        aBNO = min(aBNO2,
                   max(aBNO1, aBNO1+(ar-1)*(aBNO2-aBNO1)),
                   np.sin(self.thetarad))

        aCNO1 = 0.96*(psi**(-0.37))
        aCNO2 = 1.05*(psi**(-0.36))
        aCNO = min(aCNO2,
                   max(aCNO1, aCNO1+(ar-1)*(aCNO2-aCNO1)),
                   np.cos(self.thetarad))

        aBSE1 = 1.03*(psi**(-0.32))
        aBSE2 = 0.68*(psi**(-0.32))
        aBSE = min(aBSE2,
                   max(aBSE1, aBSE1+(ar-1)*(aBSE2-aBSE1)),
                   np.sin(self.thetarad))

        aCSE1 = 1.03*(psi**(-0.35))
        aCSE2 = 1.08*(psi**(-0.30))
        aCSE = min(aCSE2,
                   max(aCSE1, aCSE1+(ar-1)*(aCSE2-aCSE1)),
                   np.cos(self.thetarad))

        self.Alphas = [aBNO, aCNO, aBSE, aCSE]

        self.RelatedFrames = []
        self.RelatedNodes = []
        Frames = ['BNO_Frame', 'CNO_Frame', 'BSE_Frame', 'CSE_Frame']
        Nodes = ['BNO_Node', 'CNO_Node', 'BSE_Node', 'CSE_Node']

        for F in Frames:
            try:
                self.RelatedFrames.append(int(self.InfD[F]))
            except ValueError:
                self.RelatedFrames.append(int(0))

        for N in Nodes:
            try:
                self.RelatedNodes.append(int(self.InfD[N]))
            except ValueError:
                self.RelatedNodes.append(int(0))

    def Build(self):
        """
        Create OpenSees material and element models for infill panel.

        Generates complete OpenSees model combining capacity, backbone,
        and hysteresis parameters as Pinching4 material with truss element.
        """
        self.Capacity()
        self.BackBone()
        self.Hysteresis()

        materialTag = int(float(f"8{self.InfD['ID']}"))
        ElementTag = int(self.InfD['ID'])
        NodeITag = int(self.InfD['i_ID'])
        NodeJTag = int(self.InfD['j_ID'])

        UniaxialPinchingFun(materialTag,
                            self.ModellingMaterialInfo['pF'],
                            self.ModellingMaterialInfo['nF'],
                            self.ModellingMaterialInfo['pD'],
                            self.ModellingMaterialInfo['nD'],
                            self.ModellingMaterialInfo['rDisp'],
                            self.ModellingMaterialInfo['rForce'],
                            self.ModellingMaterialInfo['uForce'],
                            self.ModellingMaterialInfo['gammaK'],
                            self.ModellingMaterialInfo['gammaD'],
                            self.ModellingMaterialInfo['gammaF'],
                            self.ModellingMaterialInfo['gammaE'],
                            self.ModellingMaterialInfo['damage'])
        ops.element('Truss', ElementTag, NodeITag, NodeJTag,
                    self.Aw, materialTag)

    def BuildWInteraction(self):
        """
        Build infill model with frame-infill interaction effects.

        Extends basic Build() method by additionally calculating interaction
        coefficients for modifying surrounding frame element properties.
        """
        self.Build()
        self.Frame_Interaction()


def UniaxialPinchingFun(materialTag,
                        pEnvelopeStress,
                        nEnvelopeStress,
                        pEnvelopeStrain,
                        nEnvelopeStrain,
                        rDisp, rForce, uForce,
                        gammaK, gammaD, gammaF, gammaE,
                        damage):
    """
    Create a uniaxial Pinching4 material for OpenSees.

    Args:
        material_tag (int): Unique material tag.
        p_envelope_force (list): [ePf1, ePf2, ePf3, ePf4] positive envelope force points.
        n_envelope_force (list): [eNf1, eNf2, eNf3, eNf4] negative envelope force points.
        p_envelope_deform (list): [ePd1, ePd2, ePd3, ePd4] positive envelope deformation points.
        n_envelope_deform (list): [eNd1, eNd2, eNd3, eNd4] negative envelope deformation points.
        r_disp (list): [rDispP, rDispN] ratio of deformation at reloading (pos, neg).
        r_force (list): [rForceP, rForceN] ratio of force at reloading (pos, neg).
        u_force (list): [uForceP, uForceN] ratio of strength upon unloading (pos, neg).
        gamma_k (list): [gK1, gK2, gK3, gK4, gKLim] unloading stiffness degradation.
        gamma_d (list): [gD1, gD2, gD3, gD4, gDLim] reloading stiffness degradation.
        gamma_f (list): [gF1, gF2, gF3, gF4, gFLim] strength degradation.
        gamma_e (float): gE, energy degradation parameter.
        damage (str): dmgType, 'energy' or 'cycle'.
    """

    ops.uniaxialMaterial('Pinching4', materialTag,
                         pEnvelopeStress[0], pEnvelopeStrain[0],
                         pEnvelopeStress[1], pEnvelopeStrain[1],
                         pEnvelopeStress[2], pEnvelopeStrain[2],
                         pEnvelopeStress[3], pEnvelopeStrain[3],
                         nEnvelopeStress[0], nEnvelopeStrain[0],
                         nEnvelopeStress[1], nEnvelopeStrain[1],
                         nEnvelopeStress[2], nEnvelopeStrain[2],
                         nEnvelopeStress[3], nEnvelopeStrain[3],
                         rDisp[0], rForce[0], uForce[0],
                         rDisp[1], rForce[1], uForce[1],
                         gammaK[0], gammaK[1], gammaK[2],
                         gammaK[3], gammaK[4],
                         gammaD[0], gammaD[1], gammaD[2],
                         gammaD[3], gammaD[4],
                         gammaF[0], gammaF[1], gammaF[2],
                         gammaF[3], gammaF[4],
                         gammaE, damage)
