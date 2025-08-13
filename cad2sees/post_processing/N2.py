"""
N2 Method Module

This module implements the N2 nonlinear static procedure for seismic performance evaluation
of structures, including the effects of masonry infills. It combines pushover analysis with
the response spectrum approach to estimate displacement demand and ductility factors,
following the recommendations of Eurocode 8 (EC8).

The implementation supports both standard N2 and enhanced N2 analyses that account for masonry infill effects.

Classes:
    n2: Main class for N2 method handling pushover curve processing

References:
- Fajfar, P. (2000). “A Nonlinear Analysis Method for Performance-Based Seismic Design.” 
  Earthquake Spectra, 16(3), 573–592. https://doi.org/10.1193/1.1586128
- Martinelli, E., Lima, C., & De Stefano, G. (2015). “A Simplified Procedure for Nonlinear Static Analysis of Masonry-Infilled RC Frames.” 
  Engineering Structures, 101, 591–608. https://doi.org/10.1016/j.engstruct.2015.07.023
"""

import os
import json
import numpy as np
from cad2sees.post_processing import spectra
from cad2sees.helpers import general_util_functions as guf


class N2:
    """
    N2 method implementation.

    Performs nonlinear static (pushover) analysis according to EC8 standards
    with optional infill panel effects. Calculates displacement demand and
    ductility factors from pushover curves and response spectra.
    """
    def __init__(self,
                 pga,
                 SoilClass,
                 SPOCurve,
                 SPOInfo,
                 OutDir,
                 CapacityDir,
                 Nation='',
                 SpectrumType=1):
        """
        Initialize N2 analysis with pushover data and seismic parameters.

        Parameters
        ----------
        pga : float
            Peak ground acceleration.
        SoilClass : str
            Soil classification ('A', 'B', 'C', 'D', 'E').
        SPOCurve : array_like
            Pushover curve data [displacement, base_shear].
        SPOInfo : array_like
            Structural information [height, mass, mode_shape].
        OutDir : str
            Output directory path.
        CapacityDir : str
            Capacity data directory path.
        Nation : str, optional
            National annex (default: '').
        SpectType : int, optional
            Spectrum type: 1=Type1, 2=Type2 (default: 1).
        """
        self.OutDir = OutDir
        self.CapacityDir = CapacityDir
        self.pga = pga
        self.SoilClass = SoilClass
        self.Nation = Nation
        self.SpectType = SpectrumType

        if isinstance(SPOInfo, list):
            SPOInfo = np.array(SPOInfo)
        if isinstance(SPOCurve, list):
            SPOCurve = np.array(SPOCurve)
            
        if SPOInfo.ndim == 1:
            Masses = [SPOInfo[1]]
            PhisNorm = [1]
        else:
            Masses = SPOInfo[:, 1]
            Phis = SPOInfo[:, -1]
            Zs = SPOInfo[:, 0]
            TopPhi = Phis[np.where(Zs == max(Zs))[0][0]]
            PhisNorm = [P/TopPhi for P in Phis]

        M = np.diag(Masses)
        SDoFMass = PhisNorm @ M @ np.array([1]*len(PhisNorm))
        self.gamma = (SDoFMass)/(PhisNorm @ M @ PhisNorm)
        self.TopDisp = abs(SPOCurve[0, :])
        self.BaseShear = abs(SPOCurve[1, :])
        BaseShear = abs(SPOCurve[1, :])
        self.STopDisp = self.TopDisp/self.gamma
        SBaseShear = BaseShear/(self.gamma)
        self.SBaseShearStar = SBaseShear/(9.81*SDoFMass)  # g
        self.SaStar = max(self.SBaseShearStar)  # g
        self.Tall = np.linspace(0.0001, 6, 1000)
        self.Sae, self.Sde, _ = spectra.EC8(self.pga,
                                            self.SoilClass,
                                            T=self.Tall,
                                            Nation=self.Nation,
                                            SpectTyp=self.SpectType)

    def _WithInfill(self):
        """
        Perform N2 analysis with infill panel effects.
        
        Calculates displacement demand considering infill panel contribution
        and failure modes. Uses collapse states to modify capacity curves.
        """
        InfillCollapseFile = os.path.join(self.OutDir, 'InfillDCR',
                                          'DCR_DS4.csv')
        InfillCollapse = np.loadtxt(InfillCollapseFile)
        InfillDataFile = os.path.join(self.CapacityDir, 'Infills.json')
        with open(InfillDataFile, 'r') as f:
            Infills = json.load(f)

        ZsFull = np.array([Infills[k]['BotZ'] for k in Infills.keys()])
        DirsFull = np.array([Infills[k]['Direction'] for k in Infills.keys()])
        Zs = np.unique(ZsFull)
        Dirs = np.unique(DirsFull)
        InfillFailStep = 1e16
        for Dir in Dirs:
            for Z in Zs:
                idx = np.where((DirsFull == Dir) & (ZsFull == Z))[0]
                CP = abs(InfillCollapse[:, idx]) < 1
                try:
                    CFS = np.where(CP.sum(axis=1) != CP.sum(axis=1)[0])[0][0]
                    InfillFailStep = min(InfillFailStep, CFS)
                    # print(InfillFailStep)
                except IndexError:
                    pass
        FMinIdx = InfillFailStep
        FMin = self.SBaseShearStar[FMinIdx]
        DFMin = self.STopDisp[FMinIdx]
        FMaxIdx = np.argmax(self.SBaseShearStar[:FMinIdx])
        FMax = self.SBaseShearStar[FMaxIdx]
        DFMax = self.STopDisp[FMaxIdx]
        if FMaxIdx < InfillFailStep:
            FMin = min(self.SBaseShearStar[FMaxIdx:InfillFailStep])
        else:
            FMin = self.SBaseShearStar[InfillFailStep]

        FMinIdx = np.where(self.SBaseShearStar == FMin)[0][-1]
        DFMin = self.STopDisp[FMinIdx]

        EhFMax = np.trapz(self.SBaseShearStar[:FMaxIdx],
                          x=self.STopDisp[:FMaxIdx])

        EhFMin = np.trapz(self.SBaseShearStar[:FMinIdx],
                          x=self.STopDisp[:FMinIdx])

        Dy = 2*(DFMax - EhFMax/FMax)

        if FMax == FMin:
            Ds = DFMin
        else:
            Ds = (2*(EhFMin - EhFMax + FMax*DFMax - 0.5*DFMin*(FMax+FMin)) /
                  (FMax-FMin))

        self.IdealF = [0,
                       FMax,
                       FMax,
                       FMin]
        self.IdealD = [0,
                       Dy,
                       Ds,
                       DFMin]

        TStar = 2*np.pi*(Dy/(FMax*9.81))**0.5
        MuS = Ds/Dy
        ru = FMin/FMax

        SaeStar, _, SpectreProps = spectra.EC8(self.pga,
                                               SoilTyp=self.SoilClass,
                                               T=np.asarray([TStar]),
                                               Nation=self.Nation,
                                               SpectTyp=self.SpectType)

        R = SaeStar/FMax

        Tc = SpectreProps['Tc']
        Td = SpectreProps['Td']

        TdStar = Td*((2-ru)**0.5)
        deltaT = (TStar-Tc)/(TdStar-Tc)

        # R(muS)
        if TStar <= Tc:
            RMuS = 0.7*(TStar/Tc)*(MuS-1) + 1
            if R <= RMuS:
                c = 0.7*TStar/Tc
            else:
                c = 0.7*(ru**0.5)*((TStar/Tc)**(1/(ru**0.5)))
        elif TStar <= TdStar:
            RMuS = (0.7+0.3*deltaT)*(MuS-1) + 1
            if R <= RMuS:
                c = 0.7+0.3*deltaT
            else:
                c = 0.7*(ru**0.5)*(1-deltaT)+deltaT
        else:
            RMuS = MuS
            c = 1

        # R0
        if R <= RMuS:
            R0 = 1
            mu0 = 1
        else:
            R0 = RMuS
            mu0 = MuS
        self.mu = (R[0]-R0)/c + mu0
        self.Dm = Dy*self.mu
        self.Sain, self.Sdin = spectra.EC8_N2WithInfill(
            self.pga, self.SoilClass, self.mu, MuS, ru,
            Tp=self.Tall, Nation=self.Nation, SpectTyp=self.SpectType)
        self.DispDemand = self.Dm*self.gamma

    def _WOInfill(self):
        """
        Perform N2 analysis without infill panel effects.
        
        Standard N2 method implementation using only frame capacity.
        Calculates displacement demand through iterative solution.
        """
        def _Emfun(Dm, Displacement, Force):
            """Calculate equivalent elastic energy from capacity curve."""
            dVals = np.linspace(0, Dm, 10000)
            aVals = np.interp(dVals, Displacement, Force)
            Em = np.trapz(aVals, x=dVals)
            return Em

        def AccelerationDiff(TCur, AccelerationCap, mu, pga, SoilTyp):
            """Calculate difference between capacity and demand accel."""
            AccelerationDem, _ = spectra.EC8_N2(pga, SoilTyp,
                                                mu, Tp=np.asarray([TCur]),
                                                Nation=self.Nation,
                                                SpectTyp=self.SpectType)
            return AccelerationCap-AccelerationDem[0]

        def _findDmDemand(DmCapacity, Displacement,
                          Acceleration, pga, SoilTyp):
            """Find displacement demand for given capacity displacement."""
            SaStar = max(Acceleration)
            DyStar = 2*(DmCapacity-(_Emfun(DmCapacity,
                                           Displacement,
                                           Acceleration)/SaStar))
            if DyStar > DmCapacity:
                DyStar = DmCapacity
            mu = DmCapacity/DyStar
            # print(DyStar)
            lowT = 0.0001
            upT = 6
            fixVars = [SaStar, mu, pga, SoilTyp]
            # print('Finding Tp based on Acc Diff == 0g')
            try:
                TpCur = guf.find_zero(AccelerationDiff, fixVars, lowT, upT)
                # print(f'Found {TpCur}')
                _, DmDemand = spectra.EC8_N2(pga, SoilTyp,
                                             mu, Tp=np.asarray([TpCur]),
                                             Nation=self.Nation,
                                             SpectTyp=self.SpectType)
            except ValueError:
                DmDemand = Displacement[-1]
            return DmDemand

        def _OptDmWoInfill(Dm, Displacement, Acceleration, pga, SoilTyp):
            """Optimization function to find equilibrium displacement."""
            OptDm = _findDmDemand(Dm, Displacement,
                                  Acceleration, pga, SoilTyp) - Dm
            return OptDm

        lowDm = 1e-6
        upDm = self.STopDisp[-1]
        fixVars2 = [self.STopDisp, self.SBaseShearStar,
                    self.pga, self.SoilClass]
        try:
            self.Dm = guf.find_zero(_OptDmWoInfill, fixVars2, lowDm, upDm)
        except ValueError:
            self.Dm = upDm
        DyStar = min(2*(self.Dm-(_Emfun(self.Dm,
                                        self.STopDisp,
                                        self.SBaseShearStar)/self.SaStar)),
                     self.Dm)
        self.mu = self.Dm/DyStar
        self.IdealD = [0,
                       DyStar,
                       self.Dm]
        self.IdealF = [0,
                       self.SaStar,
                       self.SaStar]

        self.Sain, self.Sdin = spectra.EC8_N2(self.pga, self.SoilClass,
                                              self.mu, Tp=self.Tall,
                                              Nation=self.Nation,
                                              SpectTyp=self.SpectType)
        self.DispDemand = self.Dm*self.gamma

    def do(self, forceme=0):
        """
        Execute N2 analysis.

        Parameters
        ----------
        forceme : int, optional
            Force analysis mode: 1=without infill, 0=auto-detect (default: 0).
        """
        if forceme == 1:
            self._WOInfill()
        else:
            if os.path.exists(os.path.join(self.OutDir, 'InfillDCR')):
                self._WithInfill()
            else:
                self._WOInfill()

    def plot_it(self, AddName=''):
        """
        Generate N2 analysis visualization plot.

        Parameters
        ----------
        AddName : str, optional
            Additional name suffix for output file (default: '').
        """
        self.StepNum = np.where(self.TopDisp >= self.DispDemand)[0][0]
        txt = (f'Displacement Demand: {round(self.DispDemand,3)} [m] '
               f'Being Reached at Step {self.StepNum}')
        print(txt)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.Sde, self.Sae,
                 c='orange', label='Elastic Demand')
        plt.plot(self.Sdin, self.Sain,
                 label=f'Mu = {round(self.mu,3)}')
        plt.plot(self.IdealD, self.IdealF,
                 lw=3, c='black', label='Idealised')
        plt.plot(self.STopDisp, self.SBaseShearStar,
                 c='blue', label='SDOF-WO/Linearise')
        plt.scatter(self.Dm, self.IdealF[1],
                    c='black')
        plt.legend()
        plt.xlim([0, 1.5*max(self.Sde)])
        plt.ylim([0, 1.5*max(self.Sae)])
        plt.xlabel('Spectral Displacement [m]')
        plt.ylabel('Spectral Acceleration [m/s2]')
        plt.title(txt)
        FigureFile = os.path.join(self.OutDir, f'N2_{AddName}.jpg')
        plt.savefig(FigureFile)
        plt.close()

        # plt.show()
