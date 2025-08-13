import numpy as np
from cad2sees.helpers import units
from scipy.optimize import fsolve
import openseespy.opensees as ops


class MC:
    """
    Moment-curvature analysis class for reinforced concrete sections.

    Provides multiple methods for moment-curvature analysis with different
    levels of sophistication:

    1. fiber_analysis() - Custom fibre-based analysis using Mander models
    2. simple_analysis() - Simplified analytical approach
    3. FiberOPS() - OpenSees-based fibre analysis (requires openseespy)

    Attributes:
        SP: Section properties containing geometry and material data
        NLoad: Applied axial load in consistent units (typically N)
        alpha: Section rotation angle in radians
        numYPoint: Discretisation points in Y direction for fibre analysis
        numZPoint: Discretisation points in Z direction for fibre analysis
    """

    def __init__(self, section_props, n_load, alpha):
        """
        Initialise the moment-curvature analysis class.

        Args:
            section_props: Section properties containing geometry and
                material data including coordinates, reinforcement,
                concrete/steel strengths, and elastic moduli
            n_load: Applied axial load (positive for compression)
            alpha: Section rotation angle in radians
        """
        self.SP = section_props
        self.NLoad = n_load * units.kN
        self.alpha = alpha
        self.numYPoint = 100  # Fiber discretization in Y direction
        self.numZPoint = 100  # Fiber discretization in Z direction

    def _get_section_dict_data(self):
        """
        Extract and convert section properties to consistent units.

        Sets instance attributes for section dimensions, material
        properties, and reinforcement details using the units module.
        """
        self.b = self.SP['b']*units.mm
        self.h = self.SP['h']*units.mm
        self.fc = self.SP['fc0']*units.MPa
        self.fy = self.SP['fy']*units.MPa
        self.cv = self.SP['Cover']*units.mm
        self.dbL = np.mean(self.SP['ReinfL'][:, -1]**2)**0.5*units.mm
        self.dbV = self.SP['phi_T']*units.mm
        self.Es = self.SP['Es']*units.MPa
        self.Ec = self.SP['Ec']*units.MPa

    def fiber_analysis(self, direction):
        """
        Perform fibre-based moment-curvature analysis using Mander models.

        Conducts detailed fibre-based analysis using custom Mander concrete
        models for confined and unconfined concrete with incremental analysis.

        Args:
            direction: Analysis direction ('Positive' or 'Negative')

        Returns:
            tuple: Five-element tuple containing:
                - Mzs: Moments about Z-axis (kN.m)
                - Mys: Moments about Y-axis (kN.m)
                - Curvs: Curvatures (1/m)
                - xMaxs: Maximum rebar strains at each step
                - [eps_y, eps_u]: Steel yield and ultimate strains

        Note:
            Analysis continues until steel fracture or convergence failure.
        """
        def _get_topology():
            """
            Generate fibre topology for the reinforced concrete section.

            Discretises the section into concrete and steel fibres,
            assigns material types, and calculates fibre areas and
            coordinates using Mander model parameters.
            """
            """
            Generate fiber topology for the reinforced concrete section.

            This function discretizes the section into concrete and steel
            fibers, assigns material types, and calculates fiber areas and
            coordinates. Sets confined concrete properties using Mander model
            parameters.
            """
            # Assuming Rectangular Sections
            # As Built Section Dimensions
            self.b = (max(self.SP['Coords'][:, 0]) -
                      min(self.SP['Coords'][:, 0]))
            self.bc = self.b - self.SP['Cover']
            self.bc2 = self.b - 2*self.SP['Cover']
            self.h = (max(self.SP['Coords'][:, 1]) -
                      min(self.SP['Coords'][:, 1]))
            self.hc = self.h - self.SP['Cover']

            # As Built Section Transversal Rebars
            self.dbV = self.SP['phi_T']
            self.YdirStr = self.SP['NumofStrYDir']
            self.ZdirStr = self.SP['NumofStrZDir']
            self.s = self.SP['s']
            self.s_ = self.s-self.dbV
            AsT = 0.25*np.pi*(self.dbV**2)
            self.AsZ = AsT*self.YdirStr
            self.AsY = AsT*self.ZdirStr
            self.rho_Z = self.AsZ/(self.s*(self.bc))
            self.rho_Y = self.AsY/(self.s*(self.hc))
            self.rho_Tm = 0.5*(self.rho_Z + self.rho_Y)

            # As Built Section Longitidunal Rebars
            self.reinfL = self.SP['ReinfL'][:, 2]
            self.AsL = 0.25*np.pi*np.sum(self.reinfL**2)
            self.dbL = (self.AsL/len(self.reinfL))**0.5
            self.rho_L = self.AsL/(self.bc*self.hc)

            # As Built Matrial Props
            # Concrete
            self.fc0 = self.SP['fc0']
            self.Ec = self.SP['Ec']

            # Steel
            self.eps_y = self.SP['eps_y']
            self.eps_u = self.SP['eps_u']
            self.fy = self.SP['fy']
            self.fu = self.SP['fu']
            self.Es = self.fy/self.eps_y
            self.fyw = self.SP['fyw']

            # Section Coordinate Arrangement - work with copies to avoid
            # modifying original
            self.SecCenter = self.SP['Coords'].mean(axis=0)
            # Create working copies for this analysis
            self.coords_centered = self.SP['Coords'].copy()
            self.reinf_centered = self.SP['ReinfL'].copy()
            self.coords_centered[:, 0] -= self.SecCenter[0]
            self.coords_centered[:, 1] -= self.SecCenter[1]
            self.reinf_centered[:, 0] -= self.SecCenter[0]
            self.reinf_centered[:, 1] -= self.SecCenter[1]

            # Generation of mesh data
            SecZMin = min(self.coords_centered[:, 0])
            SecZMax = max(self.coords_centered[:, 0])
            SecYMin = min(self.coords_centered[:, 1])
            SecYMax = max(self.coords_centered[:, 1])

            CoreZMin = SecZMin + self.SP['Cover']
            CoreZMax = SecZMax - self.SP['Cover']
            CoreYMin = SecYMin + self.SP['Cover']
            CoreYMax = SecYMax - self.SP['Cover']

            FiberYLength = (SecYMax-SecYMin)/self.numYPoint
            FiberZLength = (SecZMax-SecZMin)/self.numZPoint

            # LeftCover
            # Y
            yyLR = np.linspace(SecYMin+0.5, SecYMax-0.5, self.numYPoint,
                               endpoint=True)
            # Z
            zzLC = np.array([(SecZMin + CoreZMin)*0.5])

            zvLC, yvLC = np.meshgrid(zzLC, yyLR, indexing='xy')
            zLC = zvLC.reshape(zvLC.shape[0]*zvLC.shape[1])
            yLC = yvLC.reshape(yvLC.shape[0]*yvLC.shape[1])
            FiberAreaLR = self.SP['Cover']*FiberYLength
            AreaLC = np.full(len(yLC), FiberAreaLR)

            # RightCover
            # Z
            zzRC = np.array([(SecZMax + CoreZMax)*0.5])

            zvRC, yvRC = np.meshgrid(zzRC, yyLR, indexing='xy')
            zRC = zvRC.reshape(zvRC.shape[0]*zvRC.shape[1])
            yRC = yvRC.reshape(yvRC.shape[0]*yvRC.shape[1])
            AreaRC = np.full(len(yRC), FiberAreaLR)
            # TopCover
            # Y
            yyTC = np.array([(SecYMax + CoreYMax)*0.5])
            # Z
            zzTB = np.linspace(CoreZMin+0.5, CoreZMax-0.5, self.numZPoint-2,
                               endpoint=True)

            zvTC, yvTC = np.meshgrid(zzTB, yyTC, indexing='xy')
            zTC = zvTC.reshape(zvTC.shape[0]*zvTC.shape[1])
            yTC = yvTC.reshape(yvTC.shape[0]*yvTC.shape[1])
            FiberAreaTB = self.SP['Cover']*FiberZLength
            AreaTC = np.full(len(yTC), FiberAreaTB)

            # BottomCover
            # Y
            yyBC = np.array([(SecYMin + CoreYMin)*0.5])

            zvBC, yvBC = np.meshgrid(zzTB, yyBC, indexing='xy')
            zBC = zvBC.reshape(zvBC.shape[0]*zvBC.shape[1])
            yBC = yvBC.reshape(yvBC.shape[0]*yvBC.shape[1])
            AreaBC = np.full(len(yBC), FiberAreaTB)

            # Core
            yyCore = np.linspace(CoreYMin+0.5, CoreYMax-0.5, self.numYPoint,
                                 endpoint=True)

            zzCore = np.linspace(CoreZMin+0.5, CoreZMax-0.5, self.numZPoint,
                                 endpoint=True)

            zvCore, yvCore = np.meshgrid(zzCore, yyCore, indexing='xy')
            zCore = zvCore.reshape(zvCore.shape[0]*zvCore.shape[1])
            yCore = yvCore.reshape(yvCore.shape[0]*yvCore.shape[1])
            FiberAreaCore = FiberYLength*FiberZLength
            AreaCore = np.full(len(yCore), FiberAreaCore)

            self.z = np.concatenate((zCore, zLC, zRC, zBC, zTC))
            self.y = np.concatenate((yCore, yLC, yRC, yBC, yTC))
            self.Area = np.concatenate((AreaCore,
                                        AreaLC, AreaRC,
                                        AreaBC, AreaTC))

            # All Section Unconfined Concrete
            self.mat = np.full(len(self.z), 100)
            # Locate Reinforcement
            for Ri in self.reinf_centered:
                self.z = np.append(self.z, Ri[0])
                self.y = np.append(self.y, Ri[1])
                BarArea = 0.25*(Ri[-1]**2)*np.pi
                self.Area = np.append(self.Area, BarArea)
                self.mat = np.append(self.mat, 106)

            zreinf = self.z[self.mat == 106]
            yreinf = self.y[self.mat == 106]
            zreinfmax = max(zreinf)
            zreinfmin = min(zreinf)
            yreinfmax = max(yreinf)
            yreinfmin = min(yreinf)
            maskz = (self.z <= zreinfmax) & (self.z >= zreinfmin)
            masky = (self.y <= yreinfmax) & (self.y >= yreinfmin)
            maskr = (self.mat != 106)
            ABConfMask = maskz & masky & maskr
            self.mat[ABConfMask] = 101

        def _find_epsilon(x, y, NA, alpha, Curvature):
            xy = np.vstack((x, y))
            # Alpha is already in radians, use directly for trigonometric
            # functions
            RotXY = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha), np.cos(alpha)]])
            xynew = RotXY@xy
            xnew = xynew[0, :]
            ynew = xynew[1, :]  # Fixed: was xynew[0, :] in reference
            ynew += NA
            znew = ynew*Curvature
            return xnew, ynew, znew

        def _unconfined_mander(xcur, fc0=self.SP['fc0'],
                               Ec=self.SP['Ec']):
            eps_c0 = 0.002
            eps_sp = 0.005
            x = xcur/eps_c0
            Esec = fc0/eps_c0
            r = Ec/(Ec-Esec)
            fc_end = (fc0*2*r)/(r-1+(2**r))
            mask1 = (xcur >= 0)
            mask2 = (xcur >= -2*eps_c0) & ~mask1
            mask3 = (xcur >= -eps_sp) & ~mask2 & ~mask1
            fc = np.zeros(xcur.shape)
            fc[mask1] = 0
            fc[mask2] = (fc0*(-x[mask2])*r)/(r-1+((-x[mask2])**r))
            fc[mask3] = np.maximum(fc_end - (-xcur[mask3]-0.004)*(fc_end) /
                                   (eps_sp-0.004), 0.0)
            return fc

        def _confined_mander(xcur):
            eps_c0 = 0.002

            rebarsZ = self.reinf_centered[:, 0]
            rebarsY = self.reinf_centered[:, 1]

            botBarMap = (rebarsZ == min(rebarsZ))
            topBarMap = (rebarsZ == max(rebarsZ))
            leftBarMap = (rebarsY == min(rebarsY))
            rightBarMap = (rebarsY == max(rebarsY))

            # botBars
            Aineff = 0
            botBars = self.SP['ReinfL'][botBarMap, :]
            topBars = self.SP['ReinfL'][topBarMap, :]
            BarsTopBot = [botBars, topBars]
            leftBars = self.SP['ReinfL'][leftBarMap, :]
            rightBars = self.SP['ReinfL'][rightBarMap, :]
            BarsRightLeft = [rightBars, leftBars]

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

            kp = (1-(Aineff/(self.bc*self.hc)))
            kv = (1-0.5*self.s_/(self.hc))*(1-0.5*self.s_/self.bc)
            ke = kp*kv

            if self.SP['Hook'] != 135:
                ke *= 0.3

            f_l = ke*0.5*(self.rho_Z+self.rho_Y)*self.fyw
            f_cc = (self.fc0 *
                    (-1.254 +
                     2.254*((1+7.94*f_l/self.fc0)**0.5) -
                     2*f_l/self.fc0))
            eps_cc = eps_c0*(1+5*(f_cc/self.fc0-1))
            x = xcur/eps_cc
            Esec = f_cc/eps_cc
            r = self.Ec/(self.Ec-Esec)
            fc = np.zeros(x.shape)
            self.SP['eps_cu'] = (0.004 +
                                 (1.4*0.5*(self.rho_Y+self.rho_Z) *
                                  self.fy*self.eps_u/f_cc))
            mask1 = (xcur >= 0)
            mask3 = xcur < -self.SP['eps_cu']
            mask2 = (~mask1) & (~mask3)
            fc[mask1] = 0
            fc[mask3] = 0
            fc[mask2] = (f_cc*(-x[mask2])*r)/(r-1+((-x[mask2])**r))

            return fc

        def _steel(xcur, Type='Bi-Linear'):
            if Type == 'Bi-Linear':
                Es2 = (self.fu-self.fy)/(self.eps_u-self.eps_y)
                mask1 = (xcur <= -self.eps_u)
                mask2 = (xcur <= -self.eps_y) & ~mask1
                mask3 = (xcur <= self.eps_y) & ~mask2
                mask4 = (xcur <= self.eps_u) & ~mask3
                mask5 = xcur > self.eps_u

                fs = np.zeros(xcur.shape)
                fs[mask1] = 0.01*self.fy  # Residual compression stress
                # Compression hardening
                fs[mask2] = self.fu-Es2*(xcur[mask2]+self.eps_u)
                # Linear elastic range
                fs[mask3] = self.fy-self.Es*(xcur[mask3]+self.eps_y)
                # Tension hardening
                fs[mask4] = -self.fy-Es2*(xcur[mask4]-self.eps_y)
                fs[mask5] = -0.01*self.fy  # Residual tension stress
                # print(max(abs(fs)))
            else:
                # TODO Add other type of steel materials
                raise Exception('Undefined Steel')
            return fs

        def _current_ss(NA, Curvature, mat, z, y, alpha):
            _, _, x = _find_epsilon(z, y, NA, alpha, Curvature)
            f = np.zeros(x.shape)
            for mati in np.unique(mat):
                # xcur = -x[mat == mati]
                xcur = x[mat == mati]
                if np.any(np.isnan(xcur)):
                    print(xcur)
                if mati == 100:
                    fcur = _unconfined_mander(xcur, fc0=self.fc0, Ec=self.Ec)
                elif mati == 101:
                    fcur = _confined_mander(xcur)
                elif mati == 106:
                    fcur = _steel(xcur)
                    if ((max(abs(xcur)) >= abs(self.eps_y)) and
                       (self.SteelYeild == 99)):
                        self.SteelYeild = Curvature
                    if ((max(abs(xcur)) >= abs(self.eps_u)) and
                       (self.SteelFracture == 99)):
                        self.SteelFracture = Curvature
                f[mat == mati] = fcur
            return f, x

        def _balance(NA, Curvature, NLoad, mat, z, y, Area, alpha):
            # if np.isnan(NA):
            #     print('NAN')
            f, _ = _current_ss(NA, Curvature, mat, z, y, alpha)
            Pcur = np.sum(f*Area)
            # Following reference code equilibrium: NLoad - Pcur = 0
            # This matches reference: return sumF - P
            Rcur = NLoad - Pcur  # Corrected to match reference
            if np.isnan(Rcur):
                Rcur = 1e18
            return Rcur

        def _calculate_M(Direct):
            Mzs = []
            Mys = []
            Curvs = []
            xMaxs = []
            i = 0
            ErrFlag = 0
            if Direct == 'Positive':
                deltaCurv = 1e-7
            elif Direct == 'Negative':
                deltaCurv = -1e-7
            # Initialize neutral axis to mid-depth of section
            # for better convergence
            section_depth = max(self.y) - min(self.y)
            NA = section_depth / 2  # Better initial guess
            self.SteelYeild = 99
            self.SteelFracture = 99
            self.ConcCrush = 99
            Curvature = 0
            while ErrFlag < 20 and i < 10000:
                i += 1
                Curvature = deltaCurv*i
                try:
                    NA, info, ier, msg = fsolve(_balance, NA,
                                                args=(Curvature, self.NLoad,
                                                      self.mat, self.z, self.y,
                                                      self.Area, self.alpha),
                                                full_output=True,
                                                xtol=0.1)
                    if ier != 1:
                        print(f"Message: {msg} \n Info: {info}")
                    if np.isnan(NA):
                        print('NAN')
                    f, x = _current_ss(NA, Curvature, self.mat,
                                       self.z, self.y, self.alpha)
                    rebarMap = (self.mat == 106)
                    xcurMax = max(abs(x[rebarMap]))
                    xMaxs.append(xcurMax)
                    # Calculate moments including P-delta effects
                    # (Following reference implementation pattern)
                    Mz = (np.sum(f*self.Area*self.y)*10**(-6) -
                          np.cos(self.alpha)*NA *
                          self.NLoad*(10**(-3)))  # kN.m
                    My = (np.sum(f*self.Area*self.z)*10**(-6) -
                          np.sin(self.alpha)*NA *
                          self.NLoad*(10**(-3)))  # kN.m
                    Mzs.append(-Mz)
                    Mys.append(-My)
                    Curvs.append(Curvature*1000)
                    if xcurMax >= self.eps_u:
                        ErrFlag = 50
                except Exception as e:
                    ErrFlag += 1
                    print(f'ERROR Number {ErrFlag}: {e}')
            return Mzs, Mys, Curvs, xMaxs
        _get_topology()
        Mzs, Mys, Curvs, xMaxs = _calculate_M(direction)
        return Mzs, Mys, Curvs, xMaxs, [self.eps_y, self.eps_u]

    def simple_analysis(self):
        """
        Perform simplified moment-curvature analysis.

        Uses simplified analytical expressions to estimate moment-curvature
        behaviour for both biaxial bending directions as a faster alternative
        for initial design estimates.

        Returns:
            tuple: Four moment capacities:
                - MpZ: Positive moment capacity about Z-axis (kN.m)
                - MnZ: Negative moment capacity about Z-axis (kN.m)
                - MpY: Positive moment capacity about Y-axis (kN.m)
                - MnY: Negative moment capacity about Y-axis (kN.m)
        """
        self._get_section_dict_data()
        TopSecReifLMap = self.SP['ReinfL'][:, 1] >= self.SP['h']/6
        rhoTop = (0.25*units.pi *
                  np.sum((self.SP['ReinfL'][TopSecReifLMap, -1] *
                          units.mm)**2)/(self.b*self.h))

        BotSecReifLMap = self.SP['ReinfL'][:, 1] <= self.SP['h']/6
        rhoBot = (0.25*units.pi *
                  np.sum((self.SP['ReinfL'][BotSecReifLMap, -1] *
                          units.mm)**2)/(self.b*self.h))

        MidSecReinfLMap = (~BotSecReifLMap & ~TopSecReifLMap)
        rhoMid = (0.25*units.pi *
                  np.sum((self.SP['ReinfL'][MidSecReinfLMap, -1] *
                          units.mm)**2)/(self.b*self.h))

        # n term for concrete
        n_c = 0.8+self.fc/(18*units.MPa)
        # epsilon_c' for concrete
        e_c = 1.0*self.fc/self.Ec*(n_c/(n_c-1))
        # Steel yield strain
        e_s = self.fy/self.Es
        # Yield Curvature
        phiY = 2.1*(self.fy/self.Es)/self.h

        # Do a Moment Curvature Analysis
        # Depth to Top Bars
        d1 = self.cv+self.dbV+self.dbL*0.5
        # Depth to Middle Bars
        d2 = 0.5*self.h
        # Depth to Bottom Bars
        d3 = self.h-self.cv-self.dbV-self.dbL*0.5

        # Possitive Bending
        # initial trial of NA depth
        c = self.h*0.5
        count = 0
        err = 0.5

        while err > 0.001 and count < 1000:
            # Strain in top steel (in strains)
            e_s1 = (c-d1)*phiY
            # Strain in middle steel
            e_s2 = (d2-c)*phiY
            # Strain in bottom steel
            e_s3 = (d3-c)*phiY
            # Strain in top of section
            e_top = c*phiY

            # Steel Related
            if e_s1 < e_s:
                f_s1 = e_s1*self.Es
            else:
                f_s1 = self.fy

            if e_s2 < e_s:
                f_s2 = e_s2*self.Es
            else:
                f_s2 = self.fy

            if e_s3 < e_s:
                f_s3 = e_s3*self.Es
            else:
                f_s3 = self.fy

            Fs1 = f_s1*rhoTop*self.b*d3
            Fs2 = f_s2*rhoMid*self.b*d3
            Fs3 = f_s3*rhoBot*self.b*d3

            # Concrete Related
            # alpha1beta1 term
            a1b1 = (e_top/e_c) - ((e_top/e_c)**2)/3
            # beta1
            b1 = (4-(e_top/e_c))/(6-2*(e_top/e_c))
            # Concrete block force
            Fc = a1b1*c*self.fc*self.b

            Psec = self.NLoad+Fs2+Fs3-Fc-Fs1
            # Adjust NA depth to balance section forces
            if Psec < 0:
                c -= 0.001
            elif Psec > 0:
                c += 0.001
            err = abs(Psec)

            if err < 5:
                break
            count += 1

        # Compute the moment
        # Positive moment at yield
        MpZ = (self.NLoad*(0.5*self.h-c) +
               Fs1*(c-d1) +
               Fs3*(d3-c) +
               Fs2*(d2-c) +
               Fc*c*(1-b1*0.5))

        # Negative Bending
        c = self.h*0.5 				        # initial trial of NA depth
        count = 0
        err = 0.5
        while err > 0.001 and count < 1000:
            e_s1 = (c-d1)*phiY          # Strain in top steel (in strains)
            e_s2 = (d2-c)*phiY          # Strain in middle steel
            e_s3 = (d3-c)*phiY          # Strain in bottom steel
            e_top = c*phiY              # Strain in top of section

            # Steel Related
            if e_s1 < e_s:
                f_s1 = e_s1*self.Es
            else:
                f_s1 = self.fy
            if e_s2 < e_s:
                f_s2 = e_s2*self.Es
            else:
                f_s2 = self.fy
            if e_s3 < e_s:
                f_s3 = e_s3*self.Es
            else:
                f_s3 = self.fy

            Fs1 = f_s1*rhoBot*self.b*d3
            Fs2 = f_s2*rhoMid*self.b*d3
            Fs3 = f_s3*rhoTop*self.b*d3

            # Concrete Related
            # alpha1beta1 term
            a1b1 = (e_top/e_c) - ((e_top/e_c)**2)/3
            # beta1
            b1 = (4-(e_top/e_c))/(6-2*(e_top/e_c))
            # Concrete block force
            Fc = a1b1*c*self.fc*self.b
            # Section Force
            Psec = self.NLoad + Fs2 + Fs3 - Fc - Fs1

            # Adjust NA depth to balance section forces
            if Psec < 0:
                c -= 0.001
            elif Psec > 0:
                c += 0.001
            err = abs(Psec)
            if err < 5:
                break
            count += 1
        # Compute the moment
        MnZ = (self.NLoad*(0.5*self.h-c) +
               Fs1*(c-d1) +
               Fs3*(d3-c) +
               Fs2*(d2-c) +
               Fc*c*(1-b1*0.5))

        RightSecReifLMap = self.SP['ReinfL'][:, 0] >= self.SP['b']/6
        rhoRight = (0.25*units.pi *
                    np.sum((self.SP['ReinfL'][RightSecReifLMap, -1] *
                            units.mm)**2)/(self.b*self.h))

        LeftSecReifLMap = self.SP['ReinfL'][:, 0] <= self.SP['b']/6
        rhoLeft = (0.25*units.pi *
                   np.sum((self.SP['ReinfL'][LeftSecReifLMap, -1] *
                           units.mm)**2)/(self.b*self.h))

        MidSecReinfLMap = (~LeftSecReifLMap & ~RightSecReifLMap)
        rhoMid = (0.25*units.pi *
                  np.sum((self.SP['ReinfL'][MidSecReinfLMap, -1] *
                          units.mm)**2)/(self.b*self.h))

        # n term for concrete
        n_c = 0.8+self.fc/(18*units.MPa)
        # epsilon_c' for concrete
        e_c = 1.0*self.fc/self.Ec*(n_c/(n_c-1))
        # Steel yield strain
        e_s = self.fy/self.Es
        # Yield Curvature
        phiY = 2.1*(self.fy/self.Es)/self.b

        # Do a Moment Curvature Analysis
        # Depth to Top Bars
        d1 = self.cv+self.dbV+self.dbL*0.5
        # Depth to Middle Bars
        d2 = 0.5*self.b
        # Depth to Bottom Bars
        d3 = self.b-self.cv-self.dbV-self.dbL*0.5

        # Possitive Bending
        # initial trial of NA depth
        c = self.b*0.5
        count = 0
        err = 0.5

        while err > 0.001 and count < 1000:
            # Strain in top steel (in strains)
            e_s1 = (c-d1)*phiY
            # Strain in middle steel
            e_s2 = (d2-c)*phiY
            # Strain in bottom steel
            e_s3 = (d3-c)*phiY
            # Strain in top of section
            e_top = c*phiY

            # Steel Related
            if e_s1 < e_s:
                f_s1 = e_s1*self.Es
            else:
                f_s1 = self.fy

            if e_s2 < e_s:
                f_s2 = e_s2*self.Es
            else:
                f_s2 = self.fy

            if e_s3 < e_s:
                f_s3 = e_s3*self.Es
            else:
                f_s3 = self.fy

            Fs1 = f_s1*rhoLeft*self.h*d3
            Fs2 = f_s2*rhoMid*self.h*d3
            Fs3 = f_s3*rhoRight*self.h*d3

            # Concrete Related
            # alpha1beta1 term
            a1b1 = (e_top/e_c) - ((e_top/e_c)**2)/3
            # beta1
            b1 = (4-(e_top/e_c))/(6-2*(e_top/e_c))
            # Concrete block force
            Fc = a1b1*c*self.fc*self.h

            Psec = self.NLoad+Fs2+Fs3-Fc-Fs1
            # Adjust NA depth to balance section forces
            if Psec < 0:
                c -= 0.001
            elif Psec > 0:
                c += 0.001
            err = abs(Psec)

            if err < 5:
                break
            count += 1

        # Compute the moment
        # Positive moment at yield
        MpY = (self.NLoad*(0.5*self.b-c) +
               Fs1*(c-d1) +
               Fs3*(d3-c) +
               Fs2*(d2-c) +
               Fc*c*(1-b1*0.5))

        # Negative Bending
        c = self.b*0.5 				        # initial trial of NA depth
        count = 0
        err = 0.5
        while err > 0.001 and count < 1000:
            e_s1 = (c-d1)*phiY          # Strain in top steel (in strains)
            e_s2 = (d2-c)*phiY          # Strain in middle steel
            e_s3 = (d3-c)*phiY          # Strain in bottom steel
            e_top = c*phiY              # Strain in top of section

            # Steel Related
            if e_s1 < e_s:
                f_s1 = e_s1*self.Es
            else:
                f_s1 = self.fy
            if e_s2 < e_s:
                f_s2 = e_s2*self.Es
            else:
                f_s2 = self.fy
            if e_s3 < e_s:
                f_s3 = e_s3*self.Es
            else:
                f_s3 = self.fy

            Fs1 = f_s1*rhoRight*self.b*d3
            Fs2 = f_s2*rhoMid*self.b*d3
            Fs3 = f_s3*rhoLeft*self.b*d3

            # Concrete Related
            # alpha1beta1 term
            a1b1 = (e_top/e_c) - ((e_top/e_c)**2)/3
            # beta1
            b1 = (4-(e_top/e_c))/(6-2*(e_top/e_c))
            # Concrete block force
            Fc = a1b1*c*self.fc*self.h
            # Section Force
            Psec = self.NLoad + Fs2 + Fs3 - Fc - Fs1

            # Adjust NA depth to balance section forces
            if Psec < 0:
                c -= 0.001
            elif Psec > 0:
                c += 0.001
            err = abs(Psec)
            if err < 5:
                break
            count += 1
        # Compute the moment
        MnY = (self.NLoad*(0.5*self.b-c) +
               Fs1*(c-d1) +
               Fs3*(d3-c) +
               Fs2*(d2-c) +
               Fc*c*(1-b1*0.5))

        return MpZ, MnZ, MpY, MnY

    def FiberOPS(self, Direction, axialLoad=None, dK=None):
        """
        OpenSees-based moment-curvature analysis with subprocess isolation.

        Performs moment-curvature analysis using OpenSees in a separate
        subprocess to prevent interference between concurrent tasks.

        Args:
            Direction: Analysis direction ('Positive' or 'Negative')
            axialLoad: Axial load in kN (optional, uses self.NLoad if None)
            dK: Curvature increment for analysis (optional)

        Returns:
            tuple: Five-element tuple containing:
                - Mzs: Global moments about Z-axis (kN.m)
                - Mys: Global moments about Y-axis (kN.m)
                - Curvs: Curvatures about principal axis (1/m)
                - xMaxs: Maximum rebar strains at each step
                - [eps_y, eps_u]: Steel yield and ultimate strains

        Note:
            Uses subprocess to isolate OpenSees execution.
        """
        import subprocess
        import json
        import os
        import sys
        import tempfile

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            """Recursively convert numpy arrays to lists for JSON."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value)
                        for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        # Prepare section properties for JSON serialization
        sp_json_ready = convert_for_json(self.SP)
        
        # Use provided axial load or default to instance axial load
        if axialLoad is None:
            axialLoad = self.NLoad / units.kN

        if isinstance(axialLoad, (list, np.ndarray)):
            axialLoad = axialLoad[0]
        
        # Use default delta_k if not provided
        if dK is None:
            dK = 1e-7

        # Get the path to mc_ops.py in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mc_ops_path = os.path.join(current_dir, "mc_ops.py")
        
        mc_ops_data = {
            "analysisType": "fiber_ops",
            "SP": sp_json_ready,
            "alpha": self.alpha,
            "Direction": Direction,
            "axialLoad": axialLoad,
            "dK": dK,
            "numYPoint": self.numYPoint,
            "numZPoint": self.numZPoint
        }

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
            json.dump(mc_ops_data, temp_file)
            temp_file_path = temp_file.name

        result = subprocess.run(
            [sys.executable, mc_ops_path, temp_file_path],
            capture_output=True, text=True
        )

        results_dict = json.loads(result.stdout)
        Mzs = results_dict['Mzs']
        Mys = results_dict['Mys']
        Curvs = results_dict['Curvs']
        xMaxs = results_dict['xMaxs']
        limits = results_dict['limits']
        return Mzs, Mys, Curvs, xMaxs, limits

    def FiberOPS2(self, Direction, axialLoad=None, dK=None):
        """
        OpenSees-based biaxial moment-curvature analysis.

        Performs biaxial moment-curvature analysis using OpenSees with
        thread-based isolation to prevent interference between tasks.

        Args:
            Direction: Analysis direction ('Positive' or 'Negative')
            axialLoad: Axial load in kN (optional, uses self.NLoad if None)
            dK: Curvature increment for analysis (optional)

        Returns:
            tuple: Six-element tuple containing:
                - Mzs: Moments about Z-axis (kN.m)
                - Mys: Moments about Y-axis (kN.m)
                - Curvzs: Curvatures about Z-axis (1/m)
                - Curvys: Curvatures about Y-axis (1/m)
                - xMaxs: Maximum rebar strains at each step
                - [eps_y, eps_u]: Steel yield and ultimate strains

        Note:
            Uses threading to isolate OpenSees execution.
        """
        import subprocess
        import json
        import os
        import sys
        import tempfile

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            """Recursively convert numpy arrays to lists for JSON."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_for_json(value)
                        for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj
        
        # Prepare section properties for JSON serialization
        sp_json_ready = convert_for_json(self.SP)
        
        # Use provided axial load or default to instance axial load
        if axialLoad is None:
            axialLoad = self.NLoad / units.kN

        if isinstance(axialLoad, (list, np.ndarray)):
            axialLoad = axialLoad[0]
        
        # Use default delta_k if not provided
        if dK is None:
            dK = 1e-7
        
        # Get the path to mc_ops.py in the same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mc_ops_path = os.path.join(current_dir, "mc_ops.py")
        
        mc_ops_data = {
            "analysisType": "fiber_ops_biax",
            "SP": sp_json_ready,
            "alpha": self.alpha,
            "Direction": Direction,
            "axialLoad": axialLoad,
            "dK": dK,
            "numYPoint": self.numYPoint,
            "numZPoint": self.numZPoint
        }

        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json') as temp_file:
            json.dump(mc_ops_data, temp_file)
            temp_file_path = temp_file.name

        result = subprocess.run(
            [sys.executable, mc_ops_path, temp_file_path],
            capture_output=True, text=True
        )

        results_dict = json.loads(result.stdout)
        Mzs = results_dict['Mzs']
        Mys = results_dict['Mys']
        Curvzs = results_dict['Curvzs']
        Curvys = results_dict['Curvys']
        xMaxs = results_dict['xMaxs']
        strain_limits = results_dict['limits']
        return Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits

    # Backward compatibility methods for non-OpenSees analyses
    def Fiber(self, Direction):
        """Backward compatibility wrapper for fiber_analysis."""
        return self.fiber_analysis(Direction)

    def Simple(self):
        """Backward compatibility wrapper for simple_analysis."""
        return self.simple_analysis()
