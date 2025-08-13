"""
Response Spectra Module

Generate response spectra for seismic analysis according to EC8 standards.
Includes standard and N2 method implementations with optional infill effects.

Functions:
    EC8: Standard EC8 response spectrum generation
    EC8_N2WithInfill: N2 method response spectrum for frames with infills
    EC8_N2: N2 method response spectrum for bare frames
"""

import numpy as np


def EC8(ag, SoilTyp, Nation='', eta=1, SpectTyp=1,
        T=np.linspace(0, 3.999, 1000)):
    """
    Generate EC8 response spectrum for seismic analysis.

    Parameters
    ----------
    ag : float
        Design ground acceleration.
    SoilTyp : str
        Soil type classification ('A', 'B', 'C', 'D', 'E').
    Nation : str, optional
        National annex ('Portugal' or '') (default: '').
    eta : float, optional
        Damping correction factor (default: 1).
    SpectTyp : int, optional
        Spectrum type: 1=Type1, 2=Type2 (default: 1).
    T : array_like, optional
        Period array (default: linspace(0, 3.999, 1000)).

    Returns
    -------
    tuple
        (Sae, Sde, SpectreProps) - Acceleration spectrum, displacement
        spectrum, and spectrum properties dictionary.
    """
    if Nation == '':
        if SpectTyp == 1:
            if SoilTyp == 'A':
                S = 1.0
                Tb = 0.15
                Tc = 0.4
                Td = 2.0
                Te = 4.5
                Tf = 10.0
            elif SoilTyp == 'B':
                S = 1.2
                Tb = 0.15
                Tc = 0.5
                Td = 2.0
                Te = 5.0
                Tf = 10.0
            elif SoilTyp == 'C':
                S = 1.15
                Tb = 0.2
                Tc = 0.6
                Td = 2.0
                Te = 6.0
                Tf = 10.0
            elif SoilTyp == 'D':
                S = 1.35
                Tb = 0.2
                Tc = 0.8
                Td = 2.0
                Te = 6.0
                Tf = 10.0
            elif SoilTyp == 'E':
                S = 1.4
                Tb = 0.15
                Tc = 0.5
                Td = 2.0
                Te = 6.0
                Tf = 10.0
        elif SpectTyp == 2:
            if SoilTyp == 'A':
                S = 1.0
                Tb = 0.05
                Tc = 0.25
                Td = 1.2
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'B':
                S = 1.35
                Tb = 0.05
                Tc = 0.25
                Td = 1.2
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'C':
                S = 1.5
                Tb = 0.10
                Tc = 0.25
                Td = 1.2
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'D':
                S = 1.8
                Tb = 0.10
                Tc = 0.30
                Td = 1.2
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'E':
                S = 1.6
                Tb = 0.05
                Tc = 0.25
                Td = 1.2
                Te = 99.0
                Tf = 100.0
    elif Nation == 'Portugal':
        if SpectTyp == 1:
            if SoilTyp == 'A':
                SMax = 1.0
                Tb = 0.1
                Tc = 0.6
                Td = 2.0
                Te = 4.5
                Tf = 10.0
            elif SoilTyp == 'B':
                SMax = 1.35
                Tb = 0.1
                Tc = 0.6
                Td = 2.0
                Te = 5.0
                Tf = 10.0
            elif SoilTyp == 'C':
                SMax = 1.6
                Tb = 0.1
                Tc = 0.6
                Td = 2.0
                Te = 6.0
                Tf = 10.0
            elif SoilTyp == 'D':
                SMax = 2.0
                Tb = 0.1
                Tc = 0.8
                Td = 2.0
                Te = 6.0
                Tf = 10.0
            elif SoilTyp == 'E':
                SMax = 1.8
                Tb = 0.1
                Tc = 0.6
                Td = 2.0
                Te = 6.0
                Tf = 10.0
        elif SpectTyp == 2:
            if SoilTyp == 'A':
                SMax = 1.0
                Tb = 0.1
                Tc = 0.25
                Td = 2.0
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'B':
                SMax = 1.35
                Tb = 0.1
                Tc = 0.25
                Td = 2.0
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'C':
                SMax = 1.6
                Tb = 0.10
                Tc = 0.25
                Td = 2.0
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'D':
                SMax = 2.0
                Tb = 0.10
                Tc = 0.30
                Td = 2.0
                Te = 99.0
                Tf = 100.0
            elif SoilTyp == 'E':
                SMax = 1.8
                Tb = 0.1
                Tc = 0.25
                Td = 2.0
                Te = 99.0
                Tf = 100.0
        if ag <= 1/9.81:
            S = SMax
        elif ag <= 4/9.81:
            S = SMax - (SMax-1)*(ag-(1/9.81))/3
        else:
            S = 1.0
    dg = 0.025*ag*S*Tc*Td
    Sae = np.piecewise(T,
                       [(T >= 0) & (Tb >= T),
                        (T > Tb) & (Tc >= T),
                        (T > Tc) & (Td >= T),
                        (T > Td) & (Te >= T),
                        (T > Te) & (Tf >= T),
                        (T > Tf)],
                       [lambda T: ag*S*(1+(T/Tb)*(eta*2.5-1)),
                        lambda T: ag*S*eta*2.5,
                        lambda T: ag*S*eta*2.5*Tc/T,
                        lambda T: ag*S*eta*2.5*Tc*Td/(T**2),
                        lambda T: ((2*np.pi/T)**2)*dg*(
                            2.5*eta + ((T-Te)/(Tf-Te))*(1-2.5*eta)),
                        lambda T: ((2*np.pi/T)**2)*dg
                        ])
    Sde = Sae*9.81*(0.5*T/np.pi)**2
    SpectreProps = {'S': S,
                    'Tb': Tb,
                    'Tc': Tc,
                    'Td': Td}
    return Sae, Sde, SpectreProps


def EC8_N2WithInfill(ag, SoilTyp, mu, MuS, ru, Nation='', eta=1, SpectTyp=1,
                     Tp=np.linspace(0.001, 3.999, 1000)):
    """
    Generate N2 method response spectrum with infill panel effects.

    Parameters
    ----------
    ag : float
        Design ground acceleration.
    SoilTyp : str
        Soil type classification ('A', 'B', 'C', 'D', 'E').
    mu : float
        Target ductility factor.
    MuS : float
        Infill contribution ductility factor.
    ru : float
        Reduction factor for infill effects.
    Nation : str, optional
        National annex (default: '').
    eta : float, optional
        Damping correction factor (default: 1).
    SpectTyp : int, optional
        Spectrum type: 1=Type1, 2=Type2 (default: 1).
    Tp : array_like, optional
        Period array (default: linspace(0.001, 3.999, 1000)).

    Returns
    -------
    tuple
        (Sain, Sdin) - Inelastic acceleration and displacement spectra.
    """
    Sae, Sde, SpectreProps = EC8(ag, SoilTyp, eta=eta, SpectTyp=SpectTyp,
                                 T=Tp, Nation=Nation)

    Tc = SpectreProps['Tc']
    Td = SpectreProps['Td']

    TdStar = Td*((2-ru)**0.5)
    deltaT = (Tp-Tc)/(TdStar-Tc)

    TcDown = np.where(Tp <= Tc)
    TdDown = np.where((Tp > Tc) & (Tp <= TdStar))
    TdUp = np.where(Tp > TdStar)

    RMuS = np.zeros(Tp.shape)
    RMuS[TcDown] = 0.7*(Tp[TcDown]/Tc) * (MuS-1) + 1
    RMuS[TdDown] = (0.7+0.3*deltaT[TdDown])*(MuS-1) + 1
    RMuS[TdUp] = MuS
    c = np.ones(Tp.shape)
    if mu <= MuS:
        R0 = np.ones(Tp.shape)
        mu0 = 1
        c[TcDown] = 0.7*Tp[TcDown]/Tc
        c[TdDown] = 0.7 + 0.3*deltaT[TdDown]
    else:
        R0 = RMuS
        mu0 = MuS
        c[TcDown] = 0.7*(ru**0.5)*((Tp[TcDown]/Tc)**(1/(ru**0.5)))
        c[TdDown] = 0.7*(ru**0.5)*(1-deltaT[TdDown]) + deltaT[TdDown]
    R = c*(mu-mu0) + R0
    Sain = Sae/R
    Sdin = Sde*mu/R
    return Sain, Sdin


def EC8_N2(ag, SoilTyp, mu, eta=1, SpectTyp=1, Nation='',
           Tp=np.linspace(0.001, 6, 1000)):
    """
    Generate simplified N2 method response spectrum.

    Parameters
    ----------
    ag : float
        Design ground acceleration.
    SoilTyp : str
        Soil type classification ('A', 'B', 'C', 'D', 'E').
    mu : float
        Target ductility factor.
    eta : float, optional
        Damping correction factor (default: 1).
    SpectTyp : int, optional
        Spectrum type: 1=Type1, 2=Type2 (default: 1).
    Nation : str, optional
        National annex (default: '').
    Tp : array_like, optional
        Period array (default: linspace(0.001, 6, 1000)).

    Returns
    -------
    tuple
        (Sain, Sdin) - Inelastic acceleration and displacement spectra.
    """
    Sae, Sde, SpectreProps = EC8(ag, SoilTyp, eta=eta, SpectTyp=SpectTyp,
                                 T=Tp, Nation=Nation)
    Tc = SpectreProps['Tc']
    R = np.zeros(Sae.shape)
    R[np.where(Tp < Tc)] = 1+(mu-1)*Tp[np.where(Tp < Tc)]/Tc
    R[np.where(Tp >= Tc)] = mu
    Sain = Sae/R
    Sdin = Sde*mu/R
    return Sain, Sdin
