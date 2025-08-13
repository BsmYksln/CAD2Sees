"""
Infill strut creation module for CAD2Sees.

Provides functionality for creating and configuring masonry infill struts
with various mechanical properties, geometric configurations, and modelling
approaches for nonlinear structural analysis.
"""

from cad2sees.helpers import units
from .component_modeller import infill_modeller as im


def create(InfillData, InfillProps):
    """
    Create infill strut models for structural analysis.

    Processes infill data and properties to create individual infill strut
    models with appropriate mechanical properties, geometric configurations,
    and modelling parameters including unit conversions and data preparation.

    Parameters
    ----------
    InfillData : dict
        Infill strut data including ID, Type, geometric dimensions,
        material properties, and connectivity information
    InfillProps : dict
        Infill properties indexed by type including mechanical properties,
        reduction methods, model types, and opening dimensions

    Returns
    -------
    dict
        Infill model outputs indexed by ID including displacement limits,
        elevations, interaction coefficients, and related element IDs
    """
    InfillOuts = {}

    # Process each infill strut
    for i, ID in enumerate(InfillData['ID']):
        # Get infill type and corresponding properties
        InfillType = InfillData['Type'][i]
        CurInfillProp = InfillProps[InfillType]

        # Extract current infill data from arrays
        CurInfillData = {k: InfillData[k][i] for k in InfillData.keys()}

        # Apply unit conversions
        CurInfillData['B'] *= units.cm      # Bay width: cm → m
        CurInfillData['H'] *= units.cm      # Story height: cm → m
        CurInfillData['i_Z'] *= units.cm    # Bottom elevation: cm → m
        CurInfillData['hc'] *= units.mm     # Column height: mm → m
        CurInfillData['bc'] *= units.mm     # Column width: mm → m
        CurInfillData['hb'] *= units.mm     # Beam height: mm → m
        CurInfillData['bb'] *= units.mm     # Beam width: mm → m
        CurInfillData['Ec'] *= units.MPa    # Elastic modulus: MPa → Pa

        # Set additional parameters
        CurInfillData['Fv'] = 0  # Vertical load (handled separately)
        CurInfillData['lp'] = CurInfillProp['lp'] * units.m  # Opening length
        CurInfillData['hp'] = CurInfillProp['hp'] * units.m  # Opening height

        # Create infill model instance
        CurrentInfill = im.In_Plane(
            CurInfillData,
            CurInfillProp['MechanicalProps'],
            CurInfillProp['StiffnessReductionType'],
            CurInfillProp['StrutWidthType'],
            CurInfillProp['CriticalStressFlag'],
            CurInfillProp['BackboneType'],
            CurInfillProp['HysteresisType'],
            EpsThetaFlag=CurInfillProp['EpsThetaFlag']
        )

        # Build model with frame-infill interaction effects
        CurrentInfill.BuildWInteraction()

        # Store output data for structural analysis
        InfillOuts[str(ID)] = {
            'ULimits': CurrentInfill.ULimits,           # Displacement limits
            'BotZ': float(CurInfillData['i_Z']),        # Bottom elevation
            'Alphas': CurrentInfill.Alphas,             # Interaction coeffs
            'Related_Nodes': CurrentInfill.RelatedNodes,  # Related node IDs
            'Related_Frames': CurrentInfill.RelatedFrames,  # Related frame IDs
            'Direction': int(CurInfillData['Direction'])  # Infill orientation
        }

    return InfillOuts
