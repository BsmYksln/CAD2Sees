"""
CAD2Sees module dedicated to the modeling of reinforced concrete frame elements.

Classes for modelling RC frame elements in OpenSees with plastic hinges.
"""

import openseespy.opensees as ops
from cad2sees.helpers import units
from cad2sees.struct_utils.MC import MC
from cad2sees.struct_utils import capacity as C
from cad2sees.struct_utils.ideal_fit import multi_linear
import numpy as np


class BWH_Frame:
    """
    Beam-with-hinges (BWH) frame element for reinforced concrete structures.

    Models RC frame elements with plastic hinges at both ends, 
    supporting multiple moment-curvature analysis methods and capacity calculations.

    Args:
        modelling_type (str): Hinge modelling approach ('Fiber', 'FiberOPS', 'FiberOPS2', 'Simple')
        frame_id (int): Unique identifier for the frame element
        frame_direction (int): Frame orientation (1=X, 2=Y, 3=Vertical)
        geometric_transform_tag (int): OpenSees geometric transformation tag
        i_section_props (dict): Section properties at the I-end
        j_section_props (dict): Section properties at the J-end
        i_node_data (dict): Node data at the I-end
        j_node_data (dict): Node data at the J-end
        shear_flag (int, optional): Include shear behaviour (0=no, 1=yes)
    """

    def __init__(self, modelling_type, frame_id, frame_direction,
                 geometric_transform_tag, i_section_props, j_section_props,
                 i_node_data, j_node_data, shear_flag=0):
        """
        Initialise BWH_Frame element.

        Args:
            modelling_type (str): Type of hinge modelling
            frame_id (int): Unique frame element identifier
            frame_direction (int): Frame direction (1=X, 2=Y, 3=Vertical)
            geometric_transform_tag (int): OpenSees geometric transformation
                                       tag
            i_section_props (dict): Section properties at I-end
            j_section_props (dict): Section properties at J-end
            i_node_data (dict): Node data at I-end
            j_node_data (dict): Node data at J-end
            shear_flag (int, optional): Include shear behaviour (0=no, 1=yes)
        """

        self.FID = frame_id
        self.FDir = frame_direction
        self.Type = modelling_type
        self.GTTAG = geometric_transform_tag
        self.ISP = i_section_props
        self.JSP = j_section_props
        self.IND = i_node_data
        self.JND = j_node_data
        self.SF = shear_flag
        self.RigidMatTag = int(999999)
        self.Outs = {}

    def build(self):
        """
        Build the complete frame element model.
        
        Creates materials, sections, and elements in sequence.
        """
        self._define_flexural_materials()
        # self._define_shear_materials()
        self._assemble_section()
        self._assemble_element()

    def buildWcapacity(self, collect_hinge_properties=True):
        """
        Build frame element with capacity analysis and hinge properties.
        
        Args:
            collect_hinge_properties (bool, optional): Whether to collect
                detailed hinge properties for output
                
        Returns:
            tuple: (Outs, MCOutsI, MCOutsJ) - Output dictionaries with
                  flexural/shear capacities and moment-curvature data
        """
        self.build()

        # Cache moment-curvature results to avoid recalculation
        if 'MCOut' not in self.ISP:
            self.ISP['MCOut'] = {}
        if str(self.IND['NLoad']) not in self.ISP['MCOut']:
            self.ISP['MCOut'][str(self.IND['NLoad'])] = self.MCOutsI
        if 'MCOut' not in self.JSP:
            self.JSP['MCOut'] = {}
        if str(self.JND['NLoad']) not in self.JSP['MCOut']:
            self.JSP['MCOut'][str(self.JND['NLoad'])] = self.MCOutsJ

        _ = self.capacity()

        if collect_hinge_properties:
            # Collect hinge properties for both directions and ends
            # I-end properties
            moment_pz_i = list(self.MCOutsI['MIdealPZ'])
            moment_nz_i = list(self.MCOutsI['MIdealNZ'])
            curvature_pz_i = list(self.MCOutsI['CurvIdealPZ'])
            curvature_nz_i = list(self.MCOutsI['CurvIdealNZ'])

            # J-end properties
            moment_pz_j = list(self.MCOutsJ['MIdealPZ'])
            moment_nz_j = list(self.MCOutsJ['MIdealNZ'])
            curvature_pz_j = list(self.MCOutsJ['CurvIdealPZ'])
            curvature_nz_j = list(self.MCOutsJ['CurvIdealNZ'])

            # Y-direction properties for both ends
            moment_py_i = list(self.MCOutsI['MIdealPY'])
            moment_ny_i = list(self.MCOutsI['MIdealNY'])
            curvature_py_i = list(self.MCOutsI['CurvIdealPY'])
            curvature_ny_i = list(self.MCOutsI['CurvIdealNY'])

            moment_py_j = list(self.MCOutsJ['MIdealPY'])
            moment_ny_j = list(self.MCOutsJ['MIdealNY'])
            curvature_py_j = list(self.MCOutsJ['CurvIdealPY'])
            curvature_ny_j = list(self.MCOutsJ['CurvIdealNY'])

            # Store plastic hinge lengths
            self.Outs['Flexural']['LpI'] = self.Lpi
            self.Outs['Flexural']['LpJ'] = self.Lpj

            # Store Z-Z axis properties
            self.Outs['Flexural']['Izz']['MP'] = moment_pz_i
            self.Outs['Flexural']['Izz']['MN'] = moment_nz_i
            self.Outs['Flexural']['Izz']['CP'] = curvature_pz_i
            self.Outs['Flexural']['Izz']['CN'] = curvature_nz_i

            self.Outs['Flexural']['Jzz']['MP'] = moment_pz_j
            self.Outs['Flexural']['Jzz']['MN'] = moment_nz_j
            self.Outs['Flexural']['Jzz']['CP'] = curvature_pz_j
            self.Outs['Flexural']['Jzz']['CN'] = curvature_nz_j

            # Store Y-Y axis properties
            self.Outs['Flexural']['Iyy']['MP'] = moment_py_i
            self.Outs['Flexural']['Iyy']['MN'] = moment_ny_i
            self.Outs['Flexural']['Iyy']['CP'] = curvature_py_i
            self.Outs['Flexural']['Iyy']['CN'] = curvature_ny_i

            self.Outs['Flexural']['Jyy']['MP'] = moment_py_j
            self.Outs['Flexural']['Jyy']['MN'] = moment_ny_j
            self.Outs['Flexural']['Jyy']['CP'] = curvature_py_j
            self.Outs['Flexural']['Jyy']['CN'] = curvature_ny_j

        return self.Outs, self.MCOutsI, self.MCOutsJ

    def capacity(self):
        """
        Calculate flexural and shear capacities for the frame element.
        
        Returns:
            dict: Dictionary containing flexural and shear capacity results
        """
        frame_capacity = C.Frame(self.ISP, self.JSP, self.IND, self.JND)
        _ = frame_capacity.Flexural()
        _ = frame_capacity.Shear()
        self.Outs['Flexural'] = frame_capacity.FlexuralOut
        self.Outs['Shear'] = frame_capacity.ShearOut
        return self.Outs

    def _make_multi_linear(self, moments, curvatures, x_maxs, limits):
        """
        Create multi-linear idealisation of moment-curvature relationship.
        
        Args:
            moments (array): Moment values from analysis
            curvatures (array): Corresponding curvature values
            x_maxs (array): Maximum strain/stress values
            limits (list): Limit points for multi-linear idealisation
            
        Returns:
            tuple: ([curvatures, moments], [ideal_curvatures, ideal_moments])
        """
        # Filter data to maintain consistent sign
        sign_mask = (np.sign(moments) == np.sign(moments[1])).reshape(-1)
        moments_clean = np.array(moments)[sign_mask]
        curvatures_clean = np.array(curvatures)[sign_mask]
        x_maxs_clean = np.array(x_maxs)[sign_mask]

        try:
            peak_index = np.where(
                abs(moments_clean) == max(abs(moments_clean))
            )[0][0]
        except IndexError:
            print('No Peak Found!')
            peak_index = len(moments_clean) - 1

        # Process post-peak behavior
        post_peak_moments = moments_clean[peak_index:]
        post_peak_x_maxs = x_maxs_clean[peak_index:]

        # Adjust limits based on post-peak degradation
        if (abs(post_peak_moments) < max(abs(moments_clean)) * 0.1).any():
            degradation_index = np.where(
                abs(post_peak_moments) < max(abs(moments_clean)) * 0.1
            )[0][0]
            x_limit = post_peak_x_maxs[degradation_index]
            limits[-1] = min(x_limit, limits[-1])

        # Create flag points for multi-linear idealization
        flag_points = []
        for limit in limits:
            try:
                flag_points.append(
                    curvatures_clean[np.where(x_maxs_clean >= limit)[0][0]]
                )
            except IndexError:
                flag_points.append(curvatures_clean[-1])

        # Add origin point
        moments_with_origin = np.insert(moments_clean, 0, 0)
        curvatures_with_origin = np.insert(curvatures_clean, 0, 0)

        # Create idealized multi-linear relationship
        ideal_curvatures, ideal_moments = multi_linear(
            curvatures_with_origin, moments_with_origin, flag_points
        )

        return ([curvatures_with_origin, moments_with_origin],
                [ideal_curvatures, ideal_moments])

    def _define_flexural_materials(self):
        if self.Type == 'Fiber':
            LoadStrI = str(self.IND['NLoad'])
            if 'MCOut' in self.ISP and LoadStrI in self.ISP['MCOut']:
                CurvIdealPZI = self.ISP['MCOut'][LoadStrI]['CurvIdealPZ']
                MIdealPZI = self.ISP['MCOut'][LoadStrI]['MIdealPZ']
                CurvIdealPYI = self.ISP['MCOut'][LoadStrI]['CurvIdealPY']
                MIdealPYI = self.ISP['MCOut'][LoadStrI]['MIdealPY']
                CurvIdealNZI = self.ISP['MCOut'][LoadStrI]['CurvIdealNZ']
                MIdealNZI = self.ISP['MCOut'][LoadStrI]['MIdealNZ']
                CurvIdealNYI = self.ISP['MCOut'][LoadStrI]['CurvIdealNY']
                MIdealNYI = self.ISP['MCOut'][LoadStrI]['MIdealNY']
                CurvPZI = self.ISP['MCOut'][LoadStrI]['CurvPZ']
                MPZI = self.ISP['MCOut'][LoadStrI]['MPZ']
                CurvPYI = self.ISP['MCOut'][LoadStrI]['CurvPY']
                MPYI = self.ISP['MCOut'][LoadStrI]['MPY']
                CurvNZI = self.ISP['MCOut'][LoadStrI]['CurvNZ']
                MNZI = self.ISP['MCOut'][LoadStrI]['MNZ']
                CurvNYI = self.ISP['MCOut'][LoadStrI]['CurvNY']
                MNYI = self.ISP['MCOut'][LoadStrI]['MNY']
            else:
                # I Section Rotation Around Z-Z
                CurSectID = self.ISP['ID']
                CurSectL = self.IND['NLoad']
                # Positive
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0.0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Positive')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPZI, MPZI], [CurvIdealPZI, MIdealPZI] = CleanCM, IdealCM
                # Negative
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0.0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Negative')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNZI, MNZI], [CurvIdealNZI, MIdealNZI] = CleanCM, IdealCM

                # I Section Rotation Around Y-Y
                # Positive
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                _, Ms, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Positive')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPYI, MPYI], [CurvIdealPYI, MIdealPYI] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                _, Ms, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Negative')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNYI, MNYI], [CurvIdealNYI, MIdealNYI] = CleanCM, IdealCM

            self.MCOutsI = {'CurvIdealPZ': CurvIdealPZI,
                            'MIdealPZ': MIdealPZI,
                            'CurvIdealNZ': CurvIdealNZI,
                            'MIdealNZ': MIdealNZI,
                            'CurvIdealPY': CurvIdealPYI,
                            'MIdealPY': MIdealPYI,
                            'CurvIdealNY': CurvIdealNYI,
                            'MIdealNY': MIdealNYI,
                            'CurvPZ': CurvPZI,
                            'MPZ': MPZI,
                            'CurvNZ': CurvNZI,
                            'MNZ': MNZI,
                            'CurvPY': CurvPYI,
                            'MPY': MPYI,
                            'CurvNY': CurvNYI,
                            'MNY': MNYI}

            if self.ISP['ID'] == self.JSP['ID']:
                if 'MCOut' not in self.JSP:
                    self.JSP['MCOut'] = {}
                self.JSP['MCOut'][LoadStrI] = self.MCOutsI

            LoadStrJ = str(self.JND['NLoad'])
            if 'MCOut' in self.JSP and LoadStrJ in self.JSP['MCOut']:
                CurvIdealPZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPZ']
                MIdealPZJ = self.JSP['MCOut'][LoadStrJ]['MIdealPZ']
                CurvIdealNZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNZ']
                MIdealNZJ = self.JSP['MCOut'][LoadStrJ]['MIdealNZ']
                CurvIdealPYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPY']
                MIdealPYJ = self.JSP['MCOut'][LoadStrJ]['MIdealPY']
                CurvIdealNYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNY']
                MIdealNYJ = self.JSP['MCOut'][LoadStrJ]['MIdealNY']
                CurvPZJ = self.JSP['MCOut'][LoadStrJ]['CurvPZ']
                MPZJ = self.JSP['MCOut'][LoadStrJ]['MPZ']
                CurvNZJ = self.JSP['MCOut'][LoadStrJ]['CurvNZ']
                MNZJ = self.JSP['MCOut'][LoadStrJ]['MNZ']
                CurvPYJ = self.JSP['MCOut'][LoadStrJ]['CurvPY']
                MPYJ = self.JSP['MCOut'][LoadStrJ]['MPY']
                CurvNYJ = self.JSP['MCOut'][LoadStrJ]['CurvNY']
                MNYJ = self.JSP['MCOut'][LoadStrJ]['MNY']
            else:
                # J Section Rotation Around Z-Z
                CurSectID = self.JSP['ID']
                CurSectL = self.JND['NLoad']
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0.0)  # 0 degrees in radians
                # Positive
                Ms, _, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Positive')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPZJ, MPZJ], [CurvIdealPZJ, MIdealPZJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0.0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Negative')

                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNZJ, MNZJ], [CurvIdealNZJ, MIdealNZJ] = CleanCM, IdealCM

                # J Section Rotation Around Y-Y
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                # Positive
                _, Ms, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Positive')
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPYJ, MPYJ], [CurvIdealPYJ, MIdealPYJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                _, Ms, Curvs, xMaxs, xLims = MCCur.fiber_analysis('Negative')

                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNYJ, MNYJ], [CurvIdealNYJ, MIdealNYJ] = CleanCM, IdealCM

            self.MCOutsJ = {'CurvIdealPZ': CurvIdealPZJ,
                            'MIdealPZ': MIdealPZJ,
                            'CurvIdealNZ': CurvIdealNZJ,
                            'MIdealNZ': MIdealNZJ,
                            'CurvIdealPY': CurvIdealPYJ,
                            'MIdealPY': MIdealPYJ,
                            'CurvIdealNY': CurvIdealNYJ,
                            'MIdealNY': MIdealNYJ,
                            'CurvPZ': CurvPZJ,
                            'MPZ': MPZJ,
                            'CurvNZ': CurvNZJ,
                            'MNZ': MNZJ,
                            'CurvPY': CurvPYJ,
                            'MPY': MPYJ,
                            'CurvNY': CurvNYJ,
                            'MNY': MNYJ}

        elif self.Type == 'FiberOPS':
            LoadStrI = str(self.IND['NLoad'])
            if 'MCOut' in self.ISP and LoadStrI in self.ISP['MCOut']:
                CurvIdealPZI = self.ISP['MCOut'][LoadStrI]['CurvIdealPZ']
                MIdealPZI = self.ISP['MCOut'][LoadStrI]['MIdealPZ']
                CurvIdealPYI = self.ISP['MCOut'][LoadStrI]['CurvIdealPY']
                MIdealPYI = self.ISP['MCOut'][LoadStrI]['MIdealPY']
                CurvIdealNZI = self.ISP['MCOut'][LoadStrI]['CurvIdealNZ']
                MIdealNZI = self.ISP['MCOut'][LoadStrI]['MIdealNZ']
                CurvIdealNYI = self.ISP['MCOut'][LoadStrI]['CurvIdealNY']
                MIdealNYI = self.ISP['MCOut'][LoadStrI]['MIdealNY']
                CurvPZI = self.ISP['MCOut'][LoadStrI]['CurvPZ']
                MPZI = self.ISP['MCOut'][LoadStrI]['MPZ']
                CurvPYI = self.ISP['MCOut'][LoadStrI]['CurvPY']
                MPYI = self.ISP['MCOut'][LoadStrI]['MPY']
                CurvNZI = self.ISP['MCOut'][LoadStrI]['CurvNZ']
                MNZI = self.ISP['MCOut'][LoadStrI]['MNZ']
                CurvNYI = self.ISP['MCOut'][LoadStrI]['CurvNY']
                MNYI = self.ISP['MCOut'][LoadStrI]['MNY']
            else:
                # I Section Rotation Around Z-Z using OpenSees fiber analysis
                CurSectID = self.ISP['ID']
                CurSectL = self.IND['NLoad']
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0)  # 0 degrees in radians
                # Positive
                print(f'FiberOPS: {CurSectID} - {CurSectL} - Positive - 0 deg')
                Ms, _, Curvs, xMaxs, xLims = MCCur.FiberOPS(
                    'Positive', axialLoad=CurSectL)
                # Use default limits for OpenSees analysis
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPZI, MPZI], [CurvIdealPZI, MIdealPZI] = CleanCM, IdealCM
                # Negative
                print(f'FiberOPS: {CurSectID} - {CurSectL} - Negative - 0 deg')
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, xLims = MCCur.FiberOPS(
                    'Negative', axialLoad=CurSectL)
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNZI, MNZI], [CurvIdealNZI, MIdealNZI] = CleanCM, IdealCM

                # I Section Rotation Around Y-Y using OpenSees fiber analysis
                # Positive
                print(f'FiberOPS: {CurSectID} - {CurSectL} - '
                      f'Positive - 90 deg')
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians

                _, Ms, Curvs, xMaxs, xLims = MCCur.FiberOPS('Positive',
                                                            axialLoad=CurSectL)
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvPYI, MPYI], [CurvIdealPYI, MIdealPYI] = CleanCM, IdealCM

                # Negative
                print(f'FiberOPS: {CurSectID} - {CurSectL} - '
                      f'Negative - 90 deg')
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians

                _, Ms, Curvs, xMaxs, xLims = MCCur.FiberOPS('Negative',
                                                            axialLoad=CurSectL)
                CleanCM, IdealCM = self._make_multi_linear(Ms, Curvs,
                                                           xMaxs, xLims)
                [CurvNYI, MNYI], [CurvIdealNYI, MIdealNYI] = CleanCM, IdealCM

            self.MCOutsI = {'CurvIdealPZ': CurvIdealPZI,
                            'MIdealPZ': MIdealPZI,
                            'CurvIdealNZ': CurvIdealNZI,
                            'MIdealNZ': MIdealNZI,
                            'CurvIdealPY': CurvIdealPYI,
                            'MIdealPY': MIdealPYI,
                            'CurvIdealNY': CurvIdealNYI,
                            'MIdealNY': MIdealNYI,
                            'CurvPZ': CurvPZI,
                            'MPZ': MPZI,
                            'CurvNZ': CurvNZI,
                            'MNZ': MNZI,
                            'CurvPY': CurvPYI,
                            'MPY': MPYI,
                            'CurvNY': CurvNYI,
                            'MNY': MNYI}

            if self.ISP['ID'] == self.JSP['ID']:
                if 'MCOut' not in self.JSP:
                    self.JSP['MCOut'] = {}
                self.JSP['MCOut'][LoadStrI] = self.MCOutsI

            LoadStrJ = str(self.JND['NLoad'])
            if 'MCOut' in self.JSP and LoadStrJ in self.JSP['MCOut']:
                CurvIdealPZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPZ']
                MIdealPZJ = self.JSP['MCOut'][LoadStrJ]['MIdealPZ']
                CurvIdealNZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNZ']
                MIdealNZJ = self.JSP['MCOut'][LoadStrJ]['MIdealNZ']
                CurvIdealPYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPY']
                MIdealPYJ = self.JSP['MCOut'][LoadStrJ]['MIdealPY']
                CurvIdealNYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNY']
                MIdealNYJ = self.JSP['MCOut'][LoadStrJ]['MIdealNY']
                CurvPZJ = self.JSP['MCOut'][LoadStrJ]['CurvPZ']
                MPZJ = self.JSP['MCOut'][LoadStrJ]['MPZ']
                CurvNZJ = self.JSP['MCOut'][LoadStrJ]['CurvNZ']
                MNZJ = self.JSP['MCOut'][LoadStrJ]['MNZ']
                CurvPYJ = self.JSP['MCOut'][LoadStrJ]['CurvPY']
                MPYJ = self.JSP['MCOut'][LoadStrJ]['MPY']
                CurvNYJ = self.JSP['MCOut'][LoadStrJ]['CurvNY']
                MNYJ = self.JSP['MCOut'][LoadStrJ]['MNY']
            else:
                # J Section Rotation Around Z-Z using OpenSees fiber analysis
                CurSectID = self.JSP['ID']
                CurSectL = self.JND['NLoad']
                # Positive
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0.0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, strain_limits = MCCur.FiberOPS('Positive')
                CleanCM, IdealCM = self._make_multi_linear(
                    Ms, Curvs, xMaxs, strain_limits)
                [CurvPZJ, MPZJ], [CurvIdealPZJ, MIdealPZJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0.0)  # 0 degrees in radians
                Ms, _, Curvs, xMaxs, strain_limits = MCCur.FiberOPS('Negative')
                CleanCM, IdealCM = self._make_multi_linear(
                    Ms, Curvs, xMaxs, strain_limits)
                [CurvNZJ, MNZJ], [CurvIdealNZJ, MIdealNZJ] = CleanCM, IdealCM

                # J Section Rotation Around Y-Y using OpenSees fiber analysis
                # Positive
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians

                _, Ms, Curvs, xMaxs, strain_limits = MCCur.FiberOPS('Positive')
                CleanCM, IdealCM = self._make_multi_linear(
                    Ms, Curvs, xMaxs, strain_limits)
                [CurvPYJ, MPYJ], [CurvIdealPYJ, MIdealPYJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                _, Ms, Curvs, xMaxs, strain_limits = MCCur.FiberOPS('Negative')
                CleanCM, IdealCM = self._make_multi_linear(
                    Ms, Curvs, xMaxs, strain_limits)
                [CurvNYJ, MNYJ], [CurvIdealNYJ, MIdealNYJ] = CleanCM, IdealCM

            self.MCOutsJ = {'CurvIdealPZ': CurvIdealPZJ,
                            'MIdealPZ': MIdealPZJ,
                            'CurvIdealNZ': CurvIdealNZJ,
                            'MIdealNZ': MIdealNZJ,
                            'CurvIdealPY': CurvIdealPYJ,
                            'MIdealPY': MIdealPYJ,
                            'CurvIdealNY': CurvIdealNYJ,
                            'MIdealNY': MIdealNYJ,
                            'CurvPZ': CurvPZJ,
                            'MPZ': MPZJ,
                            'CurvNZ': CurvNZJ,
                            'MNZ': MNZJ,
                            'CurvPY': CurvPYJ,
                            'MPY': MPYJ,
                            'CurvNY': CurvNYJ,
                            'MNY': MNYJ}

        elif self.Type == 'FiberOPS2':
            LoadStrI = str(self.IND['NLoad'])
            if 'MCOut' in self.ISP and LoadStrI in self.ISP['MCOut']:
                CurvIdealPZI = self.ISP['MCOut'][LoadStrI]['CurvIdealPZ']
                MIdealPZI = self.ISP['MCOut'][LoadStrI]['MIdealPZ']
                CurvIdealPYI = self.ISP['MCOut'][LoadStrI]['CurvIdealPY']
                MIdealPYI = self.ISP['MCOut'][LoadStrI]['MIdealPY']
                CurvIdealNZI = self.ISP['MCOut'][LoadStrI]['CurvIdealNZ']
                MIdealNZI = self.ISP['MCOut'][LoadStrI]['MIdealNZ']
                CurvIdealNYI = self.ISP['MCOut'][LoadStrI]['CurvIdealNY']
                MIdealNYI = self.ISP['MCOut'][LoadStrI]['MIdealNY']
                CurvPZI = self.ISP['MCOut'][LoadStrI]['CurvPZ']
                MPZI = self.ISP['MCOut'][LoadStrI]['MPZ']
                CurvPYI = self.ISP['MCOut'][LoadStrI]['CurvPY']
                MPYI = self.ISP['MCOut'][LoadStrI]['MPY']
                CurvNZI = self.ISP['MCOut'][LoadStrI]['CurvNZ']
                MNZI = self.ISP['MCOut'][LoadStrI]['MNZ']
                CurvNYI = self.ISP['MCOut'][LoadStrI]['CurvNY']
                MNYI = self.ISP['MCOut'][LoadStrI]['MNY']
            else:
                # I Section Rotation Around Z-Z using OpenSees biaxial analysis
                CurSectID = self.ISP['ID']
                CurSectL = self.IND['NLoad']
                # Positive
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0)  # 0 degrees in radians
                result = MCCur.FiberOPS2('Positive', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mzs, Curvzs, xMaxs, strain_limits)
                [CurvPZI, MPZI], [CurvIdealPZI, MIdealPZI] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           0)  # 0 degrees in radians
                result = MCCur.FiberOPS2('Negative', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mzs, Curvzs, xMaxs, strain_limits)
                [CurvNZI, MNZI], [CurvIdealNZI, MIdealNZI] = CleanCM, IdealCM

                # I Section Rotation Around Y-Y using OpenSees biaxial analysis
                # Positive
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                result = MCCur.FiberOPS2('Positive', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mys, Curvys, xMaxs, strain_limits)
                [CurvPYI, MPYI], [CurvIdealPYI, MIdealPYI] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.ISP,
                           self.IND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                result = MCCur.FiberOPS2('Negative', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mys, Curvys, xMaxs, strain_limits)
                [CurvNYI, MNYI], [CurvIdealNYI, MIdealNYI] = CleanCM, IdealCM

            self.MCOutsI = {'CurvIdealPZ': CurvIdealPZI,
                            'MIdealPZ': MIdealPZI,
                            'CurvIdealNZ': CurvIdealNZI,
                            'MIdealNZ': MIdealNZI,
                            'CurvIdealPY': CurvIdealPYI,
                            'MIdealPY': MIdealPYI,
                            'CurvIdealNY': CurvIdealNYI,
                            'MIdealNY': MIdealNYI,
                            'CurvPZ': CurvPZI,
                            'MPZ': MPZI,
                            'CurvNZ': CurvNZI,
                            'MNZ': MNZI,
                            'CurvPY': CurvPYI,
                            'MPY': MPYI,
                            'CurvNY': CurvNYI,
                            'MNY': MNYI}

            if self.ISP['ID'] == self.JSP['ID']:
                if 'MCOut' not in self.JSP:
                    self.JSP['MCOut'] = {}
                self.JSP['MCOut'][LoadStrI] = self.MCOutsI

            LoadStrJ = str(self.JND['NLoad'])
            if 'MCOut' in self.JSP and LoadStrJ in self.JSP['MCOut']:
                CurvIdealPZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPZ']
                MIdealPZJ = self.JSP['MCOut'][LoadStrJ]['MIdealPZ']
                CurvIdealNZJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNZ']
                MIdealNZJ = self.JSP['MCOut'][LoadStrJ]['MIdealNZ']
                CurvIdealPYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealPY']
                MIdealPYJ = self.JSP['MCOut'][LoadStrJ]['MIdealPY']
                CurvIdealNYJ = self.JSP['MCOut'][LoadStrJ]['CurvIdealNY']
                MIdealNYJ = self.JSP['MCOut'][LoadStrJ]['MIdealNY']
                CurvPZJ = self.JSP['MCOut'][LoadStrJ]['CurvPZ']
                MPZJ = self.JSP['MCOut'][LoadStrJ]['MPZ']
                CurvNZJ = self.JSP['MCOut'][LoadStrJ]['CurvNZ']
                MNZJ = self.JSP['MCOut'][LoadStrJ]['MNZ']
                CurvPYJ = self.JSP['MCOut'][LoadStrJ]['CurvPY']
                MPYJ = self.JSP['MCOut'][LoadStrJ]['MPY']
                CurvNYJ = self.JSP['MCOut'][LoadStrJ]['CurvNY']
                MNYJ = self.JSP['MCOut'][LoadStrJ]['MNY']
            else:
                # J Section Rotation Around Z-Z using OpenSees biaxial analysis
                CurSectID = self.JSP['ID']
                CurSectL = self.JND['NLoad']
                print(f"Section ID: {CurSectID} | Load: {CurSectL} kN")
                # Positive
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0)  # 0 degrees in radians
                result = MCCur.FiberOPS2('Positive', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mzs, Curvzs, xMaxs, strain_limits)
                [CurvPZJ, MPZJ], [CurvIdealPZJ, MIdealPZJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           0)  # 0 degrees in radians
                result = MCCur.FiberOPS2('Negative', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mzs, Curvzs, xMaxs, strain_limits)
                [CurvNZJ, MNZJ], [CurvIdealNZJ, MIdealNZJ] = CleanCM, IdealCM

                # J Section Rotation Around Y-Y using OpenSees biaxial analysis
                # Positive
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                result = MCCur.FiberOPS2('Positive', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mys, Curvys, xMaxs, strain_limits)
                [CurvPYJ, MPYJ], [CurvIdealPYJ, MIdealPYJ] = CleanCM, IdealCM

                # Negative
                MCCur = MC(self.JSP,
                           self.JND['NLoad'],
                           np.pi*0.5)  # 90 degrees in radians
                result = MCCur.FiberOPS2('Negative', axialLoad=CurSectL)
                Mzs, Mys, Curvzs, Curvys, xMaxs, strain_limits = result
                CleanCM, IdealCM = self._make_multi_linear(
                    Mys, Curvys, xMaxs, strain_limits)
                [CurvNYJ, MNYJ], [CurvIdealNYJ, MIdealNYJ] = CleanCM, IdealCM

            self.MCOutsJ = {'CurvIdealPZ': CurvIdealPZJ,
                            'MIdealPZ': MIdealPZJ,
                            'CurvIdealNZ': CurvIdealNZJ,
                            'MIdealNZ': MIdealNZJ,
                            'CurvIdealPY': CurvIdealPYJ,
                            'MIdealPY': MIdealPYJ,
                            'CurvIdealNY': CurvIdealNYJ,
                            'MIdealNY': MIdealNYJ,
                            'CurvPZ': CurvPZJ,
                            'MPZ': MPZJ,
                            'CurvNZ': CurvNZJ,
                            'MNZ': MNZJ,
                            'CurvPY': CurvPYJ,
                            'MPY': MPYJ,
                            'CurvNY': CurvNYJ,
                            'MNY': MNYJ}
        elif self.Type == 'Simple':
            bi = self.ISP['b']*units.mm
            hi = self.ISP['h']*units.mm
            bj = self.JSP['b']*units.mm
            hj = self.JSP['h']*units.mm
            fc = self.ISP['fc0']*units.MPa
            fy = self.ISP['fy']*units.MPa
            Es = self.ISP['Es']*units.MPa
            MCCur = MC(self.ISP,
                       self.IND['NLoad'],
                       'Z')
            MyPZI, MyNZI, MyPYI, MyNYI = MCCur.Simple()
            MCCur = MC(self.JSP,
                       self.JND['NLoad'],
                       'Z')
            MyPZJ, MyNZJ, MyPYJ, MyNYJ = MCCur.Simple()
            MyPZI = float(MyPZI)
            MyNZI = float(MyNZI)
            MyPZJ = float(MyPZJ)
            MyNZJ = float(MyNZJ)
            MyPYI = float(MyPYI)
            MyNYI = float(MyNYI)
            MyPYJ = float(MyPYJ)
            MyNYJ = float(MyNYJ)

            # Computation of Capping Moments
            McPZI = 1.077*MyPZI
            McNZI = 1.077*MyNZI
            McPZJ = 1.077*MyPZJ
            McNZJ = 1.077*MyNZJ
            McPYI = 1.077*MyPYI
            McNYI = 1.077*MyNYI
            McPYJ = 1.077*MyPYJ
            McNYJ = 1.077*MyNYJ

            # Computation of Ultimate Moments
            MuPZI = 0.8*McPZI
            MuNZI = 0.8*McNZI
            MuPZJ = 0.8*McPZJ
            MuNZJ = 0.8*McNZJ

            MuPYI = 0.8*McPYI
            MuNYI = 0.8*McNYI
            MuPYJ = 0.8*McPYJ
            MuNYJ = 0.8*McNYJ

            # Computation of Maximum Capacity
            # Take the residual as 10% of the capping moment
            MmPZI = 0.1*McPZI
            MmNZI = 0.1*McNZI
            MmPZJ = 0.1*McPZJ
            MmNZJ = 0.1*McNZJ

            MmPYI = 0.1*McPYI
            MmNYI = 0.1*McNYI
            MmPYJ = 0.1*McPYJ
            MmNYJ = 0.1*McNYJ

            MIdealPZI = [0, MyPZI, McPZI, MuPZI, MmPZI]
            MIdealPYI = [0, MyPYI, McPYI, MuPYI, MmPYI]
            MIdealNZI = [0, -MyNZI, -McNZI, -MuNZI, -MmNZI]
            MIdealNYI = [0, -MyNYI, -McNYI, -MuNYI, -MmNYI]

            MIdealPZJ = [0, MyPZJ, McPZJ, MuPZJ, MmPZJ]
            MIdealPYJ = [0, MyPYJ, McPYJ, MuPYJ, MmPYJ]
            MIdealNZJ = [0, -MyNZJ, -McNZJ, -MuNZJ, -MmNZJ]
            MIdealNYJ = [0, -MyNYJ, -McNYJ, -MuNYJ, -MmNYJ]

            CurvyPZI = 2.1*(fy/Es)/(hi)
            CurvyPYI = 2.1*(fy/Es)/(bi)
            CurvyPZJ = 2.1*(fy/Es)/(hj)
            CurvyPYJ = 2.1*(fy/Es)/(bj)

            nui = abs(float(self.IND['NLoad']))*units.kN/(bi*hi*fc)
            nuj = abs(float(self.JND['NLoad']))*units.kN/(bj*hj*fc)

            if nui > 0.9999 or nuj > 0.9999:
                print('Axial Load Ratio is greater then 1.0!')
                print(f'I End: {nui}, J End: {nuj}')
            muPhiI = 22.651-47.348*max(min(nui, 0.25), 0.1)
            muPhiJ = 22.651-47.348*max(min(nuj, 0.25), 0.1)
            CurvuPZI = CurvyPZI*muPhiI
            CurvuPZJ = CurvyPZJ*muPhiJ
            CurvuPYI = CurvyPYI*muPhiI
            CurvuPYJ = CurvyPYJ*muPhiJ

            # Computation of Capping Curvature
            appI = -0.1437*max(nui, 0.1)-0.0034
            appJ = -0.1437*max(nuj, 0.1)-0.0034
            CurvcPZI = CurvuPZI+(CurvyPZI*0.2*1.077/appI)
            CurvcPYI = CurvuPYI+(CurvyPYI*0.2*1.077/appI)
            CurvcPZJ = CurvuPZJ+(CurvyPZJ*0.2*1.077/appJ)
            CurvcPYJ = CurvuPYJ+(CurvyPYJ*0.2*1.077/appJ)

            # Computation of Maximum Curvature
            # Positive  MmPZI
            CurvmPZI = CurvcPZI+(MmPZI - McPZI)*CurvyPZI/(appI*MyPZI)
            CurvmPZJ = CurvcPYI+(MmPZJ - McPZJ)*CurvyPYI/(appJ*MyPZJ)
            CurvmPYI = CurvcPZJ+(MmPYI - McPYI)*CurvyPZJ/(appI*MyPYI)
            CurvmPYJ = CurvcPYJ+(MmPYJ - McPYJ)*CurvyPYJ/(appJ*MyPYJ)

            CurvIdealPZI = [0, CurvyPZI, CurvcPZI, CurvuPZI, CurvmPZI]
            CurvIdealPYI = [0, CurvyPYI, CurvcPYI, CurvuPYI, CurvmPYI]
            CurvIdealNZI = [0, -CurvyPZI, -CurvcPZI, -CurvuPZI, -CurvmPZI]
            CurvIdealNYI = [0, -CurvyPYI, -CurvcPYI, -CurvuPYI, -CurvmPYI]

            self.CurvOut = [CurvIdealPZI[1:], CurvIdealPYI[1:],
                            CurvIdealNZI[1:], CurvIdealNYI[1:]]
            self.MomPOut = [CurvIdealPZI[1:], CurvIdealPYI[1:],
                            CurvIdealNZI[1:], CurvIdealNYI[1:]]

            CurvIdealPZJ = [0, CurvyPZJ, CurvcPZJ, CurvuPZJ, CurvmPZJ]
            CurvIdealPYJ = [0, CurvyPYJ, CurvcPYJ, CurvuPYJ, CurvmPYJ]
            CurvIdealNZJ = [0, -CurvyPZJ, -CurvcPZJ, -CurvuPZJ, -CurvmPZJ]
            CurvIdealNYJ = [0, -CurvyPYJ, -CurvcPYJ, -CurvuPYJ, -CurvmPYJ]

            self.MCOutsJ = {'CurvIdealPZ': CurvIdealPZJ,
                            'MIdealPZ': MIdealPZJ,
                            'CurvIdealNZ': CurvIdealNZJ,
                            'MIdealNZ': MIdealNZJ,
                            'CurvIdealPY': CurvIdealPYJ,
                            'MIdealPY': MIdealPYJ,
                            'CurvIdealNY': CurvIdealNYJ,
                            'MIdealNY': MIdealNYJ}

            self.MCOutsI = {'CurvIdealPZ': CurvIdealPZI,
                            'MIdealPZ': MIdealPZI,
                            'CurvIdealNZ': CurvIdealNZI,
                            'MIdealNZ': MIdealNZI,
                            'CurvIdealPY': CurvIdealPYI,
                            'MIdealPY': MIdealPYI,
                            'CurvIdealNY': CurvIdealNYI,
                            'MIdealNY': MIdealNYI}

        # Ratio of maximum deformation at which reloading begins
        rDispM = [0.1, 0.1]  # Pos_env. Neg_env.
        # Ratio of envelope force (corresponding to maximum deformation)
        # at which reloading begins
        rForceM = [0.3, 0.3]  # Pos_env. Neg_env.
        # Ratio of monotonic strength developed upon unloading
        uForceM = [-0.8, -0.8]  # Pos_env. Neg_env.
        # gammaK1 gammaK2 gammaK3 gammaK4 gammaKLimit
        gammaKM = [0.0, 0.0, 0.0, 0.0, 0.0]
        # gammaD1 gammaD2 gammaD3 gammaD4 gammaDLimit
        gammaDM = [0.0, 0.0, 0.0, 0.0, 0.0]
        # gammaF1 gammaF2 gammaF3 gammaF4 gammaFLimit
        gammaFM = [0.0, 0.0, 0.0, 0.0, 0.0]
        gammaEM = 0.0
        damM = "energy"
        ohingeMTagzzI = int(float(f"101{self.FID}"))
        ohingeMTagzzJ = int(float(f"102{self.FID}"))
        ohingeMTagyyI = int(float(f"103{self.FID}"))
        ohingeMTagyyJ = int(float(f"104{self.FID}"))
        UniaxialPinchingFun(ohingeMTagzzI,
                            MIdealPZI[1:],
                            MIdealNZI[1:],
                            CurvIdealPZI[1:],
                            CurvIdealNZI[1:],
                            rDispM, rForceM, uForceM,
                            gammaKM, gammaDM, gammaFM, gammaEM,
                            damM)
        UniaxialPinchingFun(ohingeMTagzzJ,
                            MIdealPZJ[1:],
                            MIdealNZJ[1:],
                            CurvIdealPZJ[1:],
                            CurvIdealNZJ[1:],
                            rDispM, rForceM, uForceM,
                            gammaKM, gammaDM, gammaFM, gammaEM,
                            damM)
        UniaxialPinchingFun(ohingeMTagyyI,
                            MIdealPYI[1:],
                            MIdealNYI[1:],
                            CurvIdealPYI[1:],
                            CurvIdealNYI[1:],
                            rDispM, rForceM, uForceM,
                            gammaKM, gammaDM, gammaFM, gammaEM,
                            damM)
        UniaxialPinchingFun(ohingeMTagyyJ,
                            MIdealPYJ[1:],
                            MIdealNYJ[1:],
                            CurvIdealPYJ[1:],
                            CurvIdealNYJ[1:],
                            rDispM, rForceM, uForceM,
                            gammaKM, gammaDM, gammaFM, gammaEM,
                            damM)
        self.hingeMTagzzI = int(float(f"105{self.FID}"))
        self.hingeMTagzzJ = int(float(f"106{self.FID}"))
        self.hingeMTagyyI = int(float(f"107{self.FID}"))
        self.hingeMTagyyJ = int(float(f"108{self.FID}"))
        CurvLimzzI = 100*CurvIdealPZI[1]
        CurvLimyyI = 100*CurvIdealPYI[1]
        CurvLimzzJ = 100*CurvIdealPZJ[1]
        CurvLimyyJ = 100*CurvIdealPYJ[1]
        ops.uniaxialMaterial('MinMax', self.hingeMTagzzI, ohingeMTagzzI,
                             '-min', -CurvLimzzI,
                             '-max', CurvLimzzI)
        ops.uniaxialMaterial('MinMax', self.hingeMTagzzJ, ohingeMTagzzJ,
                             '-min', -CurvLimzzJ,
                             '-max', CurvLimzzJ)
        ops.uniaxialMaterial('MinMax', self.hingeMTagyyI, ohingeMTagyyI,
                             '-min', -CurvLimyyI,
                             '-max', CurvLimyyI)
        ops.uniaxialMaterial('MinMax', self.hingeMTagyyJ, ohingeMTagyyJ,
                             '-min', -CurvLimyyJ,
                             '-max', CurvLimyyJ)

    def _define_shear_materials(self):
        # TODO
        print('Shear Behaviour Not Implemented Yet!')

    def _assemble_section(self):
        # Internal elastic section tag
        self.intSecTag = int(float(f"111{self.FID}"))
        # Plastic hinge section tags
        fTagzzI = int(float(f"112{self.FID}"))  # Section tag Mz @I
        fTagzzJ = int(float(f"113{self.FID}"))  # Section tag Mz @J

        self.phTagI = int(float(f"116{self.FID}"))
        self.phTagJ = int(float(f"117{self.FID}"))

        # Create internal elastic section

        # TODO we use unique section for entire element
        hi = self.ISP['h']*units.mm
        bi = self.ISP['b']*units.mm
        GJ = hi*bi*(hi**2+bi**2)/12
        Ag = bi*hi
        hj = self.JSP['h']*units.mm
        bj = self.JSP['b']*units.mm
        Ec = self.ISP['Ec']*units.MPa
        Gc = 0.4*Ec

        IzzI = bi*(hi**3)/12
        EIzzI = IzzI * Ec
        IyyI = hi*(bi**3)/12
        EIyyI = IyyI * Ec

        IzzJ = bj*(hj**3)/12
        EIzzJ = IzzJ * Ec
        IyyJ = hj*(bj**3)/12
        EIyyJ = IyyJ * Ec

        # ! ONLY THE POSSITIVE YEILD MOMENT BEING USED
        KIZ = self.MCOutsI['MIdealPZ'][1] / self.MCOutsI['CurvIdealPZ'][1]
        KJZ = self.MCOutsJ['MIdealPZ'][1] / self.MCOutsJ['CurvIdealPZ'][1]

        KIY = self.MCOutsI['MIdealPY'][1] / self.MCOutsI['CurvIdealPY'][1]
        KJY = self.MCOutsJ['MIdealPY'][1] / self.MCOutsJ['CurvIdealPY'][1]

        EIrZI = KIZ/EIzzI
        EIrZJ = KJZ/EIzzJ

        EIrYI = KIY/EIyyI
        EIrYJ = KJY/EIyyJ

        EIrzz = max(EIrZI, EIrZJ)
        EIryy = max(EIrYI, EIrYJ)

        IzzeI = EIrzz*IzzI
        IzzeJ = EIrzz*IzzJ

        Izze = min(IzzeI, IzzeJ)

        IyyeI = EIryy*IyyI
        IyyeJ = EIryy*IyyJ

        self.Agi = Ag
        self.Ec = Ec
        self.Gc = Gc
        self.Iyye = IyyeI
        self.Izze = IzzeI

        Iyye = min(IyyeI, IyyeJ)
        ops.section('Elastic',
                    self.intSecTag,
                    Ec,
                    Ag, Izze, Iyye, Gc, GJ)

        # Create the plastic hinge section
        # Create the plastic hinge flexural sections for about zz
        ops.section('Uniaxial', fTagzzI, self.hingeMTagzzI, 'Mz')
        ops.section('Uniaxial', fTagzzJ, self.hingeMTagzzJ, 'Mz')

        # Aggregate end sections
        if self.SF == 1:  # If shear hinge exists
            # Aggregate Vyy, Vzz and Myy behaviour to Mzz behaviour
            Imats = [self.RigidMatTag, 'P',
                     self.RigidMatTag, 'T',
                     self.hingeShTagyyI, 'Vy',
                     self.hingeShTagzzI, 'Vz',
                     self.hingeMTagyyI, 'My']
            Jmats = [self.RigidMatTag, 'P',
                     self.RigidMatTag, 'T',
                     self.hingeShTagyyJ, 'Vy',
                     self.hingeShTagzzJ, 'Vz',
                     self.hingeMTagyyJ, 'My']
        else:  # If shear hinge does not exist
            # Aggregate Myy behaviour to Mzz behaviour
            Imats = [self.RigidMatTag, 'P',
                     self.RigidMatTag, 'T',
                     self.RigidMatTag, 'Vy',
                     self.RigidMatTag, 'Vz',
                     self.hingeMTagyyI, 'My']
            Jmats = [self.RigidMatTag, 'P',
                     self.RigidMatTag, 'T',
                     self.RigidMatTag, 'Vy',
                     self.RigidMatTag, 'Vz',
                     self.hingeMTagyyJ, 'My']

        # Important If Beam2Beam Connection no pl hinge! But elastic
        if self.IND['Type'] != 'BCJ' and self.IND['u1'] == 0:
            self.phTagI = self.intSecTag
        else:
            ops.section('Aggregator', self.phTagI, *Imats, '-section', fTagzzI)

        if self.JND['Type'] != 'BCJ' and self.JND['u1'] == 0:
            self.phTagJ = self.intSecTag
        else:
            ops.section('Aggregator', self.phTagJ, *Jmats, '-section', fTagzzJ)

    def _assemble_element(self):
        # Debug print for element coordinates
        # print(f'{self.FID} | {self.IND["Coordinates"]} - '
        #       f'{self.JND["Coordinates"]}')
        Xi = self.IND['Coordinates'][0]*units.cm
        Yi = self.IND['Coordinates'][1]*units.cm
        Zi = self.IND['Coordinates'][2]*units.cm

        Xj = self.JND['Coordinates'][0]*units.cm
        Yj = self.JND['Coordinates'][1]*units.cm
        Zj = self.JND['Coordinates'][2]*units.cm

        phiLi = np.mean(self.ISP['ReinfL'][:, -1]**2)**0.5*units.mm
        phiLj = np.mean(self.JSP['ReinfL'][:, -1]**2)**0.5*units.mm

        # Member length
        L = ((Xi-Xj)**2 + (Yi-Yj)**2 + (Zi-Zj)**2)**0.5
        # Shear Span is assumed to be half length of the element
        Ls = L*0.5

        # Computation of Plastic Hinge Length
        self.Lpi = 0.08*Ls + 0.022*phiLi*(self.ISP['fy'])
        self.Lpj = 0.08*Ls + 0.022*phiLj*(self.JSP['fy'])

        # Define integration scheme and assign actual element
        intgrTag = int(float(f"10{self.FID}"))  # Integration Tag
        # ops.beamIntegration('HingeEndpoint', tag, secI, lpI, secJ, lpJ, secE)
        ops.beamIntegration('HingeEndpoint', intgrTag,
                            self.phTagI, self.Lpi, self.phTagJ, self.Lpj,
                            self.intSecTag)
        # Define Node TAGs
        if self.FDir == 3:
            if self.IND['Type'] == 'BCJ':
                EleINodeTag = int(float(f"1{self.IND['ID']}"))
            else:
                EleINodeTag = int(float(f"{self.IND['ID']}"))
            if self.JND['Type'] == 'BCJ':
                EleJNodeTag = int(float(f"1{self.JND['ID']}"))
            else:
                EleJNodeTag = int(float(f"{self.JND['ID']}"))
        else:
            if self.IND['Type'] == 'BCJ':
                EleINodeTag = int(float(f"6{self.IND['ID']}"))
            else:
                EleINodeTag = int(float(f"{self.IND['ID']}"))
            if self.JND['Type'] == 'BCJ':
                EleJNodeTag = int(float(f"6{self.JND['ID']}"))
            else:
                EleJNodeTag = int(float(f"{self.JND['ID']}"))
        eleNodeTags = [EleINodeTag, EleJNodeTag]

        ops.element('forceBeamColumn', int(self.FID), *eleNodeTags,
                    int(self.GTTAG), intgrTag, '-iter', 10, 1e-12)


def UniaxialPinchingFun(material_tag, p_envelope_force, n_envelope_force,
                        p_envelope_deform, n_envelope_deform, r_disp, r_force,
                        u_force, gamma_k, gamma_d, gamma_f, gamma_e, damage):
    """
    Create a uniaxial Pinching4 material for OpenSees.

    Args:
        material_tag (int): Unique material tag
        p_envelope_force (list): [ePf1, ePf2, ePf3, ePf4] positive envelope force points
        n_envelope_force (list): [eNf1, eNf2, eNf3, eNf4] negative envelope force points
        p_envelope_deform (list): [ePd1, ePd2, ePd3, ePd4] positive envelope deformation points
        n_envelope_deform (list): [eNd1, eNd2, eNd3, eNd4] negative envelope deformation points
        r_disp (list): [rDispP, rDispN] ratio of deformation at reloading (pos, neg)
        r_force (list): [rForceP, rForceN] ratio of force at reloading (pos, neg)
        u_force (list): [uForceP, uForceN] ratio of strength upon unloading (pos, neg)
        gamma_k (list): [gK1, gK2, gK3, gK4, gKLim] unloading stiffness degradation
        gamma_d (list): [gD1, gD2, gD3, gD4, gDLim] reloading stiffness degradation
        gamma_f (list): [gF1, gF2, gF3, gF4, gFLim] strength degradation
        gamma_e (float): gE, energy degradation parameter
        damage (str): dmgType, 'energy' or 'cycle'
    """
    ops.uniaxialMaterial(
        'Pinching4', material_tag,
        p_envelope_force[0], p_envelope_deform[0],
        p_envelope_force[1], p_envelope_deform[1],
        p_envelope_force[2], p_envelope_deform[2],
        p_envelope_force[3], p_envelope_deform[3],
        n_envelope_force[0], n_envelope_deform[0],
        n_envelope_force[1], n_envelope_deform[1],
        n_envelope_force[2], n_envelope_deform[2],
        n_envelope_force[3], n_envelope_deform[3],
        r_disp[0], r_force[0], u_force[0],
        r_disp[1], r_force[1], u_force[1],
        gamma_k[0], gamma_k[1], gamma_k[2], gamma_k[3], gamma_k[4],
        gamma_d[0], gamma_d[1], gamma_d[2], gamma_d[3], gamma_d[4],
        gamma_f[0], gamma_f[1], gamma_f[2], gamma_f[3], gamma_f[4],
        gamma_e, damage
    )
