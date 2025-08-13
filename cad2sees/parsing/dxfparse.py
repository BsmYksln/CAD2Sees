import pandas as pd
import numpy as np
import json
from cad2sees.helpers import geometric_info as gi


class dxfparse:
    """
    Parses DXF files and JSON configuration to extract structural information.

    Processes 3D CAD data, cross-sections, and project information to generate
    structural elements, boundary conditions, and load distributions for
    finite element analysis.
    """

    # Constants
    GRAVITY = 9.807
    COORDINATE_PRECISION = 2
    DEFAULT_GRID_SPACING = 10
    BCJ_PREFIX = '1'
    BEAM_NODE_PREFIX = '6'

    # DXF Group codes
    ENTITY_TYPE_CODE = '  0'
    LAYER_CODE = '  8'
    X_COORD_CODE = ' 10'
    Y_COORD_CODE = ' 20'
    Z_COORD_CODE = ' 30'

    def __init__(self, filepath_3D: str, filepath_Sections: str,
                 filepath_GeneralInformation: str):
        """
        Initialise the DXF parser with file paths.

        Parameters
        ----------
        filepath_3D : str
            Path to 3D model DXF file
        filepath_Sections : str
            Path to sections DXF file
        filepath_GeneralInformation : str
            Path to project JSON file
        """
        # Load configuration
        self.config = self._load_configuration(filepath_GeneralInformation)

        # Initialize material properties from config
        self._initialize_materials()

        # Process DXF files
        self._process_dxf_files(filepath_3D, filepath_Sections)

    def _load_configuration(self, filepath: str) -> dict:
        """Load and validate JSON configuration file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Configuration file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def _initialize_materials(self):
        """Initialise material properties from configuration."""
        rebar = self.config['Materials']['Rebar']
        concrete = self.config['Materials']['Concrete']

        # Extract section details
        self.Hook = self.config['SectionDetails']['Hook']

        # Rebar properties
        self.fy = rebar['fy']
        self.fyw = rebar['fyw']
        self.fsu = rebar['fsu']
        self.Es = rebar['Es']
        self.eps_y = self.fy / self.Es
        self.eps_su = rebar['eps_su']

        # Concrete properties
        self.fc0 = concrete['fc0']
        self.Ec = concrete.get('Ec', 4700 * (self.fc0 ** 0.5))

        # Load and infill properties
        self.AreaLoad = self.config['AreaLoad']
        self.InfillProperties = self.config['Infills']

    def _process_dxf_files(self, filepath_3D: str, filepath_Sections: str):
        """Process DXF files and extract entity data."""
        # Read raw data from DXF files
        self.Data3D = self._read_dxf(filepath_3D)
        self.DataSection = self._read_dxf(filepath_Sections)

        # Extract entity data from 3D model
        self.Elements = self._give_LINEs(self.Data3D)
        self.BoundConditions = self._give_POINTs(self.Data3D)
        self.SlabInfo = self._give_LWPOLYLINEs(self.Data3D)

        # Extract section data
        self.SectionCoords = self._give_LWPOLYLINEs(self.DataSection)
        self.SectionLongReinf = self._give_CIRCLEs(self.DataSection)
        self.SectionTransReinf = self._give_MTEXTs(self.DataSection)

    def _read_dxf(self, filepath: str) -> list:
        """
        Read and parse DXF file content with error handling.

        Parameters
        ----------
        filepath : str
            Path to the DXF file

        Returns
        -------
        list
            Strings representing DXF file lines
        """
        try:
            with open(filepath, "r", encoding='utf-8') as file:
                file_data = file.read()
                return file_data.split('\n')
        except FileNotFoundError:
            raise FileNotFoundError(f"DXF file not found: {filepath}")
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(filepath, "r", encoding='latin-1') as file:
                    file_data = file.read()
                    return file_data.split('\n')
            except Exception as e:
                raise UnicodeDecodeError(
                    f"Could not decode DXF file {filepath}: {e}")

    def _find_indices(self, to_check: list, to_find: str) -> list:
        """
        Find all indices of a specific value in a list efficiently.

        Parameters
        ----------
        to_check : list
            List to search within
        to_find : str
            Value to find in the list

        Returns
        -------
        list
            Indices where the value was found
        """
        return [idx for idx, value in enumerate(to_check) if value == to_find]

    def _extract_entity_data(self, data: list, entity_type: str,
                             group_codes: dict) -> dict:
        """
        Generic method to extract entity data from DXF with error handling.

        Args:
            data: DXF file data as list of strings
            entity_type: Entity type (LINE, POINT, etc.)
            group_codes: Mapping of property names to group codes

        Returns:
            Dictionary containing extracted entity data
        """
        exc_idxs = self._find_indices(data, self.ENTITY_TYPE_CODE)
        entity_idxs = self._find_indices(data, entity_type)

        results = {key: [] for key in group_codes.keys()}

        for i in entity_idxs:
            try:
                iend = min(filter(lambda x: x > i, exc_idxs))
                cur_entity = data[i:iend]

                for prop_name, group_code in group_codes.items():
                    try:
                        idx = cur_entity.index(group_code)
                        value = cur_entity[idx + 1]

                        # Handle coordinate values
                        if prop_name in ['X', 'Y', 'Z', 'Start X', 'Start Y',
                                         'Start Z', 'End X', 'End Y', 'End Z',
                                         'Diameter', 'Radius']:
                            results[prop_name].append(
                                round(float(value), self.COORDINATE_PRECISION))
                        else:
                            results[prop_name].append(value)

                    except (ValueError, IndexError):
                        # Handle missing optional fields
                        if prop_name == 'Z':
                            results[prop_name].append(0.0)
                        else:
                            results[prop_name].append(None)

            except (ValueError, IndexError) as e:
                print(f"Warning: Error processing {entity_type} entity: {e}")
                continue

        return results

    def _give_LINEs(self, data: list) -> dict:
        """
        Extract LINE entity data (element sections) from DXF data.

        Args:
            data: DXF file data as list of strings

        Returns:
            Dictionary containing LINE entity information
        """
        group_codes = {
            'ElementSection': self.LAYER_CODE,
            'Start X': ' 10', 'Start Y': ' 20', 'Start Z': ' 30',
            'End X': ' 11', 'End Y': ' 21', 'End Z': ' 31'
        }

        return self._extract_entity_data(data, 'LINE', group_codes)

    def _give_POINTs(self, data: list) -> dict:
        """
        Extract POINT entity data (boundary conditions) from DXF data.

        Args:
            data: DXF file data as list of strings

        Returns:
            Dictionary containing POINT entity information
        """
        group_codes = {
            'BoundCond': self.LAYER_CODE,
            'X': ' 10', 'Y': ' 20', 'Z': ' 30'
        }

        return self._extract_entity_data(data, 'POINT', group_codes)

    def _give_LWPOLYLINEs(self, data: list) -> dict:
        """
        Extract LWPOLYLINE entity data (slab info/section coords) from DXF.

        Args:
            data: DXF file data as list of strings

        Returns:
            Dictionary containing LWPOLYLINE entity information
        """
        exc_idxs = self._find_indices(data, self.ENTITY_TYPE_CODE)
        lwp_idxs = self._find_indices(data, 'LWPOLYLINE')

        lwps, lwp_xs, lwp_ys, lwp_zs = [], [], [], []

        for i in lwp_idxs:
            try:
                iend = min(filter(lambda x: x > i, exc_idxs))
                cur_lwp = data[i:iend]

                ver_num = int(cur_lwp[cur_lwp.index(' 90') + 1])
                layer_idx = cur_lwp.index(self.LAYER_CODE)
                lwp_id = cur_lwp[layer_idx + 1]

                x_idxs = self._find_indices(cur_lwp, ' 10')
                y_idxs = self._find_indices(cur_lwp, ' 20')

                try:
                    z_idx = cur_lwp.index(' 38')
                    z_val = round(float(cur_lwp[z_idx + 1]),
                                  self.COORDINATE_PRECISION)
                except ValueError:
                    z_val = 0.0

                for vi in range(ver_num):
                    lwps.append(lwp_id)
                    lwp_xs.append(round(float(cur_lwp[x_idxs[vi] + 1]),
                                        self.COORDINATE_PRECISION))
                    lwp_ys.append(round(float(cur_lwp[y_idxs[vi] + 1]),
                                        self.COORDINATE_PRECISION))
                    lwp_zs.append(z_val)

            except (ValueError, IndexError) as e:
                print(f"Warning: Error processing LWPOLYLINE entity: {e}")
                continue

        return {
            'Name': np.array(lwps),
            'X': np.array(lwp_xs),
            'Y': np.array(lwp_ys),
            'Z': np.array(lwp_zs)
        }

    def _give_CIRCLEs(self, data: list) -> dict:
        """
        Extract CIRCLE entity data (longitudinal reinforcement) from DXF.

        Args:
            data: DXF file data as list of strings

        Returns:
            Dictionary containing CIRCLE entity information
        """
        group_codes = {
            'Name': self.LAYER_CODE,
            'X': ' 10', 'Y': ' 20', 'Radius': ' 40'
        }

        circle_data = self._extract_entity_data(data, 'CIRCLE', group_codes)

        # Convert radius to diameter
        if 'Radius' in circle_data:
            circle_data['Diameter'] = [float(r) * 2 if r is not None else 0
                                       for r in circle_data['Radius']]
            del circle_data['Radius']

        # Convert to numpy arrays
        for key in circle_data:
            circle_data[key] = np.array(circle_data[key])

        return circle_data

    def _give_MTEXTs(self, data: list) -> dict:
        """
        Extract MTEXT entity data (transverse reinforcement info) from DXF.

        Args:
            data: DXF file data as list of strings

        Returns:
            Dictionary containing MTEXT entity information
        """
        group_codes = {
            'Name': self.LAYER_CODE,
            'Content': '  1'
        }

        return self._extract_entity_data(data, 'MTEXT', group_codes)

    def SaveOut(self):
        """
        Process extracted data and prepare for output.

        This method orchestrates the processing workflow, including
        adding beam-column joints, reorganizing infills, and calculating
        mass/load distributions.
        """
        try:
            if not self._validate_required_data():
                raise ValueError("Required data is missing for processing")

            self.AddBCJ()
            self.ReorganiseInfills()
            self.AddMassLoadSlab()

        except Exception as e:
            print(f"Error in SaveOut processing: {e}")
            raise

    def _validate_required_data(self) -> bool:
        """
        Validate that required data is present for processing.

        Returns:
            True if all required data is present, False otherwise
        """
        required_attrs = ['Elements', 'BoundConditions', 'SlabInfo',
                          'SectionCoords', 'SectionLongReinf',
                          'SectionTransReinf']

        for attr in required_attrs:
            if not hasattr(self, attr):
                print(f"Warning: Missing required attribute: {attr}")
                return False

        return True

    def AddMassLoadSlab(self):
        """
        Calculate and assign point loads and masses based on slab areas.

        Distributes slab loads to column points based on proximity and
        tributary area calculations.
        """
        if not self._validate_load_calculation_data():
            return

        elevations = np.unique(self.Points['Coordinates'][:, -1])
        total_point_num = len(self.Points['ID'])

        # Initialize load arrays
        load_data = self._initialize_load_arrays(total_point_num)

        # Process each elevation
        for elevation in elevations:
            if elevation in self.SlabInfo['Z']:
                self._process_elevation_loads(elevation, load_data)

        # Distribute cumulative loads from top to bottom
        self._distribute_cumulative_loads(elevations, load_data)

        # Assign final loads to points
        self._assign_loads_to_points(load_data)

    def _validate_load_calculation_data(self) -> bool:
        """Validate required data for load calculations."""
        required_attrs = ['Points', 'Frames', 'SlabInfo', 'AreaLoad']
        missing_attrs = [attr for attr in required_attrs
                         if not hasattr(self, attr)]

        if missing_attrs:
            print(f"Warning: Missing attributes for load calculation: "
                  f"{missing_attrs}")
            return False
        return True

    def _initialize_load_arrays(self, size: int) -> dict:
        """Initialize arrays for load calculations."""
        return {
            'load_factor': np.zeros(size),
            'load_elevation': np.zeros(size),
            'load': np.zeros(size),
            'mass': np.zeros(size)
        }

    def _process_elevation_loads(self, elevation: float, load_data: dict):
        """Process loads for a specific elevation."""
        # Find column points at this elevation
        col_point_ids = self._find_column_points_at_elevation(elevation)

        if not col_point_ids:
            return

        col_point_coordinates = self._get_column_coordinates(col_point_ids)

        # Get slab information for current elevation
        slab_polygons, current_weight = self._process_slab_polygons(elevation)

        if not slab_polygons:
            return

        # Create mesh and distribute loads
        self._distribute_loads_to_columns(
            col_point_ids, col_point_coordinates, slab_polygons,
            current_weight, load_data)

    def _find_column_points_at_elevation(self, elevation: float) -> list:
        """Find column points at a specific elevation."""
        col_point_ids = []
        ele_map = np.where(self.Points['Coordinates'][:, -1] == elevation)
        ele_points = self.Points['ID'][ele_map]

        for point_id in ele_points:
            frames_map = np.where((self.Frames['i_ID'] == point_id) |
                                  (self.Frames['j_ID'] == point_id))
            frame_directions = self.Frames['Direction'][frames_map]
            if 3 in frame_directions:  # Direction 3 = vertical (column)
                col_point_ids.append(point_id)

        return col_point_ids

    def _get_column_coordinates(self, col_point_ids: list) -> np.ndarray:
        """Get coordinates of column points."""
        col_point_map = np.where(np.isin(self.Points['ID'], col_point_ids))
        return self.Points['Coordinates'][col_point_map]

    def _process_slab_polygons(self, elevation: float) -> tuple:
        """Process slab polygons at a given elevation."""
        cur_elevation_map = np.where(self.SlabInfo['Z'] == elevation)
        cur_ele_slab_id = self.SlabInfo['Name'][cur_elevation_map]
        cur_ele_slab_x = self.SlabInfo['X'][cur_elevation_map]
        cur_ele_slab_y = self.SlabInfo['Y'][cur_elevation_map]

        current_weight = 0.0
        sorted_polygons = []

        for slab_id in np.unique(cur_ele_slab_id):
            sign = 1 if 'Slab' in slab_id else -1  # Negative for gaps

            polygon_map = np.where(cur_ele_slab_id == slab_id)
            polygon_x = cur_ele_slab_x[polygon_map]
            polygon_y = cur_ele_slab_y[polygon_map]
            polygon = list(zip(polygon_x, polygon_y))

            polygon_area = gi.calculate_polygon_area(polygon)
            current_weight += (polygon_area * self.AreaLoad * sign * 0.0001)

            # Sort polygon vertices for consistent processing
            sorted_polygon = sorted(polygon,
                                    key=lambda point: gi.polar_angle(point, polygon))
            sorted_polygons.append(sorted_polygon)

        return sorted_polygons, current_weight

    def _distribute_loads_to_columns(self, col_point_ids: list,
                                     col_coordinates: np.ndarray,
                                     polygons: list, weight: float,
                                     load_data: dict):
        """Distribute loads to columns based on proximity."""
        # Define mesh bounds
        x_coords = [coord for poly in polygons for coord, _ in poly]
        y_coords = [coord for poly in polygons for _, coord in poly]

        x_min = min(min(x_coords), col_coordinates[:, 0].min())
        x_max = max(max(x_coords), col_coordinates[:, 0].max())
        y_min = min(min(y_coords), col_coordinates[:, 1].min())
        y_max = max(max(y_coords), col_coordinates[:, 1].max())

        # Create mesh
        x_values = np.linspace(x_min, x_max,
                               max(1, int((x_max - x_min) / self.DEFAULT_GRID_SPACING)))
        y_values = np.linspace(y_min, y_max,
                               max(1, int((y_max - y_min) / self.DEFAULT_GRID_SPACING)))
        x_mesh, y_mesh = np.meshgrid(x_values, y_values)
        x_mesh = x_mesh.flatten()
        y_mesh = y_mesh.flatten()

        # Calculate weights for each mesh point
        w_mesh = np.zeros(y_mesh.shape)
        for polygon in polygons:
            w_mesh += gi.isin_polygon_vector(x_mesh, y_mesh, polygon)

        # Find nearest columns for each mesh point
        xy_mesh = np.array(list(zip(x_mesh, y_mesh)))
        mesh_expanded = xy_mesh[:, np.newaxis, :]
        col_expanded = col_coordinates[:, [0, 1]][np.newaxis, :, :]
        distances = np.linalg.norm(mesh_expanded - col_expanded, axis=2)
        nearest_idx = np.argmin(distances, axis=1)
        rel_point = np.asarray(col_point_ids)[nearest_idx]

        # Distribute loads
        total_weight = sum(w_mesh)
        if total_weight > 0:
            for point_id in col_point_ids:
                weight_current = sum(w_mesh[rel_point == point_id])
                ratio = round(weight_current / total_weight, 2)
                point_map = (self.Points['ID'] == point_id)

                load_data['load_factor'][point_map] = ratio
                load_data['load_elevation'][point_map] = ratio * weight
                load_data['mass'][point_map] = ratio * weight / self.GRAVITY

    def _distribute_cumulative_loads(self, elevations: np.ndarray,
                                   load_data: dict):
        """Distribute cumulative loads from higher elevations downwards."""
        for elevation in np.sort(elevations)[::-1]:  # Top to bottom
            cur_elevation_map = np.where(
                self.Points['Coordinates'][:, -1] == elevation)
            cur_ele_point_id = self.Points['ID'][cur_elevation_map]
            cur_ele_point_coords = self.Points['Coordinates'][cur_elevation_map, :]

            for pi, point_id in enumerate(cur_ele_point_id):
                # Find all points directly above current point
                x_map = (self.Points['Coordinates'][:, 0] ==
                        cur_ele_point_coords[0, pi, 0])
                y_map = (self.Points['Coordinates'][:, 1] ==
                        cur_ele_point_coords[0, pi, 1])
                z_map1 = (self.Points['Coordinates'][:, 2] ==
                         cur_ele_point_coords[0, pi, 2])
                z_map2 = (self.Points['Coordinates'][:, 2] >=
                         cur_ele_point_coords[0, pi, 2])

                load_map1 = x_map & y_map & z_map1  # Current point
                load_map2 = x_map & y_map & z_map2  # Points at/above

                load_current = sum(load_data['load_elevation'][load_map2])
                load_data['load'][load_map1] = load_current

    def _assign_loads_to_points(self, load_data: dict):
        """Assign calculated loads to point data structure."""
        self.Points['LF'] = load_data['load_factor']
        self.Points['LoadElevation'] = load_data['load_elevation']
        self.Points['Load'] = load_data['load']
        self.Points['Mass'] = load_data['mass']

        # Optional: Print load summary
        total_applied = np.sum(load_data['load_elevation'])
        print(f"Total load applied: {total_applied:.2f}")

    def ReorganiseBoundCond(self):
        """
        Reorganize boundary condition data and assign to structural points.

        Currently only supports fixed boundary conditions. Maps boundary
        condition coordinates to point IDs and sets all DOF constraints to 1.

        Returns:
            self: For method chaining
        """
        # self.ReorganisePoints()
        # self.AddBCJ()

        # TODO: This works only for fix
        # Fix Point Info
        IDs = []
        for _, CurFix in self.BoundConditions.iterrows():
            IDs.append(self.Points.loc[(self.Points['X'] == CurFix['X']) &
                                       (self.Points['Y'] == CurFix['Y']) &
                                       (self.Points['Z'] == CurFix['Z']),
                                       'PointID'].values[0])
        self.BCData = pd.DataFrame(data={'ID': IDs})
        self.BCData['u1'] = 1
        self.BCData['u2'] = 1
        self.BCData['u3'] = 1
        self.BCData['r1'] = 1
        self.BCData['r2'] = 1
        self.BCData['r3'] = 1
        return self

    def ReorganiseSections(self):
        """
        Process section data from DXF to create material and geometric
        properties.

        Extracts section information including dimensions, reinforcement
        details, material properties, and coordinate data for structural
        elements.
        """
        try:
            Sections = {}
            SectionNames = self.SectionTransReinf['Name']

            for ci in range(len(SectionNames)):
                cur_sec_name = SectionNames[ci]
                Sections[cur_sec_name] = self._create_section_data(
                    ci, cur_sec_name)

            self.Sections = Sections

        except Exception as e:
            print(f"Error in ReorganiseSections: {e}")
            raise

    def _create_section_data(self, section_index: int,
                             section_name: str) -> dict:
        """
        Create section data dictionary for a given section.

        Args:
            section_index: Index of the section in the data arrays
            section_name: Name/ID of the section

        Returns:
            Dictionary containing all section properties
        """
        section_data = {'ID': section_name}

        # Parse transverse reinforcement data
        content = self.SectionTransReinf['Content'][section_index]
        section_data.update(self._parse_reinforcement_content(content))

        # Process coordinates
        coord_data = self._process_section_coordinates(section_name)
        section_data.update(coord_data)

        # Process longitudinal reinforcement
        reinf_data = self._process_longitudinal_reinforcement(
            section_name, coord_data['CXCenter'], coord_data['CYCenter'])
        section_data.update(reinf_data)

        # Add material properties
        section_data.update(self._get_material_properties())

        return section_data

    def _parse_reinforcement_content(self, content: str) -> dict:
        """Parse reinforcement content string."""
        try:
            # Handle case where content might just be a number (skip it)
            if '/' not in content:
                # Return default values for content that doesn't match expected format
                return {
                    'phi_T': 10,  # Default stirrup diameter
                    's': 200,    # Default spacing
                    'NumofStrZDir': 2,  # Default number of stirrups in Z direction
                    'NumofStrYDir': 2   # Default number of stirrups in Y direction
                }
                
            sub_cont1 = content.split('/')
            phi_t = int(sub_cont1[0].replace('Q', ''))
            
            sub_cont2 = sub_cont1[1].split('|')
            s = float(sub_cont2[0].strip())
            
            str_contents = sub_cont2[1].strip().split('-')
            
            return {
                'phi_T': phi_t,
                's': s,
                'NumofStrZDir': int(str_contents[0]),
                'NumofStrYDir': int(str_contents[1])
            }
        except (ValueError, IndexError) as e:
            # Return default values if parsing fails
            print(f"Warning: Could not parse reinforcement content '{content}', using defaults: {e}")
            return {
                'phi_T': 10,
                's': 200,
                'NumofStrZDir': 2,
                'NumofStrYDir': 2
            }

    def _process_section_coordinates(self, section_name: str) -> dict:
        """Process section coordinate data."""
        c_names = self.SectionCoords['Name']
        idx = [i for i, x in enumerate(c_names) if x == section_name]

        if not idx:
            raise ValueError(
                f"No coordinates found for section: {section_name}")

        cx = self.SectionCoords['X'][idx]
        cx_center = np.mean(cx)
        cx -= cx_center

        cy = self.SectionCoords['Y'][idx]
        cy_center = np.mean(cy)
        cy -= cy_center

        return {
            'b': max(cx) - min(cx),
            'h': max(cy) - min(cy),
            'Coords': np.array(list(zip(cx, cy))),
            'CXCenter': cx_center,
            'CYCenter': cy_center
        }

    def _process_longitudinal_reinforcement(self, section_name: str,
                                            cx_center: float,
                                            cy_center: float) -> dict:
        """Process longitudinal reinforcement data."""
        lr_names = self.SectionLongReinf['Name']
        idx = [i for i, x in enumerate(lr_names) if x == section_name]

        if not idx:
            return {'ReinfL': np.array([]), 'Cover': 0.0}

        lrx = self.SectionLongReinf['X'][idx] - cx_center
        lry = self.SectionLongReinf['Y'][idx] - cy_center
        lrr = self.SectionLongReinf['Diameter'][idx]

        # Calculate cover
        try:
            z_r_max = max(lrx)
            idx_max = np.where(lrx == z_r_max)[0]
            phi_l = max(lrr[idx_max])
            
            # Handle various data types and formats for phi_l
            if hasattr(phi_l, '__len__') and not isinstance(phi_l, str):
                phi_l = phi_l[0] if len(phi_l) > 0 else 10.0
            
            # Convert to float, handling potential string formatting issues
            if isinstance(phi_l, str):
                # Remove any extra characters and take the first valid number
                import re
                numbers = re.findall(r'\d+\.?\d*', str(phi_l))
                phi_l = float(numbers[0]) if numbers else 10.0
            else:
                phi_l = float(phi_l)
            
            z_max = max(self.SectionCoords['X'][
                [i for i, x in enumerate(self.SectionCoords['Name'])
                 if x == section_name]] - cx_center)
            cover = z_max - z_r_max + 0.5 * phi_l
            
        except Exception as e:
            print(f"Warning: Error calculating cover for section {section_name}: {e}")
            cover = 25.0  # Default cover value

        return {
            'ReinfL': np.array(list(zip(lrx, lry, lrr))),
            'Cover': cover
        }

    def _get_material_properties(self) -> dict:
        """Get material properties from configuration."""
        return {
            'eps_y': self.eps_y,
            'fy': self.fy,
            'fyw': self.fyw,
            'eps_u': self.eps_su,
            'fu': self.fsu,
            'Es': self.Es,
            'fc0': self.fc0,
            'Ec': self.Ec,
            'Hook': self.Hook
        }

    def ReorganiseInfills(self):
        if 'Type' not in self.Points:
            self.AddBCJ()
        if 'Sections' not in locals():
            self.ReorganiseSections()

        self.Infills['CSE_Frame'] = []
        self.Infills['CSE_Node'] = []
        self.Infills['BSE_Frame'] = []
        self.Infills['BSE_Node'] = []
        self.Infills['CNO_Frame'] = []
        self.Infills['CNO_Node'] = []
        self.Infills['BNO_Frame'] = []
        self.Infills['BNO_Node'] = []

        InfillIDs = self.Infills['ID']
        i_Z = []
        Bs = []
        Hs = []
        bbs = []
        hbs = []
        bcs = []
        hcs = []
        Ecs = []
        for ci in range(len(InfillIDs)):
            i_ID = self.Infills['i_ID'][ci]
            j_ID = self.Infills['j_ID'][ci]
            Iidx = np.where(self.Points['ID'] == i_ID)[0][0]
            Jidx = np.where(self.Points['ID'] == j_ID)[0][0]
            iCoords = self.Points['Coordinates'][Iidx]
            jCoords = self.Points['Coordinates'][Jidx]

            # Define Lower Altitude
            i_Z.append(iCoords[-1])
            # Define B and H
            if self.Infills['Direction'][ci] == 4:
                Bs.append(abs(iCoords[0]-jCoords[0]))
            else:
                Bs.append(abs(iCoords[1]-jCoords[1]))
            Hs.append(abs(iCoords[-1]-jCoords[-1]))

            IConFrameMap = np.where((self.Frames['i_ID'] == i_ID) |
                                    (self.Frames['j_ID'] == i_ID))
            IConFramesID = self.Frames['ID'][IConFrameMap]
            IConFramesDir = self.Frames['Direction'][IConFrameMap]
            IConFramesIid = self.Frames['i_ID'][IConFrameMap]
            IConFramesJid = self.Frames['j_ID'][IConFrameMap]
            CSE_FrameID = IConFramesID[np.where((IConFramesIid == i_ID) &
                                                (IConFramesDir == 3))]
            iType = self.Points['Type'][Iidx]
            if iType == 'BCJ':
                CSE_NodeID = int(float(f'1{i_ID}'))
            else:
                CSE_NodeID = int(i_ID)

            JConFrameMap = np.where((self.Frames['i_ID'] == j_ID) |
                                    (self.Frames['j_ID'] == j_ID))
            JConFramesID = self.Frames['ID'][JConFrameMap]

            JConFramesDir = self.Frames['Direction'][JConFrameMap]
            JConFramesIid = self.Frames['i_ID'][JConFrameMap]
            JConFramesJid = self.Frames['j_ID'][JConFrameMap]
            CNO_FrameID = JConFramesID[np.where((JConFramesJid == j_ID) &
                                                (JConFramesDir == 3))]

            jType = self.Points['Type'][Jidx]
            if jType == 'BCJ':
                CNO_NodeID = int(float(f'1{j_ID}'))
            else:
                CNO_NodeID = int(j_ID)

            IXXConFramesID = IConFramesID[np.where(IConFramesDir == 1)]
            JXXConFramesID = JConFramesID[np.where(JConFramesDir == 1)]
            IXXConFramesIid = IConFramesIid[np.where(IConFramesDir == 1)]
            IXXConFramesJid = IConFramesJid[np.where(IConFramesDir == 1)]
            JXXConFramesIid = JConFramesIid[np.where(JConFramesDir == 1)]
            JXXConFramesJid = JConFramesJid[np.where(JConFramesDir == 1)]

            IYYConFramesID = IConFramesID[np.where(IConFramesDir == 2)]
            JYYConFramesID = JConFramesID[np.where(JConFramesDir == 2)]
            IYYConFramesIid = IConFramesIid[np.where(IConFramesDir == 2)]
            IYYConFramesJid = IConFramesJid[np.where(IConFramesDir == 2)]
            JYYConFramesIid = JConFramesIid[np.where(JConFramesDir == 2)]
            JYYConFramesJid = JConFramesJid[np.where(JConFramesDir == 2)]

            DirectionMapping = {
                '4': {
                    'CoordType': 0,
                    'Nodes': {
                        'Dir1': {
                            'BSE': [IXXConFramesID, IXXConFramesIid, i_ID,
                                    iType, i_ID],
                            'BNO': [JXXConFramesID, JXXConFramesJid, j_ID,
                                    jType, j_ID]
                        },
                        'Dir2': {
                            'BSE': [IXXConFramesID, IXXConFramesJid, i_ID,
                                    iType, i_ID],
                            'BNO': [JXXConFramesID, JXXConFramesIid, j_ID,
                                    jType, j_ID]
                        }
                    }
                },
                '5': {
                    'CoordType': 1,
                    'Nodes': {
                        'Dir1': {
                            'BSE': [IYYConFramesID, IYYConFramesIid, i_ID,
                                    iType, i_ID],
                            'BNO': [JYYConFramesID, JYYConFramesJid, j_ID,
                                    jType, j_ID]
                        },
                        'Dir2': {
                            'BSE': [IYYConFramesID, IYYConFramesJid, i_ID,
                                    iType, i_ID],
                            'BNO': [JYYConFramesID, JYYConFramesIid, j_ID,
                                    jType, j_ID]
                        }
                    }
                }
            }
            DBV = DirectionMapping[str(self.Infills['Direction'][ci])]
            CT = DBV['CoordType']
            ValsDummy = {'BSE_Frame': 0,
                         'BNO_Frame': 0,
                         'BSE_Node': 0,
                         'BNO_Node': 0}

            if iCoords[CT] < jCoords[CT]:
                for NodeType in ['BSE', 'BNO']:
                    FramesID = DBV['Nodes']['Dir1'][NodeType][0]
                    FramesNodeIDs = DBV['Nodes']['Dir1'][NodeType][1]
                    NodeID = DBV['Nodes']['Dir1'][NodeType][2]
                    NType = DBV['Nodes']['Dir1'][NodeType][3]
                    RN = DBV['Nodes']['Dir1'][NodeType][4]
                    if NodeID in FramesNodeIDs:
                        FIDMap = np.where(FramesNodeIDs == NodeID)
                        ValsDummy[NodeType+'_Frame'] = int(FramesID[FIDMap])
                        if NType == 'BCJ':
                            ValsDummy[NodeType+'_Node'] = int(float(f"6{RN}"))
                        else:
                            ValsDummy[NodeType+'_Node'] = int(RN)
            elif iCoords[CT] > jCoords[CT]:
                for NodeType in ['BSE', 'BNO']:
                    FramesID = DBV['Nodes']['Dir2'][NodeType][0]
                    FramesNodeIDs = DBV['Nodes']['Dir2'][NodeType][1]
                    NodeID = DBV['Nodes']['Dir2'][NodeType][2]
                    NType = DBV['Nodes']['Dir2'][NodeType][3]
                    RN = DBV['Nodes']['Dir2'][NodeType][4]
                    if NodeID in FramesNodeIDs:
                        FIDMap = np.where(FramesNodeIDs == NodeID)
                        ValsDummy[NodeType+'_Frame'] = int(FramesID[FIDMap])
                        if NType == 'BCJ':
                            ValsDummy[NodeType+'_Node'] = int(float(f"6{RN}"))
                        else:
                            ValsDummy[NodeType+'_Node'] = int(RN)

            BeamIDCandidates = [ValsDummy['BNO_Frame'], ValsDummy['BSE_Node']]
            ColumnIDCandidates = [CNO_FrameID,  CSE_FrameID]

            BeamMap = np.where(np.isin(self.Frames['ID'], BeamIDCandidates))[0]
            if len(BeamMap) == 0:
                bb = 0
                hb = 0
            else:
                BeamCandSections = np.array(self.Frames['Type'])[BeamMap]
                Candbb = [self.Sections[bi]['b'] for bi in BeamCandSections]
                bb = min(Candbb)

                Candhb = [self.Sections[bi]['h'] for bi in BeamCandSections]
                hb = min(Candhb)

            ColMap = np.where(np.isin(self.Frames['ID'], ColumnIDCandidates))
            ColumnCandSections = np.array(self.Frames['Type'])[ColMap]

            if self.Infills['Direction'][ci] == 4:
                bc = min(self.Sections[ColumnCandSections[0]]['b'],
                         self.Sections[ColumnCandSections[1]]['b'])
                hc = min(self.Sections[ColumnCandSections[0]]['h'],
                         self.Sections[ColumnCandSections[1]]['h'])
            elif self.Infills['Direction'][ci] == 5:
                bc = min(self.Sections[ColumnCandSections[0]]['h'],
                         self.Sections[ColumnCandSections[1]]['h'])
                hc = min(self.Sections[ColumnCandSections[0]]['b'],
                         self.Sections[ColumnCandSections[1]]['b'])

            if iType == 'BCJ':
                self.Infills['i_ID'][ci] = int(float(f"1{i_ID}"))

            if jType == 'BCJ':
                self.Infills['j_ID'][ci] = int(float(f"1{j_ID}"))

            bbs.append(bb)
            hbs.append(hb)
            bcs.append(bc)
            hcs.append(hc)

            Ecs.append(self.Sections[ColumnCandSections[0]]['Ec'])

            self.Infills['CSE_Frame'].append(CSE_FrameID)
            self.Infills['CSE_Node'].append(CSE_NodeID)
            self.Infills['CNO_Frame'].append(CNO_FrameID)
            self.Infills['CNO_Node'].append(CNO_NodeID)

            self.Infills['BSE_Frame'].append(ValsDummy['BSE_Frame'])
            self.Infills['BSE_Node'].append(ValsDummy['BSE_Node'])
            self.Infills['BNO_Frame'].append(ValsDummy['BNO_Frame'])
            self.Infills['BNO_Node'].append(ValsDummy['BNO_Node'])
        self.Infills['i_Z'] = i_Z
        self.Infills['H'] = Hs
        self.Infills['B'] = Bs
        self.Infills['bb'] = bbs
        self.Infills['hb'] = hbs
        self.Infills['bc'] = bcs
        self.Infills['hc'] = hcs
        self.Infills['Ec'] = Ecs

    def AddBCJ(self):
        """
        Add beam-column joint (BCJ) type classification to structural points.

        Analyses each point to determine if it's a beam-column joint based on
        connected frame elements. Points connected to both columns and beams
        are classified as BCJ.
        """
        if not hasattr(self, 'Frames'):
            self.ReorganiseElements()

        point_types = []
        frame_i_id = self.Frames['i_ID']
        frame_j_id = self.Frames['j_ID']
        frame_dirs = self.Frames['Direction']
        point_ids = self.Points['ID']

        self.Points['Type'] = ''

        for ci in range(len(point_ids)):
            cur_point_id = point_ids[ci]

            # Find frames connected to current point
            connected_mask = ((frame_i_id == cur_point_id) |
                              (frame_j_id == cur_point_id))
            cur_frame_dirs = frame_dirs[connected_mask]

            # Check for column and beam connections
            col_exist = 3 in cur_frame_dirs  # Direction 3 = vertical (column)
            beam_exist = ((2 in cur_frame_dirs) | (1 in cur_frame_dirs))

            if col_exist and beam_exist:
                point_types.append('BCJ')
            else:
                point_types.append('NONBCJ')

        self.Points['Type'] = point_types

    def ReorganiseElements(self):
        """
        Reorganise DXF elements into frames and infills with proper connectivity.

        Processes raw element data to create structural frames and infill
        elements, establishing node connectivity and element directions based
        on coordinate differences.
        """
        if not hasattr(self, 'Points'):
            self.ReorganisePoints()

        try:
            # Extract start and end coordinates
            elements_start, elements_end = self._extract_element_coordinates()

            # Find node indices and IDs
            element_i_idx, element_j_idx = self._find_element_node_indices(
                elements_start, elements_end)
            element_i_id = self.Points['ID'][element_i_idx]
            element_j_id = self.Points['ID'][element_j_idx]

            # Ensure consistent node ordering (i < j)
            element_i_id_final = np.minimum(element_i_id, element_j_id)
            element_j_id_final = np.maximum(element_i_id, element_j_id)

            # Determine element directions
            element_directions = self._determine_element_directions(
                elements_start, elements_end)

            # Separate frames from infills
            self._separate_frames_and_infills(
                element_i_id_final, element_j_id_final, element_directions)

        except Exception as e:
            print(f"Error in ReorganiseElements: {e}")
            raise

    def _extract_element_coordinates(self) -> tuple:
        """Extract start and end coordinates from elements."""
        elements_start = np.array(list(zip(
            self.Elements['Start X'],
            self.Elements['Start Y'],
            self.Elements['Start Z'])))

        elements_end = np.array(list(zip(
            self.Elements['End X'],
            self.Elements['End Y'],
            self.Elements['End Z'])))

        return elements_start, elements_end

    def _find_element_node_indices(self, elements_start: np.ndarray,
                                   elements_end: np.ndarray) -> tuple:
        """Find node indices for element start and end points."""
        element_i_idx = []
        element_j_idx = []

        for start_coord in elements_start:
            idx = np.where((self.Points['Coordinates'] == start_coord)
                          .all(axis=1))[0][0]
            element_i_idx.append(idx)

        for end_coord in elements_end:
            idx = np.where((self.Points['Coordinates'] == end_coord)
                          .all(axis=1))[0][0]
            element_j_idx.append(idx)

        return element_i_idx, element_j_idx

    def _determine_element_directions(self, elements_start: np.ndarray,
                                      elements_end: np.ndarray) -> np.ndarray:
        """
        Determine element directions based on coordinate differences.

        Returns
        -------
        np.ndarray
            Array with direction codes:
            1 = X-direction (beam)
            2 = Y-direction (beam)
            3 = Z-direction (column)
            4 = XZ-direction (infill)
            5 = YZ-direction (infill)
        """
        equality_mat = (elements_start == elements_end).astype(int)
        element_directions = np.array([99] * equality_mat.shape[0])

        # Direction mappings
        direction_patterns = {
            (0, 1, 1): 1,  # X-direction
            (1, 0, 1): 2,  # Y-direction
            (1, 1, 0): 3,  # Z-direction
            (0, 1, 0): 4,  # XZ-direction (infill)
            (1, 0, 0): 5   # YZ-direction (infill)
        }

        for pattern, direction in direction_patterns.items():
            mask = (equality_mat == list(pattern)).all(axis=1)
            element_directions[mask] = direction

        return element_directions

    def _separate_frames_and_infills(self, element_i_id: np.ndarray,
                                     element_j_id: np.ndarray,
                                     element_directions: np.ndarray):
        """Separate elements into frames and infills."""
        # Separate frames (directions 1, 2, 3) from infills (4, 5)
        frame_mask = np.isin(element_directions, [1, 2, 3])
        infill_mask = np.isin(element_directions, [4, 5])

        # Process frames
        if np.any(frame_mask):
            self._process_frames(element_i_id[frame_mask],
                                 element_j_id[frame_mask],
                                 element_directions[frame_mask],
                                 frame_mask)

        # Process infills
        if np.any(infill_mask):
            self._process_infills(element_i_id[infill_mask],
                                  element_j_id[infill_mask],
                                  element_directions[infill_mask],
                                  infill_mask)

    def _process_frames(self, frame_i_ids: np.ndarray, frame_j_ids: np.ndarray,
                        frame_directions: np.ndarray, frame_mask: np.ndarray):
        """Process frame elements."""
        frame_types = [self.Elements['ElementSection'][i]
                      for i, is_frame in enumerate(frame_mask) if is_frame]
        frame_ids = frame_j_ids + 1000 * frame_i_ids

        # Determine section types for beam ends
        section_type_i, section_type_j = self._determine_beam_section_types(
            frame_i_ids, frame_j_ids, frame_ids, frame_directions, frame_types)

        self.Frames = {
            'i_ID': frame_i_ids,
            'j_ID': frame_j_ids,
            'ID': frame_ids,
            'Type': frame_types,
            'Direction': frame_directions,
            'SectionTypI': section_type_i,
            'SectionTypJ': section_type_j
        }

    def _process_infills(self, infill_i_ids: np.ndarray,
                         infill_j_ids: np.ndarray,
                         infill_directions: np.ndarray,
                         infill_mask: np.ndarray):
        """Process infill elements."""
        infill_types = [self.Elements['ElementSection'][i]
                        for i, is_infill in enumerate(infill_mask) if is_infill]
        infill_ids = infill_i_ids + 1000 * infill_j_ids

        self.Infills = {
            'i_ID': infill_i_ids,
            'j_ID': infill_j_ids,
            'ID': infill_ids,
            'Type': infill_types,
            'Direction': infill_directions
        }

    def _determine_beam_section_types(self, frame_i_ids: np.ndarray,
                                      frame_j_ids: np.ndarray,
                                      frame_ids: np.ndarray,
                                      frame_directions: np.ndarray,
                                      frame_types: list) -> tuple:
        """Determine section types for beam ends (BCJ vs Mid sections)."""
        section_type_i = []
        section_type_j = []

        for ci in range(len(frame_ids)):
            if frame_directions[ci] == 3:  # Column
                section_type_i.append(frame_types[ci])
                section_type_j.append(frame_types[ci])
            else:  # Beam
                # Check i-end
                i_section = self._get_beam_end_section_type(
                    frame_i_ids[ci], frame_ids, frame_directions,
                    frame_types[ci], ci)
                section_type_i.append(i_section)

                # Check j-end
                j_section = self._get_beam_end_section_type(
                    frame_j_ids[ci], frame_ids, frame_directions,
                    frame_types[ci], ci)
                section_type_j.append(j_section)

        return section_type_i, section_type_j

    def _get_beam_end_section_type(self, node_id: int, 
                                    frame_ids: np.ndarray,
                                    frame_directions: np.ndarray,
                                    base_section_type: str,
                                    current_frame_idx: int) -> str:
        """
        Get section type for a beam end.

        Parameters
        ----------
        node_id : int
            Node identifier.
        frame_ids : np.ndarray
            Frame identifier array.
        frame_directions : np.ndarray
            Frame direction vectors.
        base_section_type : str
            Base section type.
        current_frame_idx : int
            Current frame index.

        Returns
        -------
        str
            Section type (BCJ or Mid section).
        """
        # Simplified logic for now - just return base section type
        # In a more complete implementation, this would check if the beam end
        # is connected to a column (BCJ) or continues (Mid section)
        return base_section_type

    def ReorganisePoints(self):
        """
        Extract and organise unique structural points from element data.

        Creates a comprehensive list of unique points by combining start/end
        coordinates from elements and boundary condition points. Assigns
        sequential IDs and initialises boundary condition arrays.
        """
        PointsX = (self.Elements['Start X'] +
                   self.Elements['End X'] +
                   self.BoundConditions['X'])
        PointsY = (self.Elements['Start Y'] +
                   self.Elements['End Y'] +
                   self.BoundConditions['Y'])
        PointsZ = (self.Elements['Start Z'] +
                   self.Elements['End Z'] +
                   self.BoundConditions['Z'])

        Points = np.array(list(set(zip(PointsX, PointsY, PointsZ))))
        sortidx = np.lexsort((Points[:, 1], Points[:, 0], Points[:, 2]))
        Points = Points[sortidx]
        IDs = list(range(100, 100+Points.shape[0]))

        BCs = np.zeros((Points.shape[0], 6))
        BCCoords = np.array(list(set(zip(self.BoundConditions['X'],
                                         self.BoundConditions['Y'],
                                         self.BoundConditions['Z']))))
        BCidx = np.where(np.isin(Points, BCCoords).sum(axis=1) == 3)
        BCs[BCidx, :] = 1
        self.Points = {'ID': np.array(IDs),
                       'Coordinates': Points,
                       'BoundryConditions': BCs}

    def Parse(self):
        """
        Main parsing method that orchestrates the complete workflow.

        This method should be called after initialization to process
        all data and prepare the structural model.
        """
        try:
            print("Starting DXF parsing workflow...")

            # Step 1: Reorganize points
            print("Step 1: Reorganizing points...")
            self.ReorganisePoints()

            # Step 2: Reorganize elements
            print("Step 2: Reorganizing elements...")
            self.ReorganiseElements()

            # Step 3: Add sections
            print("Step 3: Processing sections...")
            self.ReorganiseSections()

            # Step 4: Reorganize infills
            print("Step 4: Reorganizing infills...")
            self.ReorganiseInfills()

            # Step 5: Add mass/load to slabs
            print("Step 5: Adding mass and loads...")
            self.AddMassLoadSlab()

            print("DXF parsing completed successfully!")

        except Exception as e:
            print(f"Error during parsing: {e}")
            raise
