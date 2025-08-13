"""
Beam-column joint generation module for CAD2Sees.

Processes beam-column joint data and generates joint models using OpenSees
with appropriate stiffness and capacity properties.
"""

import numpy as np
from cad2sees.helpers import units
from .component_modeller import joint_modeller as jm


def create(point_data, frame_data, section_data, elastic_joint_flag=0):
    """
    Process beam-column joint data and generate joint models.

    Identifies beam-column joints from point data, determines location
    characteristics, and creates appropriate joint models with stiffness
    and capacity properties.

    Parameters
    ----------
    point_data : dict
        Point information including Type, ID, Mass, Load, and Coordinates
    frame_data : dict
        Frame element information including Type, ID, node connections,
        and Direction
    section_data : dict
        Section properties mapped by section IDs
    elastic_joint_flag : int, optional
        Joint behaviour flag (0: inelastic, 1: elastic), default 0

    Returns
    -------
    dict
        Joint outputs keyed by element tag containing joint model data
        including stiffness, capacity, and geometry
    """
    # Identify beam-column joints from point data
    bcj_mask = np.array(point_data['Type']) == 'BCJ'
    bcj_ids = point_data['ID'][bcj_mask]

    # Extract mass and load data for BCJ nodes
    mass_data = point_data['Mass'][bcj_mask]
    load_data = point_data['Load'][bcj_mask]

    # Extract BCJ coordinates
    bcj_coordinates = point_data['Coordinates'][bcj_mask]

    # Extract frame data arrays for processing
    frame_types = np.array(frame_data['Type'])
    frame_ids = frame_data['ID']
    point_ids = point_data['ID']

    outputs = {}

    # Process each beam-column joint
    for i, joint_id in enumerate(bcj_ids):
        # Find frames connected to this joint
        connected_frame_mask = ((frame_data['i_ID'] == joint_id) |
                                (frame_data['j_ID'] == joint_id))
        connected_directions = frame_data['Direction'][connected_frame_mask]

        # Determine joint location characteristics
        # Direction codes: 1=X-direction, 2=Y-direction, 3=Vertical
        joint_location_x = _determine_joint_location(connected_directions, 1)
        joint_location_y = _determine_joint_location(connected_directions, 2)

        # Extract beam sections connected to joint
        beam_x_sections = _get_beam_sections(
            connected_frame_mask, connected_directions, frame_types,
            section_data, direction=1
        )
        beam_y_sections = _get_beam_sections(
            connected_frame_mask, connected_directions, frame_types,
            section_data, direction=2
        )

        # Process column information
        column_info = _process_column_data(
            joint_id, connected_frame_mask, connected_directions,
            frame_data, frame_ids, point_data, point_ids, section_data
        )

        # Calculate nodal mass and loads
        node_mass = np.asarray([
            mass_data[i], mass_data[i], 1e-16, 1e-16, 1e-16, 1e-16
        ])
        node_load = load_data[i] * units.kN

        # Apply elastic joint modification if requested
        if elastic_joint_flag == 1:
            joint_location_x *= 10
            joint_location_y *= 10

        # Prepare joint data dictionary
        joint_data = {
            'ID': joint_id,
            'X': bcj_coordinates[i][0] * units.cm,
            'Y': bcj_coordinates[i][1] * units.cm,
            'Z': bcj_coordinates[i][2] * units.cm,
            'Hint': column_info['inter_storey_height'],
            'NLoad': node_load,
            'NMass': node_mass,
            'JointLocX': joint_location_x,
            'JointLocY': joint_location_y
        }

        # Create joint model and build capacity
        joint_type = 0  # Default joint type
        joint_model = jm.Joint(
            joint_type, joint_data, column_info['top_section'],
            beam_x_sections, beam_y_sections
        )

        current_output = joint_model.buildWcapacity()
        outputs[str(current_output['ElementTag'])] = current_output

    return outputs


def _determine_joint_location(connected_directions, direction):
    """
    Determine joint location type based on connected frame directions.

    Parameters
    ----------
    connected_directions : np.array
        Array of frame directions
    direction : int
        Direction to check (1 for X, 2 for Y)

    Returns
    -------
    int
        Location code (1: Exterior, 2: Interior, 3: No connection)
    """
    direction_count = sum(connected_directions == direction)
    if direction_count == 1:
        return 1  # Exterior
    elif direction_count > 1:
        return 2  # Interior
    else:
        return 3  # No connection


def _get_beam_sections(connected_frame_mask, connected_directions,
                       frame_types, section_data, direction):
    """
    Extract beam section data for a specific direction.

    Parameters
    ----------
    connected_frame_mask : np.array
        Boolean mask for connected frames
    connected_directions : np.array
        Array of frame directions
    frame_types : np.array
        Array of frame section type IDs
    section_data : dict
        Dictionary of section properties
    direction : int
        Direction to filter (1 for X, 2 for Y)

    Returns
    -------
    list
        Section data for beams in the specified direction
    """
    if sum(connected_directions == direction) >= 1:
        beam_section_ids = frame_types[connected_frame_mask][
            connected_directions == direction
        ]
        return [section_data[k] for k in beam_section_ids]
    else:
        return []


def _process_column_data(joint_id, connected_frame_mask, connected_directions,
                         frame_data, frame_ids, point_data, point_ids,
                         section_data):
    """
    Process column data connected to the joint.

    Parameters
    ----------
    joint_id : int
        ID of the current joint
    connected_frame_mask : np.array
        Boolean mask for connected frames
    connected_directions : np.array
        Array of frame directions
    frame_data : dict
        Frame element data
    frame_ids : np.array
        Array of frame element IDs
    point_data : dict
        Point data
    point_ids : np.array
        Array of point IDs
    section_data : dict
        Dictionary of section properties

    Returns
    -------
    dict
        Dictionary containing top_section and inter_storey_height
    """
    # Find columns (direction == 3) connected to this joint
    column_ids = np.array(frame_ids)[connected_frame_mask][
        connected_directions == 3
    ]
    column_section_ids = np.array(frame_data['Type'])[connected_frame_mask][
        connected_directions == 3
    ]
    column_i_ids = frame_data['i_ID'][connected_frame_mask][
        connected_directions == 3
    ]

    # Identify top and bottom columns
    top_column_id = column_ids[column_i_ids == max(column_i_ids)][0]
    bottom_column_id = column_ids[column_i_ids == min(column_i_ids)][0]
    top_column_section_id = column_section_ids[
        column_i_ids == max(column_i_ids)
    ][0]
    top_column_section = section_data[top_column_section_id]

    # Calculate inter-storey height
    top_col_i_id = max(column_i_ids)
    bottom_col_i_id = min(column_i_ids)
    top_col_j_id = frame_data['j_ID'][frame_ids == top_column_id]
    bottom_col_j_id = frame_data['j_ID'][frame_ids == bottom_column_id]

    # Extract Z-coordinates
    top_col_i_z = point_data['Coordinates'][point_ids == top_col_i_id][0][-1]
    bottom_col_i_z = point_data['Coordinates'][
        point_ids == bottom_col_i_id
    ][0][-1]
    top_col_j_z = point_data['Coordinates'][point_ids == top_col_j_id][0][-1]
    bottom_col_j_z = point_data['Coordinates'][
        point_ids == bottom_col_j_id
    ][0][-1]

    # Calculate average inter-storey height
    inter_storey_height_top = abs(top_col_j_z - top_col_i_z) * units.cm
    inter_storey_height_bottom = abs(
        bottom_col_j_z - bottom_col_i_z
    ) * units.cm
    inter_storey_height = 0.5 * (
        inter_storey_height_top + inter_storey_height_bottom
    )

    return {
        'top_section': top_column_section,
        'inter_storey_height': inter_storey_height
    }
