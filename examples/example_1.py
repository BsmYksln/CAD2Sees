"""
CAD2Sees Example 1: Complete Seismic Assessment Workflow

This comprehensive example demonstrates the full CAD2Sees workflow for 
performing seismic assessment of a reinforced concrete frame structure:

1. Parse CAD drawings (DXF files) and configuration
2. Generate finite element model with beam-column joints and infills
3. Perform gravity, modal, and pushover analyses
4. Apply EC8 N2 method for seismic assessment
5. Visualize results with interactive 3D plots

Required Input Files:
- 3d_v2.dxf: 3D structural geometry (frames, nodes, elevations)
- sections.dxf: Cross-section definitions for structural members
- general_info.json: Material properties, loads, and analysis parameters
"""

import os

# Import miscellaneous modules
import numpy as np
import pandas as pd
import pickle as pkl

# =============================================================================
# CAD2Sees Module Imports
# =============================================================================

# Parsing Module - Extracts geometry and properties from CAD files
from cad2sees.parsing.dxfparse import dxfparse

# Model Generation Modules - Create OpenSees finite element model
from cad2sees.model_generation import model_init
from cad2sees.model_generation import node
from cad2sees.model_generation import restraint
from cad2sees.model_generation import geometric_transformation as gt
from cad2sees.model_generation import joint
from cad2sees.model_generation import frame
from cad2sees.model_generation import infill
from cad2sees.model_generation import constraint

# Helpers Module - Utility functions for capacity curves and OpenSees operations
from cad2sees.helpers import save_capacity
from cad2sees.helpers import ops_utils

# Analysis Modules - Perform structural analyses
from cad2sees.analysis import gravity
from cad2sees.analysis import modal
from cad2sees.analysis import pushover

# Post-Processing Modules - Process analysis results and assess performance
from cad2sees.post_processing import demand_capacity
from cad2sees.post_processing import N2
from cad2sees.post_processing import get_failed

# Visualization Module - Create interactive 3D plots and animations
from cad2sees.visualise import visualise as vis

# =============================================================================
# Analysis Configuration
# =============================================================================

# Lateral load pattern for pushover analysis
# Options: 'Modal' (single mode), 'Uniform', 'Linear', 'Triangular'
LoadPattern = 'Modal'

# Direction of lateral loading
# 1: X-direction, 2: Y-direction
Direction = int(1)

# =============================================================================
# File Paths and Directory Setup
# =============================================================================

# Get the directory where this script is located
cwd = os.path.dirname(os.path.abspath(__file__))

# Input files - These should be prepared using CAD software (AutoCAD, etc.)
dxf_file_3d = os.path.join(cwd, 'inputs', '3D.dxf')          # 3D geometry
dxf_file_sections = os.path.join(cwd, 'inputs', 'sections.dxf') # Cross-sections
json_file_general_info = os.path.join(cwd, 'inputs', 'general_info.json')  # Config

# Output directories for analysis results
output_dir = os.path.join(cwd, 'outputs')              # Main output directory
capacity_outdir = os.path.join(output_dir, 'capacity') # Element capacities
modal_outdir = os.path.join(output_dir, 'modal')       # Modal analysis results
push_outdir = os.path.join(output_dir, 'pushover')     # Pushover analysis

# Create output directories if they don't exist
for directory in [output_dir, capacity_outdir, modal_outdir, push_outdir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# =============================================================================
# STEP 1: Parse CAD Files and Initialize Model
# =============================================================================

print("Step 1: Parsing CAD files...")

# Parse the DXF files and JSON configuration file
# This extracts: geometry, cross-sections, materials, loads, boundary conditions
Parser = dxfparse(
    filepath_3D=dxf_file_3d,                    # 3D structural geometry
    filepath_Sections=dxf_file_sections,        # Member cross-sections
    filepath_GeneralInformation=json_file_general_info  # Analysis parameters
)
Parser.Parse()

# Initialize 3D model in OpenSees
# Returns: model dimensions, degrees of freedom per node
_, ndf = model_init.do3D()

# =============================================================================
# STEP 2: Create Geometric Transformations and Model Components
# =============================================================================

print("\nStep 2: Creating model components...")
# Create geometric transformations for frame elements
# This defines local coordinate systems for each element
Parser.Frames, GeometricTransformations = gt.do(Parser.Frames,
                                                Parser.Sections)

# Identify Beam-Column Joint (BCJ) nodes vs regular nodes
# BCJ nodes will be modeled with special joint elements
BCJMap = (np.asarray(Parser.Points['Type']) == 'BCJ')
regular_nodes = ~BCJMap

# Create regular nodes first (BCJ nodes created later with joint elements)
node.node(IDs=Parser.Points['ID'][regular_nodes],
          Coordinates=Parser.Points['Coordinates'][regular_nodes])

# Apply boundary conditions (fixed supports)
# Find nodes where all DOFs are constrained
FixedNodesMap = (Parser.Points['BoundryConditions'].sum(axis=1) == ndf)
restraint.fixall(Parser.Points['ID'][FixedNodesMap])

# =============================================================================
# STEP 3: Create Structural Elements
# =============================================================================

print("\nStep 3: Creating structural elements...")

# Create Beam-Column Joints (BCJ)
# These model joint panel zones and beam-column connections
JointOuts = joint.create(Parser.Points,
                         Parser.Frames,
                         Parser.Sections)

# Create frame elements with plastic hinges
# BWH_FiberOPS2: Beam With Hinges using fiber sections and OpenSees
# Other options: BWH_Fiber, BWH_Simple, BWH_FiberOPS
FrameOuts, SectionOuts = frame.create('BWH_FiberOPS2',
                                      Parser.Frames,
                                      Parser.Sections,
                                      Parser.Points,
                                      GeometricTransformations)
# Create infill wall elements
# These model masonry infill walls within frame bays
InfillOuts = infill.create(Parser.Infills,
                           Parser.InfillProperties)

# =============================================================================
# STEP 4: Save Element Capacities and Model Data
# =============================================================================

print("\nStep 4: Saving element capacities and model data...")

# Save capacity curves for all element types
# These define the force-deformation relationships for nonlinear analysis
save_capacity.BCJoint(JointOuts, capacity_outdir)
save_capacity.Frame(FrameOuts, capacity_outdir)
save_capacity.Infill(InfillOuts, capacity_outdir)

# Save model data for later use and visualization
model_data = {
    'sections.pkl': SectionOuts,    # Cross-section properties
    'frames.pkl': Parser.Frames,    # Frame element connectivity
    'nodes.pkl': Parser.Points      # Node coordinates and properties
}

for filename, data in model_data.items():
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pkl.dump(data, f)

# =============================================================================
# STEP 5: Define Constraints and Save Model Geometry
# =============================================================================

print("\nStep 5: Applying constraints...")

# Define rigid diaphragms (floors act as rigid bodies in their plane)
# This couples horizontal displacements at each floor level
rigid_diaphragm_flag = Parser.config['rigid_diaphragm']
push_nodes = constraint.rigid_diaphragm(Parser.Points,
                                        rigid_diaphragm_flag)
# Save the undeformed model geometry for visualization
ops_utils.GetShape(output_dir)

# =============================================================================
# STEP 6: Structural Analysis Sequence
# =============================================================================

print("\nStep 6: Running structural analyses...")
# 6.1 Gravity Analysis
# Apply dead and live loads to establish initial stress state
print("  6.1 Performing gravity analysis...")
gravity.do(Parser.Points)
# 6.2 Modal Analysis  
# Extract natural periods and mode shapes for dynamic characterization
print("  6.2 Performing modal analysis...")
periods, mass_participation_ratios = modal.do(3, outsavemode=1,
                                              outdir=modal_outdir,
                                              outMPR=True)
# 6.3 Setup Recorders for Pushover Analysis
# These capture response quantities during nonlinear analysis
print("  6.3 Setting up analysis recorders...")
beam_column_elements = [int(i) for i in list(FrameOuts.keys())]
bcj_elements = [int(i) for i in list(JointOuts.keys())]
infill_elements = [int(i) for i in list(InfillOuts.keys())]

# Add various recorders to capture different response quantities
ops_utils.AddDisplacementRecorders(push_outdir)
ops_utils.AddChordRotationRecorders(push_outdir, beam_column_elements)
ops_utils.AddFrameForceRecorders(push_outdir, beam_column_elements)
ops_utils.AddFrameInflectionPointRecorders(push_outdir, beam_column_elements)
ops_utils.AddBCJRecorder(push_outdir, bcj_elements)
ops_utils.AddInfillRecorder(push_outdir, infill_elements)

# 6.4 Pushover Analysis
# Nonlinear static analysis with increasing lateral loads
print("  6.4 Performing pushover analysis...")
spo_curve, spo_info = pushover.do(LoadPattern, Direction,
                                  push_nodes, nsStep=1000,
                                  dref=0.1, mu=3.0)
# Save pushover results
np.savetxt(os.path.join(push_outdir, 'SPOCurve.out'), spo_curve)
np.savetxt(os.path.join(push_outdir, 'PushInfo.out'), spo_info)

# =============================================================================
# STEP 7: Post-Processing and Demand-Capacity Assessment
# =============================================================================

print("\nStep 7: Post-processing analysis results...")

# Calculate demand-to-capacity ratios (DCRs) for all element types
demand_capacity.BCJ(push_outdir, capacity_outdir)
demand_capacity.FrameRotation(push_outdir, capacity_outdir)
demand_capacity.Infill(push_outdir, capacity_outdir)
demand_capacity.FrameShear(push_outdir, capacity_outdir,
                           addInfill=1,      # Include infill contribution
                           priestleyFlag=0)  # Use standard shear model
# =============================================================================
# STEP 8: EC8 N2 Method Seismic Assessment
# =============================================================================

print("\nStep 8: Applying EC8 N2 method...")

# Extract seismic parameters from configuration
pga = Parser.config['PGA']                    # Peak ground acceleration
soil_class = Parser.config['SoilClass']       # Soil classification
nation = Parser.config['NationalAnnex']       # National annex
spectrum_type = Parser.config['SpectrumType'] # Response spectrum type

# Apply N2 method to find performance point
n2_proc = N2.N2(pga, soil_class, spo_curve, spo_info, push_outdir,
                capacity_outdir, Nation=nation, SpectrumType=spectrum_type)
n2_proc.do()
n2_proc.plot_it()
# =============================================================================
# STEP 9: Failure Assessment and Results Summary
# =============================================================================

print("\nStep 9: Assessing structural performance...")
# Extract failure information at the performance point
frame_rotation_fail, frame_yeild = get_failed.getFrameFlexural(
    push_outdir, int(n2_proc.StepNum))
frame_shear_fail = get_failed.getFrameShear(push_outdir,
                                            int(n2_proc.StepNum))

# Organize DCR data into a comprehensive dataframe
DFCols = list(frame_rotation_fail.keys())
data_dict = {
    'YeildI': [],      # Yield DCR at element i-end
    'YeildJ': [],      # Yield DCR at element j-end  
    'RotationI': [],   # Rotation DCR at element i-end
    'RotationJ': [],   # Rotation DCR at element j-end
    'Rotation': [],    # Maximum rotation DCR
    'Shear': []        # Shear DCR
}

# Process DCR data for each frame element
for k in DFCols:
    # Rotation DCRs
    rot_i = frame_rotation_fail[k].get('I', {})
    rot_j = frame_rotation_fail[k].get('J', {})
    rot_vals = [v for d in [rot_i, rot_j] for v in d.values()]
    max_rot = max(rot_vals) if rot_vals else float('nan')
    data_dict['Rotation'].append(max_rot)
    data_dict['RotationI'].append(max(rot_i.values()) if rot_i
                                  else float('nan'))
    data_dict['RotationJ'].append(max(rot_j.values()) if rot_j
                                  else float('nan'))

    # Yield DCRs
    yield_i = frame_yeild[k].get('I', {})
    yield_j = frame_yeild[k].get('J', {})
    data_dict['YeildI'].append(max(yield_i.values()) if yield_i
                               else float('nan'))
    data_dict['YeildJ'].append(max(yield_j.values()) if yield_j
                               else float('nan'))

    # Shear DCRs
    shear_i = frame_shear_fail[k].get('I', {})
    shear_j = frame_shear_fail[k].get('J', {})
    shear_vals = [v for d in [shear_i, shear_j] for v in d.values()]
    max_shear = max(shear_vals) if shear_vals else float('nan')
    data_dict['Shear'].append(max_shear)

# Create summary dataframe
df = pd.DataFrame(data_dict, index=DFCols).T

# Identify elements that have reached failure (DCR >= 1.0)
shear_collapses = df.columns[df.loc['Shear'] >= 1.0]
rotation_collapses = df.columns[df.loc['Rotation'] >= 1.0]

print(f"  - {len(shear_collapses)} elements failed in shear")
print(f"  - {len(rotation_collapses)} elements failed in rotation")

# =============================================================================
# STEP 10: Interactive 3D Visualization
# =============================================================================

print("\nStep 10: Generating interactive visualizations...")

# Create visualization object
V = vis.visualise(output_dir)

# Generate modal visualization (uncomment to create modal animations)
V.modal(1, scalefactor=10, showme=1, saveme='mp4')  # 1st mode
# V.modal(2, scalefactor=10, showme=1, saveme='html')  # 2nd mode  
# V.modal(3, scalefactor=10, showme=1, saveme='html')  # 3rd mode

# V = vis.visualise(output_dir)
# Generate pushover visualization at performance point
# Shows: 3D deformed shape, DCR color-coding, pushover curve
V.Pushover(push_outdir,
           showDCRframeShear=True,        # Show shear DCR labels
           step=int(n2_proc.StepNum),     # At performance point
           showme=1,                      # Display interactive plot
           saveme=1)                      # Save HTML file
print("\n" + "="*60)
print(f"All results saved to: {output_dir}")
print("="*60)
