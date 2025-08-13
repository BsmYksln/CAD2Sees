# CAD2Sees

CAD2Sees is a Python-based tool for automated generation and analysis of reinforced concrete (RC) building models in [OpenSeesPy](https://github.com/zhuminjie/OpenSeesPy) from the CAD files. The tool parses building geometry and properties from two DXF files and one JSON file, automatically creating an OpenSeesPy model suitable for seismic assessment and retrofitting studies.

Currently, CAD2Sees supports lumped plasticity modelling, masonry infill representation (including multiple typologies), and EC-8 based performance checks. Post-processing includes bare-frame and infilled N2 method assessments, as well as interactive 3D visualisation of results using PyVista. Future releases will expand modelling capabilities and time-history analysis support.

![CAD2Sees Workflow](docs/CAD2Sees_Flow_Small.svg)

<div style="text-align:center;">

<details style="display: inline-block; text-align: center;">
<summary>CAD2Sees Workflow Details</summary>

![CAD2Sees Workflow](docs/CAD2Sees_Flow.svg) ➡️

</details>

</div>

## Features


- **Parsing** – automatic extraction of geometry, connectivity, material properties, and boundary conditions from DXF and JSON files
- **Model Generation** – automated creation of RC frame models with optional masonry infill elements and beam-column joints
- **Analysis** – gravity, modal, pushover analyses (loading shapes: mass-proportional, uniform, triangular, modal)
- **Post-processing** – EC-8 demand–capacity checks, N2 method for bare and infilled frames, shear and chord rotation checks
- **Visualisation** – interactive 3D visualisation using PyVista
- **Extensible** – modular structure for future inclusion of additional modelling types and analysis procedures

## Installation

CAD2Sees has been tested with Python 3.10 on Windows.
It is recommended to install in a virtual environment:

```bash
python -m venv cad2sees_env
cad2sees_env\Scripts\activate
pip install -r requirements.txt
```

There is currently no PyPI package; installation is done by cloning the repository:

```bash
git clone https://github.com/BsmYksln/CAD2Sees.git
cd CAD2Sees
```

## Basic Usage

1. **Prepare the minimum required input files:**
    - `3D.dxf` – 3D frame-like drawing of the structure
    - `Sections.dxf` – 2D cross-section details  
    - `GeneralInformation.json` – material properties, modelling assumptions, and assessment parameters

2. **Place the files in the directory referenced by your script** (see `example_1.py` for guidance)

3. **Run the example:**
    ```bash
    python examples/example_1.py
    ```

This will parse the input files, generate the OpenSeesPy model, run analyses, and produce results including interactive 3D visualisations.

## Examples

Example scripts are provided in the `examples/` directory:

- `example_1.py` – runs a complete workflow from input parsing to post-processing with comprehensive comments explaining each step

## Citing CAD2Sees

If you are using CAD2Sees in your research, please cite:

Yükselen, B., Mucedero, G., & Monteiro, R. (2025). Evaluation and Enhancement of Optimisation Algorithms for Automated Seismic Retrofitting of Existing Buildings. *Journal of Earthquake Engineering*, 1–34. https://doi.org/10.1080/13632469.2025.2541237

<details>
<summary>BibTeX Citation</summary>

```bibtex
@article{Yükselen06082025,
author = {Besim Yükselen and Gianrocco Mucedero and Ricardo Monteiro},
title = {Evaluation and Enhancement of Optimisation Algorithms for Automated Seismic Retrofitting of Existing Buildings},
journal = {Journal of Earthquake Engineering},
volume = {0},
number = {0},
pages = {1--34},
year = {2025},
publisher = {Taylor \& Francis},
doi = {10.1080/13632469.2025.2541237},
URL = {https://doi.org/10.1080/13632469.2025.2541237},
eprint = {https://doi.org/10.1080/13632469.2025.2541237}
}
```

</details>


## Planned Features

### Upcoming
- Addition of more extensive examples
- Addition and re-organisation of time-history analysis functionality
- Development and inclusion of tests

### Future Plans
- Addition of various modelling strategies for frame members
- Addition of various joint modelling approaches
- Addition of various infill modelling stategies

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to help improve CAD2Sees.

## License

This project is released under the GNU Affero General Public License v3.0 (AGPL-3.0).
See the LICENSE file for details.