"""
Structural analysis visualisation module for CAD2Sees.

Provides 3D visualisation for CAD2Sees analysis results including undeformed
geometry, modal analysis, pushover analysis, and EC8 N2 method results.
"""
import json
import os
import copy
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
import pandas as pd


class visualise:
    """
    3D visualisation class for structural analysis results.

    Provides visualisation capabilities for CAD2Sees analysis results
    including undeformed geometry, modal shapes, pushover analysis
    animations, and EC8 N2 method assessment results.

    Parameters
    ----------
    OutDir : str
        Output directory path containing analysis results.
    """
    def __init__(self,
                 OutDir):
        """Initialise the visualise class.

        Args:
            OutDir (str): Output directory path.
        """
        self.OutDir = OutDir
        self.p = pv.Plotter()

    def _find_third_point(self, p1, p2, distance_ratio):
        """Find a third point on a line defined by two points.

        Args:
            p1 (list): First point coordinates [x, y, z].
            p2 (list): Second point coordinates [x, y, z].
            distance_ratio (float): Ratio of distance from p1 to new point
                                   relative to distance from p1 to p2.

        Returns:
            tuple: Third point coordinates (x3, y3, z3).
        """
        x3 = p1[0] + distance_ratio * (p2[0] - p1[0])
        y3 = p1[1] + distance_ratio * (p2[1] - p1[1])
        z3 = p1[2] + distance_ratio * (p2[2] - p1[2])
        return x3, y3, z3

    def _UpdateNodes(self, NodeIDs, Nodes, CurrentModeOuts, scale=1):
        """Update node coordinates based on modal analysis results.

        Args:
            NodeIDs (list): Node IDs.
            Nodes (list): Node coordinates.
            CurrentModeOuts (dict): Modal output for current mode.
            scale (int, optional): Scale factor for displacements.
                                   Defaults to 1.

        Returns:
            tuple: Updated node coordinates and scalar values.
        """
        NewNodes = copy.deepcopy(Nodes)
        SCVals = []
        for ni, nn in enumerate(NodeIDs):
            NodeChange3D = np.array(CurrentModeOuts[str(nn)][:3]) * scale
            SCVals.append(np.linalg.norm(NodeChange3D))
            NewNodes[ni] = np.array(NewNodes[ni]) + NodeChange3D
        return NewNodes, SCVals

    def _findColor(self, Step, ID, Tags, Datas, CtrlType):
        """
        Find appropriate colour and DCR values based on analysis type.

        Parameters
        ----------
        Step : int
            Analysis step number.
        ID : int or str
            Element or node ID.
        Tags : array-like
            Array of element/node tags.
        Datas : list
            List of data arrays for analysis.
        CtrlType : str
            Type of analysis control.

        Returns
        -------
        tuple
            Colour ID and DCR values.
        """
        if CtrlType == 'FrameShear':
            DCRFrameShear = Datas[0]
            if DCRFrameShear.ndim == 1:
                MaxDCRFrameShear = copy.deepcopy(DCRFrameShear)
            else:
                MaxDCRFrameShear = abs(DCRFrameShear[:Step, :]).max(axis=0)
            if isinstance(Tags[0], str):
                CurDCR = MaxDCRFrameShear[np.where(Tags == str(ID))][0]
            elif isinstance(Tags[0], int):
                CurDCR = MaxDCRFrameShear[np.where(Tags == int(ID))][0]
            elif isinstance(Tags[0], float):
                CurDCR = MaxDCRFrameShear[np.where(Tags == float(ID))][0]

            if CurDCR >= 1:
                ColID = '#D32F2F'
            else:
                ColID = 'black'
            DCR = [CurDCR]
        elif CtrlType == 'BCJDeformation':
            DS1 = Datas[0]
            DS2 = Datas[1]
            DS3 = Datas[2]
            MaxDCRDS1 = abs(DS1[:Step, :]).max(axis=0)
            MaxDCRDS2 = abs(DS2[:Step, :]).max(axis=0)
            MaxDCRDS3 = abs(DS3[:Step, :]).max(axis=0)
            CurDCRDS1 = MaxDCRDS1[np.where(Tags == ID)][0]
            CurDCRDS2 = MaxDCRDS2[np.where(Tags == ID)][0]
            CurDCRDS3 = MaxDCRDS3[np.where(Tags == ID)][0]
            DCR = [CurDCRDS1,
                   CurDCRDS2,
                   CurDCRDS3]
            if CurDCRDS3 >= 1:
                # ColID = '#FF0000'
                ColID = 1
            elif CurDCRDS2 >= 1:
                # ColID = '#FF8800'
                ColID = 0.75
            elif CurDCRDS1 >= 1:
                # ColID = '#EDFA00'
                ColID = 0.5
            else:
                # ColID = '00FF15'
                ColID = 0
        elif CtrlType == 'BCJForce':
            DCR = Datas[0]
            MaxDCR = abs(DCR[:Step, :]).max(axis=0)
            CurDCR = MaxDCR[np.where(Tags == ID)][0]
            if CurDCR >= 1:
                ColID = 1
            else:
                ColID = 0
            DCR = [CurDCR]
        elif CtrlType == 'FrameDef':
            IDCR = Datas[0]
            JDCR = Datas[1]
            IDCRY = Datas[2]
            JDCRY = Datas[3]
            if IDCR.ndim == 1:
                MaxDCRI = copy.deepcopy(IDCR)
                MaxDCRJ = copy.deepcopy(JDCR)
                MaxDCRYI = copy.deepcopy(IDCRY)
                MaxDCRYJ = copy.deepcopy(JDCRY)
            else:
                MaxDCRI = abs(IDCR[:Step, :]).max(axis=0)
                MaxDCRJ = abs(JDCR[:Step, :]).max(axis=0)
                MaxDCRYI = abs(IDCRY[:Step, :]).max(axis=0)
                MaxDCRYJ = abs(JDCRY[:Step, :]).max(axis=0)
            INodeIdx = np.where(Tags == ID)[0][0]
            JNodeIdx = np.where(Tags == ID)[0][0]
            IDCR = MaxDCRI[INodeIdx]
            JDCR = MaxDCRJ[JNodeIdx]
            IDCRY = MaxDCRYI[INodeIdx]
            JDCRY = MaxDCRYJ[JNodeIdx]
            # SD and NC
            if IDCR >= 1:
                IColID = 1
            elif IDCR >= 0.75:
                IColID = 0.75
            elif IDCRY >= 1:
                IColID = 0.5
            else:
                IColID = 0

            if JDCR >= 1:
                JColID = 1
            elif JDCR >= 0.75:
                JColID = 0.75
            elif JDCRY >= 1:
                JColID = 0.5
            else:
                JColID = 0

            # # SD and NC
            # if IDCR >= 1:
            #     IColID = '#C02B2B'
            # elif IDCR >= 0.75:
            #     IColID = '#FFA000'
            # elif IDCRY >= 1:
            #     IColID = '#FFEB3B'
            # else:
            #     IColID = '#808080'

            # if JDCR >= 1:
            #     JColID = '#C02B2B'
            # elif JDCR >= 0.75:
            #     JColID = '#FFA000'
            # elif JDCRY >= 1:
            #     JColID = '#FFEB3B'
            # else:
            #     JColID = '#808080'
            ColID = [IColID, JColID]
            DCR = [[IDCR, JDCR], [IDCRY, JDCRY]]
        else:
            ColID = '#808080'
            # ColID = '00FF15'
        return ColID, DCR

    def unDeformated(self, showme=1, frametags=0, saveme=0,
                     frame_props={'color': 'k', 'lw': 1, 'alpha': 0.5},
                     infill_props={'color': 'k', 'lw': 1, 'alpha': 0.5},
                     camera_position=None,
                     camera_zoom=None,
                     called_from=False):
        """Visualise the undeformed shape of the structure.

        Args:
            showme (int, optional): Show plot interactively. Defaults to 1.
            frametags (int, optional): Show frame tags. Defaults to 0.
            saveme (str, optional): Save format ("html", "png", "jpeg",
                                   "tiff"). Defaults to None.
            frame_props (dict, optional): Frame element properties (color,
                                         lw, alpha). Defaults to
                                         {'color': 'k', 'lw': 1, 'alpha': 0.5}.
            infill_props (dict, optional): Infill element properties (color,
                                          lw, alpha). Defaults to
                                          {'color': 'k', 'lw': 1,
                                          'alpha': 0.5}.
            camera_position (list, optional): Custom camera position.
                                              Defaults to None.
            camera_zoom (float, optional): Custom camera zoom factor.
                                          Defaults to None.
        """
        # Create a new plotter instance, use off_screen if only saving
        use_off_screen = bool(saveme and not showme)
        if called_from is False:
            self.p = pv.Plotter(off_screen=use_off_screen)
        unDefShapeDataFile = os.path.join(self.OutDir, 'ShapeData.json')
        with open(unDefShapeDataFile, 'r') as j:
            unDefData = json.loads(j.read())

        # Reorganise Undeformated shape data
        self.Elements = unDefData['Elements']
        self.Nodes = unDefData['Nodes']
        NodeCoordinates = self.Nodes['Coordinates']
        self.NodeIDs = self.Nodes['Tags']
        FrameEdges = []
        InfillEdges = []

        FrameLabels = []
        MidPoints = []
        for k in self.Elements.keys():
            FrameLabels.append(k)
            EdgeI = self.Elements[k]['i']['Tag']
            mapEdgeI = np.where(np.array(self.NodeIDs) == EdgeI)[0][0]
            EdgeJ = self.Elements[k]['j']['Tag']
            mapEdgeJ = np.where(np.array(self.NodeIDs) == EdgeJ)[0][0]
            CoordI = np.array(self.Nodes['Coordinates'][mapEdgeI])
            CoordJ = np.array(self.Nodes['Coordinates'][mapEdgeJ])
            CurMidF = (CoordI + CoordJ) * 0.5
            MidPoints.append(CurMidF)
            if (CoordI == CoordJ).all():
                # Skip this that is zero length
                continue
            elif (CoordI[2] == CoordJ[2] and
                  (CoordI[0] == CoordJ[0] or CoordI[1] == CoordJ[1])):
                # This is a Beam
                FrameEdges.append([mapEdgeI, mapEdgeJ])
            elif (CoordI[2] != CoordJ[2] and CoordI[0] == CoordJ[0] and
                  CoordI[1] == CoordJ[1]):
                # This is a Column
                FrameEdges.append([mapEdgeI, mapEdgeJ])
            else:
                # This is an Infill
                InfillEdges.append([mapEdgeI, mapEdgeJ])
        MidF = np.array(MidPoints)
        Nodes = np.array(NodeCoordinates)
        FrameEdges = np.array(FrameEdges)
        InfillEdges = np.array(InfillEdges)

        # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(FrameEdges.shape[0], int) * 2
        padding[:] = 2

        self.PaddedFrameEdges = np.vstack((padding, FrameEdges.T)).T

        meshFrame = pv.PolyData()
        meshFrame.points = Nodes
        meshFrame.lines = self.PaddedFrameEdges

        self.p.add_mesh(
            meshFrame,
            render_lines_as_tubes=False,
            style='wireframe',
            color=frame_props['color'],
            line_width=frame_props['lw'],
            opacity=frame_props['alpha'])

        padding = np.empty(InfillEdges.shape[0], int) * 2
        padding[:] = 2
        self.PaddedInfillEdges = np.vstack((padding, InfillEdges.T)).T

        meshInfill = pv.PolyData()
        meshInfill.points = Nodes
        meshInfill.lines = self.PaddedInfillEdges

        self.p.add_mesh(
            meshInfill,
            render_lines_as_tubes=False,
            style='wireframe',
            color=infill_props['color'],
            line_width=infill_props['lw'],
            opacity=infill_props['alpha'])

        if frametags == 1:
            self.p.add_point_labels(
                MidF,
                FrameLabels,
                font_size=15,
                point_size=0.1,
                text_color='grey',
                fill_shape=False,
                bold=False)

        if camera_position:
            self.p.camera_position = camera_position
        if camera_zoom:
            self.p.camera.zoom(camera_zoom)

        # Handle saving first if we're in off_screen mode
        if saveme and use_off_screen:
            filename_base = "unDeformated_plot"
            save_path_base = os.path.join(self.OutDir, filename_base)
            
            if saveme == 'html':
                self.p.show(auto_close=False)  # Need to show for html export
                self.p.export_html(f"{save_path_base}.html")
                print(f"Plot saved to {save_path_base}.html")
            elif saveme in ['png', 'jpeg', 'tiff']:
                # For image formats, just render once off-screen
                self.p.show(auto_close=False)
                self.p.screenshot(f"{save_path_base}.{saveme}")
                print(f"Plot saved to {save_path_base}.{saveme}")
            else:
                print(
                    f"Unsupported save format: {saveme}. "
                    f"Supported formats: html, png, jpeg, tiff.")

        # Show interactively if requested
        if showme == 1:
            # Only show if we haven't already in off_screen mode
            if not use_off_screen:
                self.p.show(auto_close=False)

            # Save after showing if requested and we're not in off_screen mode
            if saveme and not use_off_screen:
                filename_base = "unDeformated_plot"
                save_path_base = os.path.join(self.OutDir, filename_base)
                if saveme == 'html':
                    self.p.export_html(f"{save_path_base}.html")
                    print(f"Plot saved to {save_path_base}.html")
                elif saveme in ['png', 'jpeg', 'tiff']:
                    self.p.screenshot(f"{save_path_base}.{saveme}")
                    print(f"Plot saved to {save_path_base}.{saveme}")
                else:
                    print(
                        f"Unsupported save format: {saveme}. "
                        f"Supported formats: html, png, jpeg, tiff.")

        # Always clean up
        if showme == 1 or saveme:
            self.p.clear()
        
        # Close plotter if it was only used for saving
        if use_off_screen and not showme:
            self.p.close()

    def unDeformatedWithDCR(self, showme=1, frametags=0, DCRFile=None,
                            saveme=None):
        """Visualise the undeformed shape with DCR values.

        Args:
            showme (int, optional): Show the plot. Defaults to 1.
            frametags (int, optional): Show frame tags. Defaults to 0.
            DCRFile (str, optional): Path to DCR file. Defaults to None.
            saveme (str, optional): Save format ("html", "png", "jpeg",
                                   "tiff"). Defaults to None.
        """
        unDefShapeDataFile = os.path.join(self.OutDir, 'ShapeData.json')
        with open(unDefShapeDataFile, 'r') as j:
            unDefData = json.loads(j.read())

        # Reorganise Undeformated shape data
        self.Elements = unDefData['Elements']
        self.Nodes = unDefData['Nodes']
        NodeCoordinates = self.Nodes['Coordinates']
        self.NodeIDs = self.Nodes['Tags']
        Edges = []
        MidF = []
        for k in self.Elements.keys():
            EdgeI = self.Elements[k]['i']['Tag']
            mapEdgeI = np.where(np.array(self.NodeIDs) == EdgeI)[0][0]
            EdgeJ = self.Elements[k]['j']['Tag']
            mapEdgeJ = np.where(np.array(self.NodeIDs) == EdgeJ)[0][0]
            Edges.append([mapEdgeI,
                          mapEdgeJ])
            CurMidF = (np.array(self.Nodes['Coordinates'][mapEdgeI]) +
                       np.array(self.Nodes['Coordinates'][mapEdgeJ])) * 0.5
            MidF.append(CurMidF)
        Nodes = np.array(NodeCoordinates)
        Edges = np.array(Edges)

        # We must "pad" the edges to indicate to vtk how many points per edge
        padding = np.empty(Edges.shape[0], int) * 2
        padding[:] = 2

        self.PaddedEdges = np.vstack((padding, Edges.T)).T

        mesh1 = pv.PolyData()
        mesh1.points = Nodes
        mesh1.lines = self.PaddedEdges

        self.p.add_mesh(mesh1, render_lines_as_tubes=False,
                        style='wireframe',
                        line_width=3,
                        opacity=1)
        if frametags == 1:
            labels = list(self.Elements.keys())
            self.p.add_point_labels(MidF, labels, font_size=15,
                                    point_size=0.1, text_color='red')
        if DCRFile is not None:
            DCRData = pd.read_csv(DCRFile)
            labels = []
            for k_elem in self.Elements.keys():  # Renamed k to k_elem
                if k_elem in DCRData.columns:
                    if DCRData[k_elem][0] >= DCRData[k_elem][1]:
                        curlabel = f'Rotation | {DCRData[k_elem][0]}'
                    else:
                        curlabel = f'Shear | {DCRData[k_elem][1]}'
                    labels.append(curlabel)
                else:
                    labels.append('')
            self.p.add_point_labels(MidF, labels, font_size=15,
                                    point_size=0.1, text_color='red')

        if showme == 1:
            self.p.show(auto_close=False)

        if saveme:
            filename_base = 'unDeformatedWithDCR'
            if DCRFile:
                base_name_dcr = os.path.basename(DCRFile)
                filename_base += f'_{os.path.splitext(base_name_dcr)[0]}'
            filename = f'{filename_base}_plot'
            save_path_base = os.path.join(self.OutDir, filename)

            if saveme == 'html':
                self.p.export_html(f"{save_path_base}.html")
                print(f"Plot saved to {save_path_base}.html")
            elif saveme in ['png', 'jpeg', 'tiff']:
                self.p.screenshot(f"{save_path_base}.{saveme}")
                print(f"Plot saved to {save_path_base}.{saveme}")
            else:
                print(f"Unsupported save format: {saveme}. \\r"
                      f"Supported formats: html, png, jpeg, tiff.")

        if showme == 1 or saveme:
            self.p.clear()

    def modal(self, ModeNum=1, scalefactor=10,
              VideoArgs={'LoopNum': 5,
                         'OutType': 'mp4',
                         'resolution': (480, 672)},
              showme=1, saveme=None):
        """Visualise modal analysis results.

        Args:
            ModeNum (int, optional): Mode number to visualise. Defaults to 1.
            scalefactor (int, optional): Scale factor for mode shapes.
                                         Defaults to 10.
            VideoArgs (dict, optional): Video output arguments for "mp4" or
                                       "gif" formats. Defaults to
                                       {'LoopNum': 5, 'OutType': 'mp4',
                                       'resolution': (480, 672)}.
            showme (int, optional): Show plot if not saving animation.
                                   Defaults to 1.
            saveme (str, optional): Save format ("html", "png", "jpeg",
                                   "tiff", "mp4", "gif"). Defaults to None.
        """
        self.unDeformated(showme=0)  # Add undeformed shape as a base
        ModalFile = os.path.join(self.OutDir, 'Modal', 'ModalProps.json')
        with open(ModalFile, 'r') as j:
            ModeOuts = json.loads(j.read())

        CurrentModeOuts = ModeOuts[str(ModeNum)]

        modal_output_dir = os.path.join(self.OutDir, 'Modal')
        if not os.path.exists(modal_output_dir):
            os.makedirs(modal_output_dir)

        is_animation_format = saveme in ['mp4', 'gif']

        if is_animation_format:
            mesh_anim_frames = pv.PolyData()
            mesh_anim_frames.points = self.Nodes['Coordinates']
            mesh_anim_frames.lines = self.PaddedFrameEdges

            mesh_anim_infills = pv.PolyData()
            mesh_anim_infills.points = self.Nodes['Coordinates']
            mesh_anim_infills.lines = self.PaddedInfillEdges

            initial_sc_vals = [0.0] * len(self.Nodes['Coordinates'])
            mesh_anim_frames.point_data["Scalars"] = initial_sc_vals
            mesh_anim_infills.point_data["Scalars"] = initial_sc_vals

            resolution = VideoArgs.get('resolution', (480, 672))
            self.p.window_size = resolution

            anim_filename = f"modal_Mode{ModeNum}.{saveme}"
            anim_save_path = os.path.join(modal_output_dir, anim_filename)

            if saveme == 'gif':
                self.p.open_gif(anim_save_path)
            elif saveme == 'mp4':
                self.p.open_movie(anim_save_path)

            self.p.add_mesh(mesh_anim_frames, render_lines_as_tubes=False,
                            style='wireframe', line_width=1, opacity=1,
                            scalars="Scalars", show_scalar_bar=False,
                            cmap="viridis")
            self.p.add_mesh(mesh_anim_infills, render_lines_as_tubes=False,
                            style='wireframe', line_width=1, opacity=1,
                            scalars="Scalars", show_scalar_bar=False,
                            cmap="viridis")

            NumFrm = VideoArgs.get('NumFrm', 60)
            LoopNum = VideoArgs.get('LoopNum', 5)
            FrameScales = list(np.linspace(-1, 1, NumFrm))
            FrameScales = FrameScales + FrameScales[::-1]
            FrameScales = FrameScales * LoopNum

            print(f'Writing {saveme} frames for modal animation...')
            for i in range(len(FrameScales)):
                SCCur = FrameScales[i] * scalefactor
                NodesUpdated_anim, SCVals_anim = self._UpdateNodes(
                    self.NodeIDs, self.Nodes['Coordinates'], CurrentModeOuts,
                    scale=SCCur)
                mesh_anim_frames.points = NodesUpdated_anim
                mesh_anim_frames.point_data["Scalars"] = SCVals_anim

                mesh_anim_infills.points = NodesUpdated_anim
                mesh_anim_infills.point_data["Scalars"] = SCVals_anim

                self.p.write_frame()
            self.p.close()
            print(f"Animation saved to {anim_save_path}")
        else:  # Static plot (either for saving or just showing)
            NodesUpdated_static, SCVals_static = self._UpdateNodes(
                self.NodeIDs, self.Nodes['Coordinates'], CurrentModeOuts,
                scale=scalefactor)

            mesh_static_frame = pv.PolyData()
            mesh_static_frame.points = NodesUpdated_static
            mesh_static_frame.lines = self.PaddedFrameEdges

            self.p.add_mesh(mesh_static_frame, render_lines_as_tubes=False,
                            style='wireframe', line_width=1, opacity=1,
                            scalars=SCVals_static, cmap="viridis")

            mesh_static_infill = pv.PolyData()
            mesh_static_infill.points = NodesUpdated_static
            mesh_static_infill.lines = self.PaddedInfillEdges

            self.p.add_mesh(mesh_static_infill, render_lines_as_tubes=False,
                            style='wireframe', line_width=1, opacity=1,
                            scalars=SCVals_static, cmap="viridis")
            self.p.remove_scalar_bar()

            if showme == 1:
                self.p.show(auto_close=False)

            if saveme:  # Static save
                filename_base = f'modal_Mode{ModeNum}_plot'
                save_path_base = os.path.join(modal_output_dir,
                                              filename_base)
                if saveme == 'html':
                    self.p.export_html(f"{save_path_base}.html")
                    print(f"Plot saved to {save_path_base}.html")
                elif saveme in ['png', 'jpeg', 'tiff']:
                    self.p.screenshot(f"{save_path_base}.{saveme}")
                    print(f"Plot saved to {save_path_base}.{saveme}")
                else:
                    print(f"Unsupported save format for static modal plot: \\r"
                          f"{saveme}. Supported: html, png, jpeg, tiff.")

    def Pushover(self, PushoverOuts, step, CheckType='W_Infill',
                 scalefactor=10, showDCRframeShear=0, showme=1, saveme=None):
        """Visualise pushover analysis results.

        Args:
            step (int): Step number to visualise for static plot/show.
            CheckType (str, optional): Type of check ("W_Infill" or
                                      "Wo_Infill"). Defaults to "W_Infill".
            scalefactor (int, optional): Scale factor for displacements.
                                         Defaults to 10.
            showDCRframeShear (int, optional): Show DCR for frame shear.
                                              Defaults to 0.
            showme (int, optional): Show plot if not saving animation.
                                   Defaults to 1.
            saveme (str, optional): Save format ("html", "png", "jpeg",
                                   "tiff", "mp4", "gif"). Defaults to None.
        """
        # Create a fresh plotter for this visualization
        self.unDeformated(showme=0)

        # Read Curve
        PushCurveFile = os.path.join(PushoverOuts, 'SPOCurve.out')
        PushCurve = np.genfromtxt(PushCurveFile)
        TopDisp = PushCurve[0, :]
        BaseShear = abs(PushCurve[1, :])

        # Read Displacements
        XTransDefsFile = os.path.join(PushoverOuts, 'TransversalX.out')
        XTransData = np.genfromtxt(XTransDefsFile)
        XRotDefsFile = os.path.join(PushoverOuts, 'RotationalX.out')
        XRotData = np.genfromtxt(XRotDefsFile)
        YTransDefsFile = os.path.join(PushoverOuts, 'TransversalY.out')
        YTransData = np.genfromtxt(YTransDefsFile)
        YRotDefsFile = os.path.join(PushoverOuts, 'RotationalY.out')
        YRotData = np.genfromtxt(YRotDefsFile)
        ZTransDefsFile = os.path.join(PushoverOuts, 'TransversalZ.out')
        ZTransData = np.genfromtxt(ZTransDefsFile)
        ZRotDefsFile = os.path.join(PushoverOuts, 'RotationalZ.out')
        ZRotData = np.genfromtxt(ZRotDefsFile)
        PushNodesFile = os.path.join(PushoverOuts, 'OutNodes.out')
        DispNodes = np.genfromtxt(PushNodesFile)
        FramesFile = os.path.join(PushoverOuts, 'OutFrames.out')
        FrameData = np.genfromtxt(FramesFile)

        # Read Frame Shear DCR
        if CheckType == 'Wo_Infill':
            DCRFrameShearFile = os.path.join(PushoverOuts,
                                             'FrameShearDCR',
                                             'DCRShear.csv')
        elif CheckType == 'W_Infill':
            DCRFrameShearFile = os.path.join(PushoverOuts,
                                             'FrameShearDCR',
                                             'DCRShearWithInfill.csv')
        FrameDCRData = np.genfromtxt(DCRFrameShearFile)

        # Color Proxy for Frame Shear
        FrameColorProx = [FrameDCRData]

        # Reorganise Pushover Deformantion Data
        DeformationData = np.dstack([XTransData,
                                     YTransData,
                                     ZTransData,
                                     XRotData,
                                     YRotData,
                                     ZRotData])

        PushData = {}
        for Step in range(DeformationData.shape[0]):
            PushData[str(Step)] = {}
            for nodeIdx, node in enumerate(DispNodes):
                PushData[str(Step)][str(int(node))] = list(
                    DeformationData[Step, nodeIdx, :])

        # Frame Colors Based on Shear DCR
        EdgesChecked = []
        EdgesOther = []
        ColorsChecked = []
        ColorsOther = []
        DCRShear = []
        MidF = []
        ElementLabel = []
        for k in self.Elements.keys():
            EdgeI = self.Elements[k]['i']['Tag']
            mapEdgeI = np.where(np.array(self.NodeIDs) == EdgeI)[0][0]
            EdgeJ = self.Elements[k]['j']['Tag']
            mapEdgeJ = np.where(np.array(self.NodeIDs) == EdgeJ)[0][0]
            if int(k) in FrameData[0, :]:
                CurCol, DCR = self._findColor(step, int(k),
                                              FrameData[0, :],
                                              FrameColorProx,
                                              'FrameShear')
                ColorsChecked.append(CurCol)
                EdgesChecked.append([mapEdgeI,
                                     mapEdgeJ])
                DCRShear.append(DCR[0])
                ElementLabel.append(k)
                CurMidF = (np.array(self.Nodes['Coordinates'][mapEdgeI]) +
                           np.array(self.Nodes['Coordinates'][mapEdgeJ]))*0.5
                MidF.append(CurMidF)
            else:
                ColorsOther.append(0)
                EdgesOther.append([mapEdgeI,
                                   mapEdgeJ])
        EdgesChecked = np.array(EdgesChecked)
        EdgesOther = np.array(EdgesOther)
        # We must "pad" the edges to indicate to vtk how many points per edge
        paddingChecked = np.empty(EdgesChecked.shape[0], int) * 2
        paddingOther = np.empty(EdgesOther.shape[0], int) * 2
        paddingChecked[:] = 2
        paddingOther[:] = 2

        PaddedEdgesChecked = np.vstack((paddingChecked, EdgesChecked.T)).T
        PaddedEdgesOther = np.vstack((paddingOther, EdgesOther.T)).T

        mapping = np.linspace(0, 1, 256)
        newcolors = np.empty((256, 4))

        # purple = np.array([95/256, 1/256, 177/256, 1])
        green = np.array([1/256, 255/256, 21/256, 1])
        grey = np.array([189/256, 189/256, 189/256, 1])
        yellow = np.array([255/256, 247/256, 0/256, 1])
        red = np.array([1, 0, 0, 1])

        newcolors[mapping < 0.5] = grey
        newcolors[mapping >= 0.5] = green
        newcolors[mapping >= 0.75] = yellow
        newcolors[mapping >= 1] = red
        # newcolors[mapping >= 0.83] = purple
        mycolormap = ListedColormap(newcolors)

        # Add Deformed Shape
        CurrentStepOuts = PushData[str(int(step))]
        NodesUpdated, SCVals = self._UpdateNodes(self.NodeIDs,
                                                 self.Nodes['Coordinates'],
                                                 CurrentStepOuts,
                                                 scale=scalefactor)
        mesh2 = pv.PolyData()
        mesh2.points = NodesUpdated
        mesh2.lines = PaddedEdgesChecked
        mesh2.cell_data['Colors'] = ColorsChecked

        self.p.add_mesh(mesh2, render_lines_as_tubes=False,
                        style='wireframe',
                        line_width=3,
                        opacity=1,
                        scalars='Colors',
                        cmap=mycolormap)

        if showDCRframeShear == 1:
            labels = [str(round(i, 2)) for i in DCRShear]
            self.p.add_point_labels(MidF, labels, font_size=15,
                                    point_size=0.1, text_color='red')

        mesh3 = pv.PolyData()
        mesh3.points = NodesUpdated
        mesh3.lines = PaddedEdgesOther
        mesh3.cell_data['Colors'] = ColorsOther
        self.p.add_mesh(mesh3, render_lines_as_tubes=False,
                        style='wireframe',
                        line_width=3,
                        opacity=0.2,
                        scalars='Colors',
                        cmap='jet')

        # Add BeamColumn Joint Nodes
        # BCJCheckType
        # 1 : Deformation Based DS
        # 2 : Force Based [Backbone]
        # 3 : Force Based [EC8]
        # 4 : Force Based [NTC]

        BCJFile = os.path.join(PushoverOuts, 'OutBCJ.out')
        BCJData = np.genfromtxt(BCJFile)
        BCJIDX = np.where(np.isin(self.NodeIDs, BCJData[1, :]))[0]
        BCJNodesUpdated = np.asarray(NodesUpdated)[BCJIDX, :]
        BCJCheckType = 1
        if BCJCheckType in (1, '1', 'Deformation'):
            BCJCheckType = 'BCJDeformation'
            DCRBCJDefDS1File = os.path.join(PushoverOuts, 'BeamColumnJointDCR',
                                            'DCR_DS1Rotation.csv')
            DCRBCJDefDS2File = os.path.join(PushoverOuts, 'BeamColumnJointDCR',
                                            'DCR_DS2Rotation.csv')
            DCRBCJDefDS3File = os.path.join(PushoverOuts, 'BeamColumnJointDCR',
                                            'DCR_DS3Rotation.csv')
            DCRBCJDefDS1Data = np.genfromtxt(DCRBCJDefDS1File)
            DCRBCJDefDS2Data = np.genfromtxt(DCRBCJDefDS2File)
            DCRBCJDefDS3Data = np.genfromtxt(DCRBCJDefDS3File)
            DCRBCJData = [DCRBCJDefDS1Data,
                          DCRBCJDefDS2Data,
                          DCRBCJDefDS3Data]
        elif BCJCheckType in (2, '2', 'ForceBackbone'):
            BCJCheckType = 'BCJForceBackbone'
            DCRBCJForceBBFile = os.path.join(PushoverOuts,
                                             'BeamColumnJointDCR',
                                             'DCRBCJForceBB.csv')
            DCRBCJForceBB = np.genfromtxt(DCRBCJForceBBFile)
            DCRBCJData = [DCRBCJForceBB]
        elif BCJCheckType in (3, '3', 'EC8'):
            BCJCheckType = 'BCJEC8'
            DCRBCJForceEC8File = os.path.join(PushoverOuts,
                                              'BeamColumnJointDCR',
                                              'DCRBCJForceEC8.csv')
            DCRBCJForceEC8 = np.genfromtxt(DCRBCJForceEC8File)
            DCRBCJData = [DCRBCJForceEC8]
        elif BCJCheckType in (4, '4', 'NTC'):
            BCJCheckType = 'BCJNTC'
            DCRBCJForceNTCCFile = os.path.join(PushoverOuts,
                                               'BeamColumnJointDCR',
                                               'DCRBCJForceNTCC.csv')
            DCRBCJForceNTCTFile = os.path.join(PushoverOuts,
                                               'BeamColumnJointDCR',
                                               'DCRBCJForceNTCT.csv')
            DCRBCJForceNTCC = np.genfromtxt(DCRBCJForceNTCCFile)
            DCRBCJForceNTCT = np.genfromtxt(DCRBCJForceNTCTFile)
            DCRBCJForceNTC = np.maximum(DCRBCJForceNTCC, DCRBCJForceNTCT)
            DCRBCJData = [DCRBCJForceNTC]
        BCJColors = []
        for BCJNodeID in BCJData[1, :]:
            CurCol, _ = self._findColor(step, BCJNodeID, BCJData[1, :],
                                        DCRBCJData, BCJCheckType)
            BCJColors.append(CurCol)

        if BCJCheckType == 'BCJDeformation':
            print('Number of Cracked Beam Column Joints '
                  f'{len(np.where(np.asarray(BCJColors) == 0.5)[0])}')
            print('IDs of Cracked Beam Column Joints '
                  f'{BCJData[1,np.where(np.asarray(BCJColors) == 0.5)[0]]}')
            print('Number of Peak Beam Column Joints '
                  f'{len(np.where(np.asarray(BCJColors) == 0.75)[0])}')
            print('IDs of Peak Beam Column Joints '
                  f'{BCJData[1,np.where(np.asarray(BCJColors) == 0.75)[0]]}')
            print('Number of Ultimate Beam Column Joints '
                  f'{len(np.where(np.asarray(BCJColors) == 1)[0])}')
            print('IDs of Ultimate Beam Column Joints '
                  f'{BCJData[1,np.where(np.asarray(BCJColors) == 1)[0]]}')
        else:
            print(f'Number of Failed Beam Column Joints '
                  f'{len(np.where(np.asarray(BCJColors) == 1)[0])}')
            print(f'IDs of Failed Beam Column Joints '
                  f'{BCJData[1,np.where(np.asarray(BCJColors) == 1)[0]]}')
        BCJp = pv.PolyData(BCJNodesUpdated)
        BCJp['Colors'] = BCJColors
        BCJp['Labels'] = BCJData[1, :]
        self.p.add_mesh(BCJp, point_size=10,
                        opacity=1,
                        scalars='Colors',
                        cmap=mycolormap)
        # p.add_point_labels(BCJp, "Labels")

        # Frame Hinges
        DCRFrameDeformationIFile = os.path.join(PushoverOuts,
                                                'FrameRotationDCR',
                                                'DCR_I.csv')
        DCRFrameDeformationJFile = os.path.join(PushoverOuts,
                                                'FrameRotationDCR',
                                                'DCR_J.csv')
        DCRFrameYeildIFile = os.path.join(PushoverOuts,
                                          'FrameRotationDCR',
                                          'DCR_Yeild_I.csv')
        DCRFrameYeildJFile = os.path.join(PushoverOuts,
                                          'FrameRotationDCR',
                                          'DCR_Yeild_I.csv')

        DCRFrameDeformationI = np.genfromtxt(DCRFrameDeformationIFile)
        DCRFrameDeformationJ = np.genfromtxt(DCRFrameDeformationJFile)
        DCRFrameYeildI = np.genfromtxt(DCRFrameYeildIFile)
        DCRFrameYeildJ = np.genfromtxt(DCRFrameYeildJFile)

        FrameDefData = [DCRFrameDeformationI,
                        DCRFrameDeformationJ,
                        DCRFrameYeildI,
                        DCRFrameYeildJ]
        HingeColors = []
        HingePoints = []

        NCFrames = []
        SDFrames = []
        DLFrames = []
        for i in range(FrameData.shape[1]):
            FrameID = FrameData[0, i]
            INodeID = FrameData[1, i]
            JNodeID = FrameData[2, i]
            CurCol, CurDCRs = self._findColor(step, FrameID, FrameData[0, :],
                                              FrameDefData, 'FrameDef')
            INodeUpdated = np.asarray(NodesUpdated)[
                np.where(self.NodeIDs == INodeID)][0]
            JNodeUpdated = np.asarray(NodesUpdated)[
                np.where(self.NodeIDs == JNodeID)][0]
            INodeShowCoord = self._find_third_point(INodeUpdated,
                                                    JNodeUpdated, 0.05)
            JNodeShowCoord = self._find_third_point(INodeUpdated,
                                                    JNodeUpdated, 0.95)
            HingeColors.append(CurCol[0])
            HingeColors.append(CurCol[1])
            if max(CurCol) == 1:
                NCFrames.append(FrameID)
            elif max(CurCol) == 0.75:
                SDFrames.append(FrameID)
            elif max(CurCol) == 0.5:
                DLFrames.append(FrameID)
            HingePoints.append(INodeShowCoord)
            HingePoints.append(JNodeShowCoord)

        Hingesp = pv.PolyData(HingePoints)
        Hingesp['Colors'] = HingeColors
        self.p.add_mesh(Hingesp, point_size=10,
                        render_points_as_spheres=True,
                        opacity=0.75,
                        scalars='Colors',
                        cmap=mycolormap)

        print(f'Number of Frames Damage Limitation {len(DLFrames)}')
        print(f'IDs of Frames Damage Limitation {DLFrames}')
        print(f'Number of Frames Significant Damage {len(SDFrames)}')
        print(f'IDs of Frames Significant Damage {SDFrames}')
        print(f'Number of Frames Near Collapse {len(NCFrames)}')
        print(f'IDs of Frames Near Collapse {NCFrames}')

        # Add Pushover Chart
        chart = pv.Chart2D(size=(0.33, 0.33),
                           loc=(0.66, 0),
                           x_label='Top Displacement [m]',
                           y_label='Base Shear', grid=True)
        _ = chart.line(TopDisp, BaseShear, color='r', width=5)
        _ = chart.scatter([TopDisp[step]], [BaseShear[step]],
                          color='b', size=10)
        self.p.add_chart(chart)
        self.p.remove_scalar_bar()

        if showme == 1:
            self.p.show(auto_close=False)
        # Handle display and saving properly
        if saveme == 1:
            html_path = os.path.join(PushoverOuts,
                                     'PushoverResultsPlot.html')
            self.p.export_html(html_path)
            print(f"Pushover plot saved to {html_path}")
        if showme == 1 or saveme == 1:  # Clear if shown or static-saved
            self.p.clear()

    def EC8N2Res(self, EC8CheckDF, checkID='', showDCRframeShear=0,
                 showme=1, saveme=0):
        """
        Visualise EC8 N2 method seismic assessment results.

        Parameters
        ----------
        EC8CheckDF : DataFrame
            EC8 analysis results dataframe.
        checkID : str, optional
            Identification string for output files (default: '').
        showDCRframeShear : int, optional
            Show frame shear DCR values (default: 0).
        showme : int, optional
            Show plot interactively (default: 1).
        saveme : int, optional
            Save format (0: no save, 1: HTML, 2: SVG) (default: 0).
        """
        mycolormap = ListedColormap(['#C02B2B', '#D32F2F', '#FFA000',
                                     '#FFEB3B', '#808080'])

        self.unDeformated(showme=0, frametags=0, called_from=True)

        # Frame Colors Based on Shear DCR
        EdgesShearFailed = []
        EdgesShearCompliant = []
        DCRShear = []
        MidF = []
        ElementLabel = []
        EdgesOther = []
        NodesShearFailed = []
        NodesShearCompliant = []
        NodesOther = []

        for k in self.Elements.keys():
            EdgeI = self.Elements[k]['i']['Tag']
            mapEdgeI = np.where(np.array(self.NodeIDs) == EdgeI)[0][0]
            EdgeJ = self.Elements[k]['j']['Tag']
            mapEdgeJ = np.where(np.array(self.NodeIDs) == EdgeJ)[0][0]
            if k in EC8CheckDF.columns:
                CurCol, DCR = self._findColor(0, int(k),
                                              EC8CheckDF.columns,
                                              np.array([EC8CheckDF.loc['Shear']
                                                       .values]),
                                              'FrameShear')
                # If Failed
                if DCR[0] >= 1:
                    EdgesShearFailed.append([mapEdgeI,
                                             mapEdgeJ])
                    coord_i = self.Nodes['Coordinates'][mapEdgeI]
                    coord_j = self.Nodes['Coordinates'][mapEdgeJ]
                    NodesShearFailed.append(coord_i)
                    NodesShearFailed.append(coord_j)
                else:
                    EdgesShearCompliant.append([mapEdgeI,
                                                mapEdgeJ])
                    coord_i = self.Nodes['Coordinates'][mapEdgeI]
                    coord_j = self.Nodes['Coordinates'][mapEdgeJ]
                    NodesShearCompliant.append(coord_i)
                    NodesShearCompliant.append(coord_j)
                DCRShear.append(DCR[0])
                ElementLabel.append(k)
                CurMidF = (np.array(self.Nodes['Coordinates'][mapEdgeI]) +
                           np.array(self.Nodes['Coordinates'][mapEdgeJ]))*0.5
                MidF.append(CurMidF)
            else:
                EdgesOther.append([mapEdgeI,
                                   mapEdgeJ])
                NodesOther.append(self.Nodes['Coordinates'][mapEdgeI])
                NodesOther.append(self.Nodes['Coordinates'][mapEdgeJ])
        EdgesOther = np.array(EdgesOther)
        paddingOther = np.empty(EdgesOther.shape[0], int) * 2
        paddingOther[:] = 2
        PaddedEdgesOther = np.vstack((paddingOther, EdgesOther.T)).T

        EdgesShearFailed = np.array(EdgesShearFailed)
        # We must "pad" the edges to indicate to vtk how many points per edge
        paddingEdgesShearFailed = np.empty(EdgesShearFailed.shape[0], int) * 2
        paddingEdgesShearFailed[:] = 2

        PaddedEdgesShearFailed = np.vstack((paddingEdgesShearFailed,
                                            EdgesShearFailed.T)).T
        
        EdgesShearCompliant = np.array(EdgesShearCompliant)
        # We must "pad" the edges to indicate to vtk how many points per edge
        paddingEdgesShearCompliant = np.empty(
            EdgesShearCompliant.shape[0], int) * 2
        paddingEdgesShearCompliant[:] = 2

        PaddedEdgesShearCompliant = np.vstack((paddingEdgesShearCompliant,
                                               EdgesShearCompliant.T)).T

        if PaddedEdgesShearFailed.shape[0] != 0:
            mesh_Shear_Failed = pv.PolyData()
            mesh_Shear_Failed.points = np.array(self.Nodes['Coordinates'])
            mesh_Shear_Failed.lines = PaddedEdgesShearFailed
            self.p.add_mesh(mesh_Shear_Failed, render_lines_as_tubes=False,
                            style='wireframe',
                            line_width=3,
                            opacity=1,
                            color='#D32F2F',
                            cmap=mycolormap,
                            show_scalar_bar=False)

        if PaddedEdgesShearCompliant.shape[0] != 0:
            mesh_Shear_Compliant = pv.PolyData()
            mesh_Shear_Compliant.points = np.array(self.Nodes['Coordinates'])
            mesh_Shear_Compliant.lines = PaddedEdgesShearCompliant

            self.p.add_mesh(mesh_Shear_Compliant, render_lines_as_tubes=False,
                            style='wireframe',
                            line_width=3,
                            opacity=1,
                            color='Black',
                            cmap=mycolormap,
                            show_scalar_bar=False)

        if showDCRframeShear == 1:
            labels = [str(round(i, 2)) for i in DCRShear]
            self.p.add_point_labels(MidF, labels, font_size=15,
                                    point_size=0.1, text_color='red',
                                    fill_shape=False)
        
        # Other Elements [Infills]
        # NodesOther = np.unique(np.array(NodesOther), axis=0)
        # NodesOther = np.array(NodesOther)
        mesh3 = pv.PolyData()
        mesh3.points = np.array(self.Nodes['Coordinates'])
        mesh3.lines = PaddedEdgesOther
        self.p.add_mesh(mesh3, render_lines_as_tubes=False,
                        style='wireframe',
                        line_width=3,
                        opacity=0.2,
                        color='brown',
                        cmap=mycolormap,
                        show_scalar_bar=False)
        
        # Hinges
        Nodes = np.array(self.Nodes['Coordinates'])
        PushoverOuts = os.path.join(self.OutDir, 'PushoverDir-1')
        FramesFile = os.path.join(PushoverOuts, 'OutFrames.out')
        FrameData = np.genfromtxt(FramesFile)
        HingeColors = []
        HingePoints = []
        NCFrames = []
        SDFrames = []
        DLFrames = []
        NC_Nodes = []
        SD_Nodes = []
        DL_Nodes = []
        Other_Nodes = []
        for i in range(FrameData.shape[1]):
            FrameID = FrameData[0, i]
            INodeID = FrameData[1, i]
            JNodeID = FrameData[2, i]
            CurCol, _ = self._findColor(
                0, str(int(FrameID)), EC8CheckDF.columns,
                np.array([EC8CheckDF.loc['RotationI'].values,
                          EC8CheckDF.loc['RotationJ'].values,
                          EC8CheckDF.loc['YeildI'].values,
                          EC8CheckDF.loc['YeildJ'].values]), 'FrameDef')
            INode = Nodes[
                np.where(self.NodeIDs == INodeID)][0]
            JNode = Nodes[
                np.where(self.NodeIDs == JNodeID)][0]
            INodeShowCoord = self._find_third_point(INode,
                                                    JNode, 0.05)
            JNodeShowCoord = self._find_third_point(INode,
                                                    JNode, 0.95)
            HingeColors.append(CurCol[0])
            HingeColors.append(CurCol[1])

            if CurCol[0] == '#C02B2B':
                NC_Nodes.append(INodeShowCoord)
            elif CurCol[0] == '#FFA000':
                SD_Nodes.append(INodeShowCoord)
            elif CurCol[0] == '#FFEB3B':
                DL_Nodes.append(INodeShowCoord)
            elif CurCol[0] == '#808080':
                Other_Nodes.append(INodeShowCoord)

            if CurCol[1] == '#C02B2B':
                NC_Nodes.append(JNodeShowCoord)
            elif CurCol[1] == '#FFA000':
                SD_Nodes.append(JNodeShowCoord)
            elif CurCol[1] == '#FFEB3B':
                DL_Nodes.append(JNodeShowCoord)
            elif CurCol[1] == '#808080':
                Other_Nodes.append(JNodeShowCoord)

            if max(CurCol) == '#D32F2F':
                NCFrames.append(FrameID)
            elif max(CurCol) == '#FFA000':
                SDFrames.append(FrameID)
            elif max(CurCol) == '#FFEB3B':
                DLFrames.append(FrameID)
            HingePoints.append(INodeShowCoord)
            HingePoints.append(JNodeShowCoord)

        if len(NC_Nodes) > 0:
            NCHingesP = pv.PolyData(NC_Nodes)
            self.p.add_mesh(NCHingesP, point_size=10,
                            render_points_as_spheres=True,
                            opacity=0.75,
                            color='#C02B2B',
                            cmap=mycolormap,
                            show_scalar_bar=True)
        
        if len(SD_Nodes) > 0:
            SDHingesP = pv.PolyData(SD_Nodes)
            self.p.add_mesh(SDHingesP, point_size=10,
                            render_points_as_spheres=True,
                            opacity=0.75,
                            color='#FFA000',
                            cmap=mycolormap,
                            show_scalar_bar=True)
        
        if len(DL_Nodes) > 0:
            DLHingesP = pv.PolyData(DL_Nodes)
            self.p.add_mesh(DLHingesP, point_size=10,
                            render_points_as_spheres=True,
                            opacity=0.75,
                            color='#FFEB3B',
                            cmap=mycolormap,
                            show_scalar_bar=True)
        
        if len(Other_Nodes) > 0:
            OtherHingesP = pv.PolyData(Other_Nodes)
            self.p.add_mesh(OtherHingesP, point_size=10,
                            render_points_as_spheres=True,
                            opacity=0.75,
                            color='#808080',
                            cmap=mycolormap,
                            show_scalar_bar=True)

        if showme == 1:
            self.p.show(auto_close=False)
        if saveme == 1:
            EC8ResFile = os.path.join(self.OutDir, f'EC8Res_{checkID}.html')
            self.p.export_html(EC8ResFile)
        elif saveme == 2:
            self.p.view_isometric()
            self.p.camera.azimuth = 180
            self.p.camera.elevation = -20
            self.p.camera.zoom(0.75)
            self.p.camera.focal_point = [15, 13, 8.1]
            self.p.camera.position = [-25, -25, 20]
            pv.global_theme.transparent_background = True
            # Add custom scale bar
            self.p.show(auto_close=False)
            self.p.window_size = [1920, 1920]
            EC8ResFile = f'EC8Res_{checkID}.svg'
            self.p.save_graphic(filename=EC8ResFile, raster=False,
                                painter=True)

