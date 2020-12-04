"""
aRMSD plot functions
(c) 2017 by Arne Wagner
"""

# Authors: Arne Wagner
# License: MIT

from __future__ import absolute_import, division, print_function

from builtins import range
import sys

try:
    
    import numpy as np

except ImportError:

    pass

try:
    
    from vtk import (vtkCellPicker, vtkSphereSource, vtkLineSource, vtkTubeFilter, vtkPolyDataMapper, vtkActor,
                     vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor, vtkRenderLargeImage, vtkPNGWriter,
                     vtkWindowToImageFilter, vtkCamera, vtkVectorText, vtkFollower, vtkArrowSource, vtkCubeSource,
                     vtkLegendBoxActor, vtkMath, vtkMatrix4x4, vtkTransformPolyDataFilter, vtkTransform, vtkLookupTable,
                     vtkScalarBarActor, vtkScalarBarWidget, vtkInteractorStyleTrackballCamera, vtkProperty,
                     vtkPropPicker, VTK_VERSION)

    has_vtk, vtk_version = True, VTK_VERSION
    
except ImportError:
    
    has_vtk = False
    vtk_version = 'Module not available'

try:
    
    import matplotlib as mpl
    has_mpl, mpl_version = True, mpl.__version__

    if sys.version_info <= (3,0):

        mpl.use('QT4Agg')  # Set MPL backend to QT4Agg

    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    
except ImportError:
    
    has_mpl = False
    mpl_version = 'Module not available'

try:
    
    from uncertainties import unumpy as unp
    from uncertainties import ufloat
    has_uc = True

except ImportError:

    try:

        import unumpycore as unp
        from ucore import ufloat, ufloat_fromstr

    except ImportError:

        pass

# Matplotlib/pyplot settings, Set Backend to QT4Agg
# C:\Python\Lib\site-packages\matplotlib\mpl-data\matplotlibrc
almost_black = '#262626'
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.edgecolor'] = almost_black
mpl.rcParams['axes.labelcolor'] = almost_black

# Copy structural properties from core module


def geo_distance(xyz1, xyz2):
    """ Global function for distance calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    return np.sum((xyz1 - xyz2)**2)**0.5


def geo_angle(xyz1, xyz2, xyz3):
    """ Global function for angle calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    v1, v2 = xyz1 - xyz2, xyz3 - xyz2

    dv1_dot_dv2 = np.sum(v1**2)**0.5 * np.sum(v2**2)**0.5

    return (180.0/np.pi) * unp.arccos(np.dot(v1, v2) / dv1_dot_dv2)


def geo_torsion(xyz1, xyz2, xyz3, xyz4):
    """ Global function for torsion calculation - compatible with uncertainties
        coordinates are assumed to be uarrays """

    b0 = -1.0 * (xyz2 - xyz1)
    b1 = xyz3 - xyz2
    b2 = xyz4 - xyz3

    b0xb1, b1xb2 = np.cross(b0, b1), np.cross(b2, b1)  # Planes defined by the vectors
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1) * (1.0 / np.sum(b1**2)**0.5)
    x = np.dot(b0xb1, b1xb2)

    return np.abs((180.0/np.pi) * unp.arctan2(y, x))  # Ignore sign of the dihedral angle


###############################################################################
# VTK ROUTINES
###############################################################################

class aRMSD_substructure_picker(vtkInteractorStyleTrackballCamera):
    """ Class for the fractional coordinates / aRMSD substructure selection """
 
    def __init__(self, settings, atoms_to_pick, align, plot_type, picker_type):
        """ Initializes the picker interactor """

        self.plot_type = plot_type
        
        self.AddObserver('LeftButtonPressEvent', self.leftButtonPressEvent)

        # Arrays for picked atoms and actors
        self.PickedAtoms, self.PickedActors = np.array([], dtype=np.int), np.array([], dtype=np.int)
 
        self.LastPickedActor = None
        self.LastPickedProperty = vtkProperty()

        self.actors_to_pick = np.asarray(atoms_to_pick)
        self.picker_color = settings.picker_col_rgb
        self.picker_type = picker_type
        self.NewPickedActor = None

        self.sym_idf = align.sym_idf
        self.bnd_idx = align.bnd_idx
        self.colors = align.col_glob_rgb

    def full_connects(self, idx):
        """ Determines the all positions ultimately attached to the given atom """

        def _is_connected_to(idx):
            """ Determines the connections of the given index """

            ravel_bnds = np.ravel(self.bnd_idx[np.where(self.bnd_idx == idx)[0]])
            pos = np.where(ravel_bnds != idx)[0]

            return ravel_bnds[pos]

        # Set up initial connection array and evaluate first index
        connection_array = np.asarray(idx, dtype=np.int)
        connection_array = np.unique(np.hstack((connection_array, _is_connected_to(idx))))

        checked_pos = [idx]  # This list contains all positions that have been checked

        if len(connection_array) == 1:  # No atoms are connected to the picked one

            pass

        else:

            while True:  # Stay in this loop until no additional indices are added

                old_len = len(connection_array)

                for pos in connection_array:

                    if pos not in checked_pos:  # Evaluate only once

                        connection_array = np.unique(np.hstack((connection_array, _is_connected_to(pos))))
                        checked_pos.append(pos)
                        new_len = len(connection_array)

                if new_len == old_len:  # Exit loop if no changes occurred after all position were checked

                    break

        return connection_array

    def click_message(self, sym_idf, picker_type):
        """ Message displayed to user when an atom is clicked """

        if self.plot_type == 'substructure':

            print("> Atom "+str(sym_idf)+" has been added to 'substructure 1' ...")

        elif self.plot_type == 'fractional':

            if picker_type == 'cluster':

                print("> All atoms connected to "+str(sym_idf)+" will be removed ...")

            else:

                print("> Atom "+str(sym_idf)+" will be removed ...")

    def second_click_message(self, sym_idf, picker_type):
        """ Message displayed to user when a selected atom is clicked """

        if self.plot_type == 'substructure':

            print("> Atom "+str(sym_idf)+" has been removed from 'substructure 1' ...")

        elif self.plot_type == 'fractional':

            if picker_type == 'cluster':

                print("> Removal of all atoms connected to "+str(sym_idf)+" was cancelled ...")

            else:

                print("> Removal of atom "+str(sym_idf)+" was cancelled ...")

    def leftButtonPressEvent(self, obj, event):
        """ Event that will happen on left mouse click """
        
        clickPos = self.GetInteractor().GetEventPosition()  # Get the clicked position
 
        picker = vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
 
        self.NewPickedActor = picker.GetActor()  # Get the actual actor on the clicked position

        # If an actor/atom has been selected (only selective pick events via actors_to_pick)
        if self.NewPickedActor is not None and self.NewPickedActor in self.actors_to_pick:

            atom_idx = int(np.where(self.NewPickedActor == self.actors_to_pick)[0])  # Index of the atom

            if atom_idx not in self.PickedAtoms:  # Select only if it wasn't selected so far

                # Highlight the atom by changing the color
                self.click_message(self.sym_idf[atom_idx], self.picker_type)
                self.NewPickedActor.GetProperty().SetColor(self.picker_color)

                if self.picker_type == 'cluster':

                    all_positions = self.full_connects(atom_idx)
                    self.PickedActors = np.unique(np.append(self.PickedActors, self.actors_to_pick[all_positions]))
                    self.PickedAtoms = np.unique(np.append(self.PickedAtoms, all_positions))

                    # Change colors for all atoms
                    [actor.GetProperty().SetColor(self.picker_color) for actor in self.PickedActors]

                else:

                    self.PickedActors = np.unique(np.append(self.PickedActors, self.actors_to_pick[atom_idx]))
                    self.PickedAtoms = np.unique(np.append(self.PickedAtoms, atom_idx))

            else:  # Remove duplicates

                self.second_click_message(self.sym_idf[atom_idx], self.picker_type)

                if self.picker_type == 'cluster':  # Change all connected atoms

                    all_positions = self.full_connects(atom_idx)
                    pos_in_picked_atoms = np.ravel(np.asarray([np.where(self.PickedAtoms == pos)[0]
                                                               for pos in all_positions]))

                    self.PickedActors = np.unique(np.asarray([np.delete(self.PickedActors, np.where(self.PickedActors == self.actors_to_pick[pos])[0]) for pos in all_positions]))  # Remove actor from array
                    self.PickedAtoms = np.unique(np.delete(self.PickedAtoms, pos_in_picked_atoms, axis=0))  # Remove atomic index from index array
                    
                    [actor.GetProperty().SetColor(self.colors) for actor in self.PickedActors]  # Change colors for all atoms

                else:

                    self.PickedActors = np.unique(np.delete(self.PickedActors, np.where(self.PickedActors == self.actors_to_pick[atom_idx])[0]))  # Remove actor from array
                    self.PickedAtoms = np.unique(np.delete(self.PickedAtoms, np.where(self.PickedAtoms == atom_idx)[0]))  # Remove atomic index from index array

                    self.NewPickedActor.GetProperty().SetColor(self.colors)  # Reset the color to the initial value

        self.OnLeftButtonDown()
        return

# ---------------------------------------------------------------------------------

class aRMSD_plot_picker(vtkInteractorStyleTrackballCamera):
    """ Class for picking events in the aRMSD plot """
 
    def __init__(self, settings, atoms_to_pick, align):
        """ Initializes the picker interactor """
        
        self.AddObserver('LeftButtonPressEvent', self.leftButtonPressEvent)
        self.PickedAtoms, self.PickedActors = [], []  # Lists for picked atoms and actors
 
        self.LastPickedActor = None
        self.LastPickedProperty = vtkProperty()

        self.actors_to_pick = np.asarray(atoms_to_pick)
        self.picker_color = settings.picker_col_rgb
        self.std_type = settings.std_type
        self.calc_prec = settings.calc_prec
        self.use_std = settings.use_std

        self.sym_idf = align.sym_idf
        self.coords = align.cor
        self.coords_mol1 = align.cor_mol1_kbs
        self.coords_mol2 = align.cor_mol2_kbs
        self.coords_std_mol1 = align.cor_mol1_kbs_std
        self.coords_std_mol2 = align.cor_mol2_kbs_std
        self.colors = align.col_at_rgb
        self.name_mol1 = align.name1
        self.name_mol2 = align.name2
        self.RMSD_per_atom = align.msd_sum**0.5

        self.rmsd_perc = (align.msd_sum / np.sum(align.msd_sum)) * 100  # Contribution of individual atom types

    def calc_picker_property(self, list_of_picks):
        """ Calculates distances, angles or dihedral angles with or without uncertainties """

        def _proper_std(stds, list_of_picks):

            if self.std_type == 'simple':  # Check only if stds exist

                return True

            else:  # H/and some heavy atoms may have no stds

                return 0.0 not in np.sum(stds[np.asarray(list_of_picks)], axis=1)

        def _per_mol(coords, stds):
            """ Calculate for one molecule """

            if self.use_std:  # Combine coordinates and uncertainties to array

                xyz = unp.uarray(coords[np.asarray(list_of_picks)], stds[np.asarray(list_of_picks)])

            else:

                xyz = coords[np.asarray(list_of_picks)]

            if len(list_of_picks) == 2:  # Distance

                value = geo_distance(xyz[0], xyz[1])

            elif len(list_of_picks) == 3:  # Angle

               value = geo_angle(xyz[0], xyz[1], xyz[2])

            elif len(list_of_picks) == 4:  # Torsion angle

                value = geo_torsion(xyz[0], xyz[1], xyz[2], xyz[3])

            return ufloat(value.nominal_values, 0.0) if not _proper_std(stds, list_of_picks) else value

        p1, p2 = _per_mol(self.coords_mol1, self.coords_std_mol1), _per_mol(self.coords_mol2, self.coords_std_mol2)

        delta = p2 - p1

        return p1, p2, delta

    def calc_property(self, list_of_picks):
        """ Calculates different structural properties """

        def apply_format(value, n_digits):

            str_len = 12

            ft_str_norm = '{:3.2f}'

            if n_digits != 0:

                ft_str_norm = '{:'+str(n_digits)+'.'+str(n_digits)+'f}'
                ft_str_unce = '{:.1uS}'  # One digit for values with uncertainties

            if self.use_std:  # If standard deviations exist

                if value.std_dev == 0.0 or n_digits == 0:  # Different format for values without standard deviations

                    if n_digits == 0:

                        str_len = 5

                    add = str_len - len(ft_str_norm.format(value.nominal_value))

                    if n_digits == 0 and value.nominal_value < 10.0:

                        return '0'+ft_str_norm.format(value.nominal_value)+' '*(add-1)

                    else:

                        return ft_str_norm.format(value.nominal_value)+' '*add

                else:

                    add = str_len - len(ft_str_unce.format(value))

                    return ft_str_unce.format(value)+' '*add

            else:  # No ufloat values

                return ft_str_norm.format(value)

        def print_values(values, n_digits, unit=' deg.'):

            print('\n           '+str(self.name_mol1)+': '+apply_format(values[0], n_digits)+unit+
                  '\n       '+str(self.name_mol2)+': '+apply_format(values[1], n_digits)+unit+
                  '\t\tDiff. = '+apply_format(values[2], n_digits)+unit)

        if len(list_of_picks) == 1:  # Show RMSD contribution of the atom

            print('\nAtom [' +str(self.sym_idf[list_of_picks[0]])+']: RMSD = '+
                  apply_format(self.RMSD_per_atom[list_of_picks[0]], 3)+
                  ' Angstrom ('+apply_format(self.rmsd_perc[list_of_picks[0]], 0)+' % of the total RMSD)')

        elif len(list_of_picks) == 2:  # Calculate distance

            d1, d2, delta = self.calc_picker_property(list_of_picks)

            print('\nDistance between: ['+str(self.sym_idf[list_of_picks[0]])+' -- '+
                  str(self.sym_idf[list_of_picks[1]])+']')
            print_values([d1, d2, delta], n_digits=5, unit=' A')

        elif len(list_of_picks) == 3:  # Calculate angle

            a1, a2, delta = self.calc_picker_property(list_of_picks)

            print('\nAngle between: ['+str(self.sym_idf[list_of_picks[0]])+' -- '+
                  str(self.sym_idf[list_of_picks[1]])+' -- '+str(self.sym_idf[list_of_picks[2]])+']')
            print_values([a1, a2, delta], n_digits=5, unit=' deg.')

        elif len(list_of_picks) == 4:  # Calculate dihedral angle

            t1, t2, delta = self.calc_picker_property(list_of_picks)

            print('\nDihedral between: ['+str(self.sym_idf[list_of_picks[0]])+' -- '+
                  str(self.sym_idf[list_of_picks[1]])+' -- '+str(self.sym_idf[list_of_picks[2]])+' -- '+
                  str(self.sym_idf[list_of_picks[3]])+']')
            print_values([t1, t2, delta], n_digits=5, unit=' deg.')

    def leftButtonPressEvent(self, obj, event):
        """ Event that will happen on left mouse click """
        
        clickPos = self.GetInteractor().GetEventPosition()  # Get the clicked position
 
        picker = vtkPropPicker()
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())
 
        self.NewPickedActor = picker.GetActor()  # Get the actual actor on the clicked position

        # If an actor/atom has been selected (only selective pick events via actors_to_pick)
        if self.NewPickedActor is not None and self.NewPickedActor in self.actors_to_pick:

            atom_idx = int(np.where(self.NewPickedActor == self.actors_to_pick)[0])  # Index of the atom

            if len(self.PickedAtoms) <= 3:  # Maximum selection will be 4 atoms

                if atom_idx not in self.PickedAtoms:  # Select only if it wasn't selected so far

                    self.PickedActors.append(self.actors_to_pick[atom_idx])
                    self.PickedAtoms.append(atom_idx)
                    self.calc_property(self.PickedAtoms)

                    # Highlight the atom by changing the color
                    self.NewPickedActor.GetProperty().SetColor(self.picker_color)

                else:  # Remove duplicates

                    self.PickedActors.remove(self.actors_to_pick[atom_idx])  # Remove actor from list
                    self.PickedAtoms.remove(atom_idx)  # Remove atomic index from indices list
                    self.calc_property(self.PickedAtoms)

                    # Reset the color to the initial value
                    self.NewPickedActor.GetProperty().SetColor(self.colors[atom_idx])

            else:  # Reset all colors

                colors = [self.colors[index] for index in self.PickedAtoms]
                [self.PickedActors[index].GetProperty().SetColor(colors[index]) for
                 index in range(len(self.PickedActors))]

                self.PickedActors, self.PickedAtoms = [], []  # Empty the lists

        self.OnLeftButtonDown()
        return


class Molecular_Viewer_vtk(object):
    """ A molecular viewer object based on vtk used for 3d plots """
 
    def __init__(self, settings):
        """ Initializes object and creates the renderer and camera """

        self.ren = vtkRenderer()
        self.ren_win = vtkRenderWindow()
        self.ren_win.AddRenderer(self.ren)
        
        self.ren.SetBackground(settings.backgr_col_rgb)
        self.ren.SetUseDepthPeeling(settings.use_depth_peel)
               
        self.title = 'aRMSD Structure Visualizer'
        self.magnif = None
        self.save_counts = 0

        self.picker = None

        # Create the active camera
        self.camera = vtkCamera()
        self.camera.SetPosition(np.array([0.0, 0.0, 50]))
        self.ren.SetActiveCamera(self.camera)

        self.bnd_eps = 1.0E-03
        self.at_actors_list = []            # List of atomic actors (for pick events)

        # Create a renderwindowinteractor
        self.iren = vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)

    def show(self, molecule1, molecule2, settings):
        """ Shows the results in a new window """

        self.magnif = settings.magnif_fact  # Copy magnification information from settings

        # Determine file names for screenshots
        if self.plot_type == 'initial':

            self.png_file_name = 'VTK_initial_plot'

        elif self.plot_type == 'inertia':

            self.png_file_name = 'VTK_inertia_plot'

        elif self.plot_type == 'aRMSD':

            self.png_file_name = 'VTK_aRMSD_plot'

        elif self.plot_type == 'superpos':

            self.png_file_name = 'VTK_superposition_plot'

        elif self.plot_type == 'substructure':

            self.png_file_name = 'VTK_substructure_plot'

        elif self.plot_type == 'fractional':

            self.png_file_name = 'VTK_fractional_plot'

        if self.has_cam_vtk:  # Set camera properties (if they exist...)

            self.camera.SetPosition(self.cam_vtk_pos)
            self.camera.SetFocalPoint(self.cam_vtk_focal_pt)
            self.camera.SetViewUp(self.cam_vtk_view_up)

        self.iren.Initialize()
        self.ren_win.SetSize(settings.window_size)
        self.ren_win.SetWindowName(self.title)

        self.iren.AddObserver('KeyPressEvent', self.keypress)  # Key events for screenshots, etc.
        
        self.ren_win.Render()
        self.iren.Start()

        # Determine the camera properties of the final orientation and store them
        molecule1.cam_vtk_pos, molecule2.cam_vtk_pos = self.camera.GetPosition(), self.camera.GetPosition()
        molecule1.cam_vtk_wxyz, molecule2.cam_vtk_wxyz = self.camera.GetOrientationWXYZ(), self.camera.GetOrientationWXYZ()
        molecule1.cam_vtk_focal_pt, molecule2.cam_vtk_focal_pt = self.camera.GetFocalPoint(), self.camera.GetFocalPoint()
        molecule1.cam_vtk_view_up, molecule2.cam_vtk_view_up = self.camera.GetViewUp(), self.camera.GetViewUp()
        molecule1.has_cam_vtk, molecule2.has_cam_vtk = True, True

        del self.ren_win, self.iren

        if self.picker is not None and self.plot_type in ['substructure', 'fractional']:

            return np.ravel(np.asarray(self.picker.PickedAtoms, dtype=np.int))

        #self.close_window()

    def close_window(self):
        """ Not working, but intended to close the window """

        self.ren_win.Finalize()
        self.iren.TerminateApp()

    def keypress(self, obj, event):
        """ Function that handles key pressing events """

        key = obj.GetKeySym()

        if key == 's':  # Screenshots

            render_large = vtkRenderLargeImage()
            render_large.SetInput(self.ren)
            render_large.SetMagnification(self.magnif)
            writer = vtkPNGWriter()
            writer.SetInputConnection(render_large.GetOutputPort())

            if self.save_counts == 0:  # Make sure that screenshots are not overwritten by default

                export_file_name = self.png_file_name+'.png'

            else:

                export_file_name = self.png_file_name+'_'+str(self.save_counts)+'.png'

            writer.SetFileName(export_file_name)
            self.ren_win.Render()
            writer.Write()

            print('\n> The image was saved as '+export_file_name+' !')
            self.save_counts += 1  # Remember save event

            del render_large

        elif key == 'b':  # Add or remove a bond

            pass

        elif key == 'h':  # Display help

            print("\n> Press the 's' button to save the scene as .png file")

    def add_principal_axes(self, com, pa, length, col, settings):
        """ Adds the principal axes of rotation to the view """

        startPoint, endPoint = com*settings.scale_glob, (pa*2 + com)*settings.scale_glob

        normalizedX, normalizedY, normalizedZ = np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float), \
                                                np.zeros(3, dtype=np.float)

        arrow = vtkArrowSource()
        arrow.SetShaftResolution(settings.res_atom)
        arrow.SetTipResolution(settings.res_atom)
        arrow.SetShaftRadius(0.005*10)
        arrow.SetTipLength(0.4)
        arrow.SetTipRadius(0.01*10)

        # The X axis is a vector from start to end
        math = vtkMath()
        math.Subtract(endPoint, startPoint, normalizedX)
        length = math.Norm(normalizedX)
        math.Normalize(normalizedX)
     
        # The Z axis is an arbitrary vector cross X
        arbitrary = np.asarray([0.2, -0.3, 1.7])
        math.Cross(normalizedX, arbitrary, normalizedZ)
        math.Normalize(normalizedZ)
         
        # The Y axis is Z cross X
        math.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtkMatrix4x4()
         
        matrix.Identity()  # Create the direction cosine matrix
        
        for i in range(3):

            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)
         
        # Transform the polydata
        transformPD = vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrow.GetOutputPort())

        # Create a mapper and connect it to the source data, set up actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transformPD.GetOutputPort())
        arrow_actor = vtkActor()
        arrow_actor.GetProperty().SetColor(col)
        arrow_actor.GetProperty().SetLighting(settings.use_light)
        arrow_actor.GetProperty().SetOpacity(settings.alpha_arrow)
        arrow_actor.GetProperty().ShadingOn()
        arrow_actor.SetMapper(mapper)

        self.ren.AddActor(arrow_actor)

    def add_arrow(self, direction, length, settings):
        """ Adds a single arrow defined by length and reference axis """

        # Generate start and end points based on length and reference axis
        if direction == 'x':

            startPoint, endPoint = np.asarray([-length, 0.0, 0.0]), np.asarray([length, 0.0, 0.0])

        elif direction == 'y':

            startPoint, endPoint = np.asarray([0.0, -length, 0.0]), np.asarray([0.0, length, 0.0])

        elif direction == 'z':

            startPoint, endPoint = np.asarray([0.0, 0.0, -length]), np.asarray([0.0, 0.0, length])

        normalizedX, normalizedY, normalizedZ = np.zeros(3, dtype=np.float), np.zeros(3, dtype=np.float), \
                                                np.zeros(3, dtype=np.float)

        arrow = vtkArrowSource()
        arrow.SetShaftResolution(settings.res_atom)
        arrow.SetTipResolution(settings.res_atom)
        arrow.SetShaftRadius(0.005)
        arrow.SetTipLength(0.12)
        arrow.SetTipRadius(0.02)

        # The X axis is a vector from start to end
        math = vtkMath()
        math.Subtract(endPoint, startPoint, normalizedX)
        length = math.Norm(normalizedX)
        math.Normalize(normalizedX)
     
        # The Z axis is an arbitrary vector cross X
        arbitrary = np.asarray([0.2, -0.3, 1.7])
        math.Cross(normalizedX, arbitrary, normalizedZ)
        math.Normalize(normalizedZ)
         
        # The Y axis is Z cross X
        math.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtkMatrix4x4()
         
        matrix.Identity()  # Create the direction cosine matrix
        
        for i in range(3):

            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)
         
        # Transform the polydata
        transformPD = vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrow.GetOutputPort())

        # Create a mapper and connect it to the source data, set up actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(transformPD.GetOutputPort())
        arrow_actor = vtkActor()
        arrow_actor.GetProperty().SetColor(settings.arrow_col_rgb)
        arrow_actor.GetProperty().SetLighting(settings.use_light)
        arrow_actor.GetProperty().SetOpacity(settings.alpha_arrow)
        arrow_actor.GetProperty().ShadingOn()
        arrow_actor.SetMapper(mapper)

        self.ren.AddActor(arrow_actor)

    def add_atom(self, pos, radius, color, settings):
        """ Adds a single atom as vtkSphere with defined radius and color at the given position """

        # Create new SphereSource and define its properties
        atom = vtkSphereSource()
        atom.SetCenter(pos)
        atom.SetRadius(radius*settings.scale_at)
        
        atom.SetPhiResolution(settings.res_atom)
        atom.SetThetaResolution(settings.res_atom)

        # Create a mapper and connect it to the source data, set up actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(atom.GetOutputPort())
        at_actor = vtkActor()
        at_actor.GetProperty().SetColor(color)
        at_actor.GetProperty().SetOpacity(settings.alpha_at)
        at_actor.GetProperty().SetLighting(settings.use_light)
        at_actor.GetProperty().ShadingOn()
        at_actor.SetMapper(mapper)

        self.ren.AddActor(at_actor)
        self.at_actors_list.append(at_actor)

    def add_com(self, molecule, radius, color, settings):
        """ Adds center of mass """

        self.add_atom(molecule.com*settings.scale_glob, radius, color, settings)

    def add_all_atoms(self, molecule, settings):
        """ Wrapper for the addition of all atoms from the molecule """

        if settings.name == 'Wireframe':  # Wireframe plot style

            radii = np.repeat(0.76, molecule.n_atoms)
            color = molecule.col_at_rgb

        elif self.plot_type == 'substructure':  # Substructure selection

            radii = np.repeat(1.5, molecule.n_atoms)
            color = np.transpose(np.repeat(molecule.col_glob_rgb,
                                           molecule.n_atoms).reshape((3, molecule.n_atoms)))

        else:

            radii = molecule.rad_plt_vtk
            color = molecule.col_at_rgb

        [self.add_atom(molecule.cor[atom]*settings.scale_glob, radii[atom],
                       color[atom], settings) for atom in range(molecule.n_atoms)]

    def add_all_atoms_superpos(self, align, settings):
        """ Wrapper for the addition of all atoms for the superposition plot """

        if settings.name == 'Wireframe':

            radii = np.repeat(0.76, align.n_atoms)

        else:

            radii = align.rad_plt_vtk

        [self.add_atom(align.cor_mol1_kbs[atom]*settings.scale_glob, radii[atom],
                       align.col_at_mol1_rgb[atom], settings) for atom in range(align.n_atoms)]

        [self.add_atom(align.cor_mol2_kbs[atom]*settings.scale_glob, radii[atom],
                       align.col_at_mol2_rgb[atom], settings) for atom in range(align.n_atoms)]

    def add_bond(self, first_loc, second_loc, color, settings):
        """ Adds a single bond as vtkLine between two locations """

        if np.linalg.norm(first_loc - second_loc) > self.bnd_eps:

            # Create LineSource and set start and end point
            bnd_source = vtkLineSource()
            bnd_source.SetPoint1(first_loc)
            bnd_source.SetPoint2(second_loc)

            # Create a TubeFilter around the line
            TubeFilter = vtkTubeFilter()
            TubeFilter.SetInputConnection(bnd_source.GetOutputPort())
            TubeFilter.SetRadius(settings.rad_bnd)
            TubeFilter.SetNumberOfSides(settings.res_bond)
            TubeFilter.CappingOn()

            # Map data, create actor and set the color
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(TubeFilter.GetOutputPort())
            bnd_actor = vtkActor()
            bnd_actor.GetProperty().SetColor(color)
            bnd_actor.GetProperty().SetOpacity(settings.alpha_at)
            bnd_actor.GetProperty().SetLighting(settings.use_light)
            bnd_actor.GetProperty().ShadingOn()
            bnd_actor.SetMapper(mapper)

            self.ren.AddActor(bnd_actor)

    def add_kabsch_bond(self, first_loc, second_loc, color1, color2, color3, settings):
        """ Makes a single bond as a combination of three segments """

        if np.allclose(color1, color2):

            self.add_bond(first_loc, second_loc, color1, settings)

        else:

            diff = (second_loc - first_loc) / 3.0

            # Add all thirds to actor list
            self.add_bond(first_loc, first_loc+diff, color1, settings)
            self.add_bond(first_loc+diff, first_loc+2*diff, color2, settings)
            self.add_bond(first_loc+2*diff, second_loc, color3, settings)

    def add_all_bonds_regular(self, molecule, settings):
        """ Wrapper for the addition of all bonds from the molecule """

        if self.plot_type == 'substructure':

            color = np.transpose(np.repeat(molecule.col_glob_rgb,
                                           molecule.n_bonds).reshape((3, molecule.n_bonds)))

        else:

            color = molecule.col_bnd_rgb

        [self.add_bond(molecule.cor[molecule.bnd_idx[bond][0]]*settings.scale_glob,
                       molecule.cor[molecule.bnd_idx[bond][1]]*settings.scale_glob,
                       color[bond], settings) for bond in range(molecule.n_bonds)]

    def add_all_bonds_disordered(self, molecule1, molecule2, settings):
        """ Wrapper for the addition of all disordered positions between the molecules """

        if settings.n_dev > 0 and molecule1.disord_pos is not None:

            color_rgb = np.asarray(settings.col_disord_rgb)  # RGB color for disordered positions

            disord_col = np.transpose(np.repeat(color_rgb, settings.n_dev).reshape((3, settings.n_dev)))

            [self.add_bond(molecule1.cor[molecule1.disord_pos[pos]]*settings.scale_glob,
                           molecule2.cor[molecule1.disord_pos[pos]]*settings.scale_glob,
                           disord_col[pos], settings) for pos in range(settings.n_dev)]

    def add_all_bonds_kabsch(self, align, settings):
        """ Wrapper for the addition of all bonds (Kabsch) from the molecule """
        
        if align.chd_bnd_col_rgb is None:  # Check if changed bonds exist at all - if they don't: use normal bonds

            [self.add_bond(align.cor[align.bnd_idx[bond][0]]*settings.scale_glob,
                           align.cor[align.bnd_idx[bond][1]]*settings.scale_glob,
                           align.col_bnd_glob_rgb, settings) for bond in range(align.n_bonds)]

        [self.add_kabsch_bond(align.cor[align.bnd_idx[bond][0]]*settings.scale_glob,
                              align.cor[align.bnd_idx[bond][1]]*settings.scale_glob,
                              align.col_bnd_rgb[bond], align.chd_bnd_col_rgb[bond], align.col_bnd_rgb[bond], settings)
         for bond in range(align.n_bonds)]

    def add_all_bonds_superpos(self, align, settings):
        """ Wrapper for the addition of all bonds for the superposition plot """

        [self.add_bond(align.cor_mol1_kbs[align.bnd_idx[bond][0]]*settings.scale_glob,
                       align.cor_mol1_kbs[align.bnd_idx[bond][1]]*settings.scale_glob,
                       align.col_bnd_mol1_rgb[bond], settings) for bond in range(align.n_bonds)]

        [self.add_bond(align.cor_mol2_kbs[align.bnd_idx[bond][0]]*settings.scale_glob,
                       align.cor_mol2_kbs[align.bnd_idx[bond][1]]*settings.scale_glob,
                       align.col_bnd_mol2_rgb[bond], settings) for bond in range(align.n_bonds)]

    def add_label(self, coords, color, label):
        """ Adds a label at the given coordinate """

        source = vtkVectorText()
        source.SetText(label)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        follower = vtkFollower()
        follower.SetMapper(mapper)
        follower.GetProperty().SetColor(color)
        follower.SetPosition(coords)
        follower.SetScale(0.4)

        self.ren.AddActor(follower)
        follower.SetCamera(self.ren.GetActiveCamera())

    def add_all_labels(self, molecule, settings):
        """ Wrapper for the addition of all labels for the molecule """

        if settings.name == 'Wireframe':

            radii = np.transpose(np.reshape(np.repeat((0.0, 0.0, 0.76), molecule.n_atoms), (3, molecule.n_atoms)))

        elif self.plot_type == 'substructure':

            radii = np.repeat(1.5, molecule.n_atoms)

        else:

            radii = np.transpose(np.vstack((np.zeros(molecule.n_atoms), np.zeros(molecule.n_atoms),
                                            molecule.rad_plt_vtk)))

        if settings.draw_labels:

            label_color = [0.0, 0.0, 0.0]

            if settings.label_type == 'full':

                labels = molecule.sym_idf

            elif settings.label_type == 'symbol_only':

                labels = molecule.sym

            [self.add_label(molecule.cor[atom]*settings.scale_glob+radii[atom]*settings.scale_at, label_color,
                            labels[atom]) for atom in range(molecule.n_atoms)]

    def add_legend(self, molecule1, molecule2, settings):
        """ Adds a legend to the VTK renderer """

        cube_source = vtkCubeSource()
        cube_source.SetBounds(-0.001,0.001,-0.001,0.001,-0.001,0.001)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(cube_source.GetOutputPort())  # connect source and mapper

        cube_actor = vtkActor()
        cube_actor.SetMapper(mapper);
        cube_actor.GetProperty().SetColor(settings.backgr_col_rgb)  # Set cube color to background

        legendBox = vtkLegendBoxActor()  # Adds the actual legend box
        legendBox.SetBackgroundColor(settings.backgr_col_rgb)  # NOT WORKING - why
        legendBox.SetBorder(1)  # No border
        legendBox.SetBox(2)
        legendBox.SetNumberOfEntries(2)

        if self.plot_type == 'initial':

            legendBox.SetEntry(0, cube_source.GetOutput(), molecule1.name, settings.col_model_rgb)
            legendBox.SetEntry(1, cube_source.GetOutput(), molecule2.name, settings.col_refer_rgb)

        elif self.plot_type == 'superpos':

            legendBox.SetEntry(0, cube_source.GetOutput(), molecule2.name1, settings.col_model_fin_rgb)
            legendBox.SetEntry(1, cube_source.GetOutput(), molecule2.name2, settings.col_refer_fin_rgb)

        pos1, pos2 = legendBox.GetPositionCoordinate(),  legendBox.GetPosition2Coordinate()
        pos1.SetCoordinateSystemToView(), pos2.SetCoordinateSystemToView()
        pos1.SetValue(.4, -1.0), pos2.SetValue(1.0, -0.75)

        self.ren.AddActor(cube_actor)
        self.ren.AddActor(legendBox)

    def add_color_bar(self, settings):
        """ Adds a color bar to the VTK scene """

        # Generate and customize lookuptable
        lut = vtkLookupTable()
        lut.SetHueRange(1/3.0, 0.0)  # From green to red
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetAlphaRange(1.0, 1.0)
        lut.SetNumberOfColors(settings.n_col_aRMSD)
        lut.SetRange(0.0, settings.max_RMSD_diff)  # Labels from 0.0 to max_RMSD
        lut.Build()  # Build the table
         
        # Create the scalar_bar
        scalar_bar = vtkScalarBarActor()
        scalar_bar.SetTitle(' ')  # Otherwise it causes a string error
        scalar_bar.GetProperty().SetColor(0.0, 0.0, 0.0)
        scalar_bar.SetLabelFormat('%-#6.2g')  # Two digits
        scalar_bar.SetNumberOfLabels(8)
        scalar_bar.SetLookupTable(lut)

        self.ren.AddActor(scalar_bar)
         
        # Create the scalar_bar_widget
        #scalar_bar_widget = vtkScalarBarWidget()
        #scalar_bar_widget.SetInteractor(self.iren)
        #scalar_bar_widget.SetScalarBarActor(scalar_bar)
        #scalar_bar_widget.On()

    def add_camera_setting(self, molecule):
        """ Adds a camera orientation from the molecule to the VTK object """

        self.has_cam_vtk = molecule.has_cam_vtk
        self.cam_vtk_pos = molecule.cam_vtk_pos
        self.cam_vtk_focal_pt = molecule.cam_vtk_focal_pt
        self.cam_vtk_view_up = molecule.cam_vtk_view_up

    def make_initial_plot(self, molecule1, molecule2, settings):
        """ Calls all functions needed for the initial plot """

        self.plot_type = 'initial'

        arrow_length = np.max(np.abs(molecule2.cor)) * 1.25 * settings.scale_glob

        self.add_all_atoms(molecule1, settings)
        self.add_all_bonds_regular(molecule1, settings)
        self.add_all_atoms(molecule2, settings)
        self.add_all_bonds_regular(molecule2, settings)
        self.add_all_bonds_disordered(molecule1, molecule2, settings)
        self.add_all_labels(molecule1, settings)
        self.add_all_labels(molecule2, settings)

        if settings.draw_arrows:  # Draw arrows

            self.add_arrow('x', arrow_length, settings)
            self.add_arrow('y', arrow_length, settings)
            self.add_arrow('z', arrow_length, settings)

        if settings.draw_labels:  # Draw arrow labels
            
            self.add_label([arrow_length, 0.0, 0.0], settings.arrow_col_rgb, 'X')
            self.add_label([0.0, arrow_length, 0.0], settings.arrow_col_rgb, 'Y')
            self.add_label([0.0, 0.0, arrow_length], settings.arrow_col_rgb, 'Z')

        if settings.draw_legend:  # Draw legend

            self.add_legend(molecule1, molecule2, settings)
            
        self.add_camera_setting(molecule2)

    def make_inertia_plot(self, molecule1, molecule2, pa_mol1, pa_mol2, settings):
        """ Calls all functions for the inertia tensor plot """

        radius = 1.3

        self.plot_type = 'inertia'

        arrow_length = np.max(np.abs(molecule1.cor)) * 1.25 * settings.scale_glob
        arrow_length2 = np.max(np.abs(molecule2.cor)) * 0.65 * settings.scale_glob

        self.add_all_atoms(molecule1, settings)
        self.add_all_bonds_regular(molecule1, settings)
        self.add_all_atoms(molecule2, settings)
        self.add_all_bonds_regular(molecule2, settings)

        self.add_com(molecule1, radius, settings.col_model_inertia_rgb, settings)
        self.add_com(molecule2, radius, settings.col_refer_inertia_rgb, settings)

        self.add_principal_axes(molecule1.com, pa_mol1[0], arrow_length2, settings.col_model_inertia_rgb, settings)
        self.add_principal_axes(molecule1.com, pa_mol1[1], arrow_length2, settings.col_model_inertia_rgb, settings)
        self.add_principal_axes(molecule1.com, pa_mol1[2], arrow_length2, settings.col_model_inertia_rgb, settings)

        self.add_principal_axes(molecule2.com, pa_mol2[0], arrow_length2, settings.col_refer_inertia_rgb, settings)
        self.add_principal_axes(molecule2.com, pa_mol2[1], arrow_length2, settings.col_refer_inertia_rgb, settings)
        self.add_principal_axes(molecule2.com, pa_mol2[2], arrow_length2, settings.col_refer_inertia_rgb, settings)

        if settings.draw_arrows:  # Draw arrows

            self.add_arrow('x', arrow_length, settings)
            self.add_arrow('y', arrow_length, settings)
            self.add_arrow('z', arrow_length, settings)

        self.add_camera_setting(molecule2)

    def make_kabsch_plot(self, align, settings):
        """ Calls all functions needed for the Kabsch plot """

        self.plot_type = 'aRMSD'

        self.add_all_atoms(align, settings)
        self.add_all_bonds_kabsch(align, settings)
        self.add_all_labels(align, settings) 
        self.add_camera_setting(align)

        if settings.use_aRMSD_col and settings.draw_col_map:  # If aRMSD colors are requested

            self.add_color_bar(settings)

        # Connect with picker
        self.picker = aRMSD_plot_picker(settings, self.at_actors_list, align)
        self.picker.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(self.picker)

    def make_substructure_plot(self, align, settings):
        """ Calls all functions needed for the substructure selection plot """

        self.plot_type = 'substructure'

        self.add_all_atoms(align, settings)
        self.add_all_bonds_regular(align, settings)
        self.add_all_labels(align, settings) 
        self.add_camera_setting(align)

        # Connect with picker
        self.picker = aRMSD_substructure_picker(settings, self.at_actors_list,
                                                align, plot_type='substructure', picker_type='normal')
        self.picker.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(self.picker)

    def make_superpos_plot(self, align, settings):
        """ Calls all functions needed for the superposition plot """

        self.plot_type = 'superpos'

        self.add_all_atoms_superpos(align, settings)
        self.add_all_bonds_superpos(align, settings)
        self.add_all_labels(align, settings)

        if settings.draw_legend:  # Draw legend

            self.add_legend(align, align, settings)

        self.add_camera_setting(align)

    def make_fractional_plot(self, xray, settings, picker_type):
        """ Calls all functions needed for the fractional coordinates plot """

        self.plot_type = 'fractional'

        self.add_all_atoms(xray, settings)
        self.add_all_bonds_regular(xray, settings)
        self.add_camera_setting(xray)

        # Connect with picker
        self.picker = aRMSD_substructure_picker(settings, self.at_actors_list, xray, self.plot_type, picker_type)
        self.picker.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(self.picker)


class Molecular_Viewer_mpl(object):
    """ A molecular viewer object based on matplotlib used for 3d plots """

    def __init__(self):
        """ Initializes the molecular viewer """

        self.space = plt.figure() # Define plotting space and axes

        self.axes = self.space.add_subplot(111)

        self.axes.grid(False) # Switch off grid
        self.axes.axis('off') # Switch off axis

        self.n_plots = 1

    def colorbar_plot(self, align, settings):
        """ Contains all functions for the Kabsch plot in aRMSD representation """

        # Set up color map, bounds for colorbar (rounded to second digit) and normalize boundary
        cmap = mpl.colors.ListedColormap(align.plt_col_aRMSD)
        spacing = 0.1

        # Adjust the colorbar spacing for small and large RMSD distributions
        if settings.max_RMSD_diff < 0.5:

            spacing = 0.05

        if settings.max_RMSD_diff >= 2.0:

            spacing = 0.2

        # 0.0 to settings.max_RMSD_diff with given spacing
        bounds = np.around(np.arange(0.0, settings.max_RMSD_diff+0.1, spacing), 2)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # Create a second axes for the colorbar
        self.axes2 = self.space.add_axes([0.88, 0.1, 0.03, 0.8]) # Global values (do not change)

        # Create colorbar
        mpl.colorbar.ColorbarBase(self.axes2, cmap=cmap, norm=norm,
                                  spacing='proportional', ticks=bounds, boundaries=bounds)

        # Set y label and label size
        self.axes2.set_ylabel(r'RMSD / $\AA$', size=12)

        self.space.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0) # Set margins of the plot window
        plt.show() # Show result
        

class Statistics_mpl(object):
    """ A statistics object based on matplotlib used for 2d plots """

    def __init__(self):
        """ Initializes the main window via gridspec and defines plotting colors """

        # Define subplot locations and set titles and grids
        self.gs = gridspec.GridSpec(3,3, left=0.08, bottom=0.08, right=0.96, top=0.92, wspace=0.30, hspace=0.97)
    
        self.ax1 = plt.subplot(self.gs[0, :-1])      
        self.ax2 = plt.subplot(self.gs[1:, :-1])
        self.ax3 = plt.subplot(self.gs[0:, -1])

    def plot(self):
        """ Plots result """

        mng = plt.get_current_fig_manager()  # Open directly in full window

        if mpl.get_backend() == 'Qt4Agg':  # 'Qt4' backend

            mng.window.showMaximized()

        elif mpl.get_backend() == 'WxAgg':  # 'WxAgg' backend

            mng.frame.Maximize(True)

        elif mpl.get_backend() == 'TKAgg':  # 'TKAgg' backend

            mng.frame.Maximize(True)

        plt.show(all)  # Show all plots

    def linregress(self, x, y):
        """ Calculate a least-squares regression for two sets of measurements (taken from scipy) """

        eps = 1.0E-20
        x, y = np.asarray(x), np.asarray(y)
            
        n = len(x)
        xmean, ymean = np.mean(x, None), np.mean(y, None)

        # average sum of squares:
        ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
        r_num = ssxym
        r_den = np.sqrt(ssxm * ssym)
        if r_den == 0.0:
            r = 0.0
        else:
            r = r_num / r_den
            # test for numerical error propagation
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0

        df = n - 2
        t = r * np.sqrt(df / ((1.0 - r + eps)*(1.0 + r + eps)))
        slope = r_num / ssxm
        intercept = ymean - slope*xmean
        sterrest = np.sqrt((1 - r**2) * ssym / ssxm / df)

        return slope, intercept, r, sterrest

    def do_stats_quant(self, align, logger, settings, prop='bond_dist'):
        """ Wrapper for the calculation and plotting of individual statistic evaluations """

        # Details for the handling of the different quantities
        if prop == 'bond_dist':
            
            data_mol1 = align.bnd_dis_mol1
            data_mol2 = align.bnd_dis_mol2
            plot_color = settings.new_red
            plt_axis = self.ax1
            title_prefix = 'All Bond Distances:'
            label_suffix = ' distances'
            label_unit = r' $\AA$'
            extra_space = 0.2

            # Do actual statistics for the two data sets
            m, b, r, sterrest = self.linregress(data_mol2, data_mol1)
            limits, x_axis, rmsd = self.prep_simulation(data_mol2, data_mol1, settings)

            logger.prop_bnd_dist_rmsd, logger.prop_bnd_dist_r_sq = rmsd, r**2  # Log quality descriptors

        elif prop == 'bond_dist_types':

            # Mean values
            data_mol1 = np.asarray([np.mean(align.bnd_dis_mol1[align.bnd_type_pos[entry]])
                                    for entry in range(align.n_bnd_types)])
            data_mol2 = np.asarray([np.mean(align.bnd_dis_mol2[align.bnd_type_pos[entry]])
                                    for entry in range(align.n_bnd_types)])

            # Calculate error
            if settings.error_prop == 'std':  # Standard deviations

                error_prop1 = np.asarray([np.std(align.bnd_dis_mol1[align.bnd_type_pos[entry]])
                                          for entry in range(align.n_bnd_types)])
                error_prop2 = np.asarray([np.std(align.bnd_dis_mol2[align.bnd_type_pos[entry]])
                                          for entry in range(align.n_bnd_types)])

            elif settings.error_prop == 'var':  # Variances

                error_prop1 = np.asarray([np.var(align.bnd_dis_mol1[align.bnd_type_pos[entry]])
                                          for entry in range(align.n_bnd_types)])
                error_prop2 = np.asarray([np.var(align.bnd_dis_mol2[align.bnd_type_pos[entry]])
                                          for entry in range(align.n_bnd_types)])

            plot_color = settings.new_red
            plt_axis = self.ax2
            title_prefix = 'Average Distances per Bond Type:'
            label_suffix = ' distance types'
            label_unit = r' $\AA$'
            extra_space = 0.1 + np.max(np.hstack((error_prop1, error_prop2)))  # Additional extra space for markers

            # Do actual statistics for the two data sets
            m, b, r, sterrest = self.linregress(data_mol2, data_mol1)
            limits, x_axis, rmsd = self.prep_simulation(data_mol2, data_mol1, settings)

            logger.prop_bnd_dist_type_rmsd, logger.prop_bnd_dist_type_r_sq = rmsd, r**2  # Log quality descriptors

            if align.n_bnd_types <= 2:

                logger.pt_warning_bond_types()  # Warn user if 2 or less bond types were found

        elif prop == 'angles':
            
            data_mol1 = align.ang_deg_mol1
            data_mol2 = align.ang_deg_mol2
            plot_color = settings.new_green
            plt_axis = self.ax3
            title_prefix = 'All Angles:'
            label_suffix = ' angles'
            label_unit = r' $^\circ$'
            extra_space = 3.5

            # Do actual statistics for the two data sets
            m, b, r, sterrest = self.linregress(data_mol2, data_mol1)
            limits, x_axis, rmsd = self.prep_simulation(data_mol2, data_mol1, settings)

            logger.prop_ang_rmsd, logger.prop_ang_r_sq = rmsd, r**2  # Log quality descriptors

        elif prop == 'torsions':
            
            data_mol1 = align.tor_deg_mol1
            data_mol2 = align.tor_deg_mol2
            plot_color = settings.new_blue
            plt_axis = self.ax3
            title_prefix = 'All Angles / Dihedrals:'
            label_suffix = ' dihedrals'
            label_unit = r' $^\circ$'
            extra_space = 3.5

            # Do actual statistics for the two data sets
            m, b, r, sterrest = self.linregress(data_mol2, data_mol1)
            limits, x_axis, rmsd = self.prep_simulation(data_mol2, data_mol1, settings)

            logger.prop_tor_rmsd, logger.prop_tor_r_sq = rmsd, r**2  # Log quality descriptors

        # Generate all titles and labels
        ax_title = title_prefix + '    RMSE = ' + str(np.around(rmsd, settings.calc_prec_stats)) + label_unit
        xlabel = align.name2+' /' + label_unit
        ylabel = align.name1+' /' + label_unit
        
        plt_axis.set_title(ax_title, fontsize=settings.title_pt)
        plt_axis.set_xlabel(xlabel, fontsize=settings.ax_pt, style='italic')
        plt_axis.set_ylabel(ylabel, fontsize=settings.ax_pt, style='italic')
        plt_axis.grid(False)
        
        label_data = str(len(data_mol2)) + label_suffix
        label_fit = r'R$^2$ = '+str(np.around(r**2, settings.calc_prec_stats))

        log_rmsd, log_r_sq = rmsd, r**2  # Log quality of correlation

        # Plot linear correlation and fit / adjust axes limits
        if prop == 'bond_dist_types':

            plt_axis.errorbar(data_mol2, data_mol1, xerr=error_prop2, yerr=error_prop1, fmt="o",
                              ms=8.5, mfc=plot_color, mew=0.75, zorder=2, mec=plot_color, label=label_data)

            [plt_axis.text(data_mol2[pos], data_mol1[pos] - 0.1,
                           align.bnd_label[pos], zorder=3, fontsize=13) for pos in range(align.n_bnd_types)]

            add_lim = np.asarray([-0.1, 0.1], dtype=np.float)

            limits += add_lim

        else:

            plt_axis.plot(data_mol2, data_mol1, "o", ms=8.5, mfc=plot_color, mew=0.75,
                          zorder=1, mec=plot_color, label=label_data)

        plt_axis.plot(x_axis, m*x_axis+b, lw=2, zorder=1, color=plot_color, label=label_fit)
            
        plt_axis.set_xlim([limits[0] - extra_space, limits[1] + extra_space])
        plt_axis.set_ylim([limits[0] - extra_space, limits[1] + extra_space])

        # Draw legend and add grid upon request
        if settings.stats_draw_legend:
            
            plt_axis.legend(loc=settings.legend_pos, frameon=False)

        if settings.stats_draw_grid:
            
            plt_axis.grid()

    def prep_simulation(self, data1, data2, settings):
        """ Calculates the RMSE of two data sets and generates axis for linear regression """

        # Determine lowest and highest values of the combined data
        stack = np.hstack((data1, data2))
        limits = [np.min(stack), np.max(stack)]

        # Calculate RMSD of the data sets
        rmsd = np.around(np.sqrt(np.sum(np.abs(data2 - data1)**2 / len(data2))), 4)

        # Determine step size and axis
        step_size_all = ((limits[1] + settings.splitter) - (limits[0] - settings.splitter)) / len(data2)
        axis = np.arange(limits[0] - settings.splitter, limits[1] + settings.splitter, step_size_all)

        return limits, axis, rmsd
