"""
aRMSD main routines
(c) 2017 by Arne Wagner
"""

# Authors: Arne Wagner
# License: MIT

# IMPORTING ALL DEPENDENCIES HERE FIXES POTENTIAL PYINSTALLER ISSUES
from __future__ import absolute_import
from builtins import range, input

import acore as ac
import aplot as ap
import alog as al

import numpy as np

try:  # Will be checked in core module - this is just for the correct PyInstaller hooks

    import uncertainties.unumpy as unp
    from uncertainties import ufloat, ufloat_fromstr

except ImportError:

    pass

try:

    import pybel
    import openbabel

except ImportError:

    pass

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release

name = 'aRMSD'
author = 'Arne Wagner'
author_email = 'arne.wagner@aci.uni-heidelberg.de'
url = 'https://github.com/armsd/aRMSD'
doc = 'http://armsd.rtfd.io'
lic = 'MIT'

__aRMSD_version__ = '0.9.4'
__aRMSD_release__ = 2017

__log_version__ = '2017-04-05'
__core_version__ = '2017-01-03'
__plot_version__ = '2016-11-03'

__author__ = author+' <'+author_email+'>'

short_description = """
aRMSD
==================================
aRMSD is a Python module/program that allows for a fast and comprehensive
analysis of molecular structures parsed from different files. Unique features
are a support of crystallographic data, error propagation and specific types
of graphical representations. """

long_description = """
For a full description of the capabilities and usage of aRMSD see the online
documentation under """+str(doc)+""". To use the program execute aRMSD.py or
open a new shell and write

from aRMSD import run

run()

This fill start the program and print the welcome screen along all needed
instructions. Load two molecular structures from files and work your way
through the program """


# HERE START THE MAIN PROGRAM FUNCTIONS (Menues and main program)
###############################################################################
# HANDLERS FOR USER INPUT
###############################################################################


def export_structure(molecule1, molecule2, logger, settings):
    """ Handles the export of structural data to outfiles """

    def get_proj_rad(r1, r2, default):
        """ Returns the radii in correct order """

        if np.around(r1) == np.around(r2):

            return default

        else:

            return [r2, r1] if np.around(r1) > np.around(r2) else [r1, r2]

    choices = [0, 1, 2, -1, -10]  # List of accepted choices

    question = "\n>>  Enter your choice: "

    # Make copies of vtk radii for export
    rad_plt_vtk_exp1, rad_plt_vtk_exp2 = np.copy(molecule1.rad_plt_vtk), np.copy(molecule2.rad_plt_vtk)

    min_rad, max_rad = np.min(np.hstack((molecule1.rad_cov, molecule2.rad_cov))), \
                       np.max(np.hstack((molecule1.rad_cov, molecule2.rad_cov)))

    df_rad = [min_rad, max_rad]

    while True:

        logger.pt_export_structure_menu(min_rad, max_rad)
        operation = logger.get_menu_choice(choices, question, return_type='int')

        if operation == -10:  # Exit the menu

            break

        elif operation == -1:  # Project radii

            min_rad = logger.get_menu_choice([0.0, 10.0],
                                             "\n>>  Enter a value for the minimum radius (between 0.0 and 10.0): ",
                                             return_type='float')
            max_rad = logger.get_menu_choice([0.0, 10.0],
                                             "\n>>  Enter a value for the maximum radius (between 0.0 and 10.0): ",
                                             return_type='float')

            min_rad, max_rad = get_proj_rad(min_rad, max_rad, df_rad)

            rad_plt_vtk_exp1 = ac.project_radii(molecule1.rad_plt_vtk, 1000, min_rad, max_rad)
            rad_plt_vtk_exp2 = ac.project_radii(molecule2.rad_plt_vtk, 1000, min_rad, max_rad)

        elif operation in [0, 1]:

            molecule1.get_export_file(logger, example='myfile.xyz or myfile.xyzs', prefix='first')
            molecule2.get_export_file(logger, example='myfile.xyz or myfile.xyzs', prefix='second')

            if operation == 0:  # Export as two simple .xyz file

                ac.write_xyz_file(logger, molecule1.outfile_name, molecule1.sym, molecule1.cor)
                ac.write_xyz_file(logger, molecule2.outfile_name, molecule2.sym, molecule2.cor)

            elif operation == 1:  # Export as two extended aRMSD .xyzs file

                pos_dis, col_dis_rgb, pos_inter, col_inter_rgb = None, None, None, None

                if settings.name == 'Wireframe':  # Wireframe plot style: change radii

                    rad_plt_vtk_exp1, rad_plt_vtk_exp2 = np.repeat(0.76, molecule1.n_atoms), \
                                                         np.repeat(0.76, molecule2.n_atoms)

                # Scale radii
                rad_plt_vtk_exp1 *= settings.scale_at
                rad_plt_vtk_exp2 *= settings.scale_at

                ac.write_xyzs_file(logger, molecule1.outfile_name, molecule1.sym, molecule1.cor, molecule1.cor_std,
                                   molecule1.col_at_rgb, rad_plt_vtk_exp1, molecule1.bnd_idx, molecule1.col_bnd_rgb,
                                   np.repeat(settings.rad_bnd, molecule1.n_bonds), pos_dis, col_dis_rgb, pos_inter,
                                   col_inter_rgb, molecule1.cam_vtk_pos, molecule1.cam_vtk_wxyz, settings.scale_glob)

                ac.write_xyzs_file(logger, molecule2.outfile_name, molecule2.sym, molecule2.cor, molecule2.cor_std,
                                   molecule2.col_at_rgb, rad_plt_vtk_exp2, molecule2.bnd_idx, molecule2.col_bnd_rgb,
                                   np.repeat(settings.rad_bnd, molecule2.n_bonds), pos_dis, col_dis_rgb, pos_inter,
                                   col_inter_rgb, molecule2.cam_vtk_pos, molecule2.cam_vtk_wxyz, settings.scale_glob)

        elif operation == 2:  # Export a combined file

            pos_inter, col_inter_rgb = None, None

            molecule1.get_export_file(logger, example='myfile.xyzs', prefix=None)
            pos_dis, col_dis_rgb = None, None

            if molecule1.disord_pos is not None:  # Combine disordered positions

                pos_dis = np.transpose(np.vstack((molecule1.disord_pos, molecule1.disord_pos+molecule1.n_atoms)))
                col_dis_rgb = molecule1.col_disord_rgb

            if settings.name == 'Wireframe':  # Wireframe plot style: change radii

                rad_plt_vtk_exp1, rad_plt_vtk_exp2 = np.repeat(0.76, molecule1.n_atoms), \
                                                     np.repeat(0.76, molecule2.n_atoms)

            # Scale radii
            rad_plt_vtk_exp1 *= settings.scale_at
            rad_plt_vtk_exp2 *= settings.scale_at

            ac.write_xyzs_file(logger, molecule1.outfile_name, np.hstack((molecule1.sym, molecule2.sym)),
                               np.vstack((molecule1.cor, molecule2.cor)),
                               np.vstack((molecule1.cor_std, molecule2.cor_std)),
                               np.vstack((molecule1.col_at_rgb, molecule2.col_at_rgb)),
                               np.hstack((rad_plt_vtk_exp1, rad_plt_vtk_exp2)),
                               np.vstack((molecule1.bnd_idx, molecule2.bnd_idx+molecule1.n_atoms)),
                               np.vstack((molecule1.col_bnd_rgb, molecule2.col_bnd_rgb)),
                               np.repeat(settings.rad_bnd, molecule1.n_bonds+molecule2.n_bonds),
                               pos_dis, col_dis_rgb, pos_inter, col_inter_rgb,
                               molecule2.cam_vtk_pos, molecule2.cam_vtk_wxyz, settings.scale_glob)

        else:

            logger.pt_invalid_input()

###############################################################################
# MAIN MENUES
###############################################################################


def kabsch_menu(molecule1, molecule2, logger, settings):
    """ All functions related to RMSD calculations """

    def calc_bond_stats(align, settings):
        """ Calculates the bond and dihedral angles """

        align.get_all_angles(settings)
        align.get_all_torsions(settings)
        align.check_for_nan(settings)
        align.has_stats = True

    def add_or_remove_bond(align, logger):
        """ Lets the user add or remove a single bond """

        pos1 = logger.get_menu_choice(range(1, align.n_atoms + 1),
                                      "\nInput the identifier (starts at 1) of the first bond: ", return_type='int')
        pos2 = logger.get_menu_choice(range(1, align.n_atoms + 1),
                                      "\nInput the identifier (starts at 1) of the second bond: ", return_type='int')

        idx1, idx2 = pos1 - 1, pos2 - 1

        proper1 = 0 <= idx1 <= align.n_atoms
        proper2 = 0 <= idx2 <= align.n_atoms

        if proper1 and proper2:  # Indices are in order

            if not align.bond_exists(idx1, idx2)[0]:  # Add a bond, inform user

                align.add_bond(idx1, idx2)
                logger.pt_bond_added(align, idx1, idx2)

            elif align.bond_exists(idx1, idx2)[0]:  # Remove a bond, inform user

                align.remove_bond(idx1, idx2)
                logger.pt_bond_removed(align, idx1, idx2)

        else:  # Wrong indices, inform user

            logger.pt_wrong_indices(align)

    choices = [0, 1, 2, 3, 4, 5, 19, 20, 21, 24, -1, -2, -3, -4, -5, -6, -7, -8, -10]  # List of accepted choices

    question = "\n>>  Enter your choice: "

    align = ac.Kabsch(molecule1, molecule2, settings)  # Create 'Kabsch object'
    align.init_coords(molecule1, molecule2)
    align.get_symbols()
    align.set_colors(settings)  # Set initial colors
    align.get_weights(logger)  # Get initial weights

    logger.d_min, logger.d_max = settings.gard_d_min, settings.gard_d_max

    substructure = np.arange(molecule2.n_atoms)

    recalc_stats = True
    update_armsd_rep = False

    while True:

        logger.pt_kabsch_menu(align, settings)

        operation = logger.get_menu_choice(choices, question, return_type='int')

        if operation == -10:  # Exit aRMSD

            logger.pt_program_termination()
            return align

        elif operation == -8:

            if align.has_kabsch:

                viewmol_mpl = ap.Molecular_Viewer_mpl()
                logger.pt_plotting()  # Inform user of the new plot
                viewmol_mpl.colorbar_plot(align, settings)

            else:

                logger.pt_kabsch_first()

        elif operation == -7:  # Change settings of aRMSD representation

            settings.change_rmsd_settings(logger)
            align.clean_aRMSD_rep()
            update_armsd_rep = True

        elif operation == -6:  # Add or remove bonds (move to substructure selector)

            add_or_remove_bond(align, logger)
            align.update_bonds()
            align.set_colors(settings)
            recalc_stats, update_armsd_rep = True, True

        elif operation == -4:  # Change VTK settings

            settings.change_rmsd_vtk_plot_settings(align, logger)
            update_armsd_rep = True

        elif operation == -3:  # Define substructures

            settings.use_ball_and_stick()
            viewmol_vtk = ap.Molecular_Viewer_vtk(settings)
            viewmol_vtk.make_substructure_plot(align, settings)
            logger.pt_plotting()  # Inform user of the new plot
            logger.pt_plotting_substructure_def()
            pos_sub1 = viewmol_vtk.show(align, align, settings)

            mask = np.in1d(np.arange(align.n_atoms), pos_sub1, invert=True)  # Get substructure 2 from 'pos sub1' array
            pos_sub2 = np.arange(align.n_atoms)[mask]

            align.check_substructure(logger, pos_sub1, pos_sub2)  # Check if a proper substructure has been defined

        elif operation == -2:  # Change weights

            align.get_w_function(logger)
            align.get_weights(logger)
            align.has_kabsch = False

        elif operation == -1:  # Perform Kabsch algorithm

            align.kabsch_algorithm(logger)  # Calculate Kabsch superposition and similarity descriptors
            align.calc_z_matrix_rmsd()  # Calculate z-matrix RMSD and decomposition
            align.update_bonds()  # Update bonds before the generation of the aRMSD representation
            align.get_aRMSD_rep(settings, project_rad=True)
            update_armsd_rep = False
            logger.pt_rmsd_results(align)  # Show RMSD results

        elif operation in [0, 1, 2, 3]:

            if align.has_kabsch:

                if operation == 0:  # Show results in aRMSD representation

                    if update_armsd_rep:

                        align.get_aRMSD_rep(settings, project_rad=True)
                        update_armsd_rep = False

                    settings.use_ball_and_stick()
                    viewmol_vtk = ap.Molecular_Viewer_vtk(settings)
                    viewmol_vtk.make_kabsch_plot(align, settings)
                    logger.pt_plotting()  # Inform user of the new plot
                    logger.pt_plotting_screenshot()
                    viewmol_vtk.show(align, align, settings)

                elif operation == 1:  # Show superposition

                    settings.use_wireframe()
                    viewmol_vtk = ap.Molecular_Viewer_vtk(settings)
                    viewmol_vtk.make_superpos_plot(align, settings)
                    logger.pt_plotting()  # Inform user of the new plot
                    logger.pt_plotting_screenshot()
                    viewmol_vtk.show(align, align, settings)

                elif operation == 2:  # Molecular statistics

                    if recalc_stats:

                        # Get angles and torsions
                        calc_bond_stats(align, settings)
                        recalc_stats = False

                    if logger.has_mpl:

                        statplot = ap.Statistics_mpl()
                        statplot.do_stats_quant(align, logger, settings, prop='bond_dist')
                        statplot.do_stats_quant(align, logger, settings, prop='bond_dist_types')
                        statplot.do_stats_quant(align, logger, settings, prop='angles')
                        statplot.do_stats_quant(align, logger, settings, prop='torsions')
                        logger.pt_plotting()  # Inform user of the new plot
                        statplot.plot()

                        logger.pt_max_dev_internal(align, settings)  # Show internal coordinates with highest deviation

                    else:

                        logger.pt_no_mpl()

                elif operation == 3:  # Show RMSD results

                    logger.pt_rmsd_results(align)

            else:

                logger.pt_kabsch_first()

        elif operation == 4:  # Interpolate structures

            align.interpolate_structures(settings, substructure)
            align.write_interpol_structures(logger, settings)

        elif operation == 5:  # Write outfile

            if not align.has_kabsch:

                logger.pt_kabsch_first()

            elif not align.has_sub_rmsd and align.has_sub:

                logger.pt_kabsch_first()

            else:

                logger.write_logfile(align, settings)

        elif operation == 20:  # Export structures

            if not align.has_kabsch:

                logger.pt_kabsch_first()

            else:

                align.export_kabsch(logger, settings)
                
        else:

            logger.pt_invalid_input()


def symmetry_and_matching(molecule1, molecule2, logger, settings):
    """ Performs symmetry transformations on molecule 1 and matches the two data sets """

    def set_n_dev(molecule2, logger, settings):
        """ Gets a number of deviations to be shown from user """

        n_dev = int(eval(input("\nInput the number of deviations to be highlighted: ")))

        if n_dev < 1 or n_dev > molecule2.n_atoms:  # do nothing

            logger.pt_wrong_n_dev()

        else:

            settings.n_dev = n_dev  # Update n_dev

    def get_rot_n(logger):
        """ Gets a proper rotation from user """

        n_rot = int(eval(input("\nInput n for n-fold clockwise rotation: ")))

        if n_rot == 0:  # Set n_rot to 1 if it is 0 by accident

            n_rot = 1
            logger.pt_wrong_n()

        # Return value(s)
        return n_rot

    choices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 21, 24, -1, -2, -3, -4, -5, -6]  # List of accepted choices

    # Dictionary of symmetry number and transformation correlations 
    sym_number = {1: 'inv', 2: 'xy', 3: 'xz', 4: 'yz', 5: 'x', 6: 'y', 7: 'z'}
    sym_transf = {'xy': ac.z_axis, 'xz': ac.y_axis, 'yz': ac.x_axis, 'x': ac.x_axis, 'y': ac.y_axis, 'z': ac.z_axis}

    question = "\n>>  Enter your choice: "

    # Move molecules in standard orientation and print into to user
    molecule1.standard_orientation(logger, calc_for='molecule')
    molecule2.standard_orientation(logger, calc_for='molecule')
    logger.pt_standard_orientation()

    # Save the current status
    molecule1.make_save_point(reset_type='save')
    molecule2.make_save_point(reset_type='save')
    logger.pt_saved_status()

    while True:

        logger.pt_match_menu(settings)
        
        if molecule1.show_mol:

            viewmol_vtk = ap.Molecular_Viewer_vtk(settings)
            viewmol_vtk.make_initial_plot(molecule1, molecule2, settings)
            logger.pt_plotting()  # Inform user of the new plot
            logger.pt_plotting_screenshot()
            viewmol_vtk.show(molecule1, molecule2, settings)
            molecule1.show_mol = False
            
        symmetry_operation = logger.get_menu_choice(choices, question, return_type='int')

        if symmetry_operation == -6:  # Set number of deviations and update information

            set_n_dev(molecule2, logger, settings)

            if logger.is_matched:

                ac.highlight_dev(molecule1, molecule2, logger, settings)

        elif symmetry_operation == -5:  # Reset the changes made by the matching algorithm

            logger.pt_match_reset(reset_type='save')
            molecule1.reset_molecule(settings, reset_type='save')
            molecule2.reset_molecule(settings, reset_type='save')

            molecule1.disord_pos, molecule2.disord_pos = None, None
            molecule1.disord_rmsd, molecule2.disord_rmsd = None, None

            # Update properties and bonds
            molecule1.update_properties(settings, get_bonds=True)
            molecule2.update_properties(settings, get_bonds=True)

        elif symmetry_operation == -4:  # Change VTK plot settings

            settings.change_vtk_plot_settings(molecule1, molecule2, logger)

        elif symmetry_operation == -3:  # Swap atoms in molecule 1

            molecule1.swap_atoms(logger)

            # Update properties and bonds
            molecule1.update_properties(settings, get_bonds=True)
            molecule2.update_properties(settings, get_bonds=True)

            if logger.is_matched:

                ac.highlight_dev(molecule1, molecule2, logger, settings)

        elif symmetry_operation == -2:  # Change matching defaults

            molecule2.change_algorithms(logger)

        elif symmetry_operation == -1:  # Match molecules

            ac.match_molecules(molecule1, molecule2, logger)

            # Update properties and bonds
            molecule1.update_properties(settings, get_bonds=True)
            molecule2.update_properties(settings, get_bonds=True)

            # Highlight deviations
            ac.highlight_dev(molecule1, molecule2, logger, settings)
            logger.pt_highest_deviations(molecule1, molecule2, settings)

        elif symmetry_operation == 0:

            if logger.is_matched:

                molecule1.calc_total_symOP()  # Calculate total symmetry operation
                logger.pt_exit_sym_menu()
                break

            else:

                logger.pt_no_match()
                exit_yn = logger.get_menu_choice(['y', 'Y', 'n', 'N'], question, return_type='str')

                if exit_yn == 'y' or exit_yn == 'Y':

                    molecule1.calc_total_symOP()  # Calculate total symmetry operation
                    break

        elif symmetry_operation == 1:

            logger.pt_sym_inversion()
            molecule1.inversion()  # inversion

        elif symmetry_operation in [2, 3, 4]:  # Reflections

            logger.pt_sym_reflection(sym_number[symmetry_operation])
            molecule1.sigma_refl(sym_transf[sym_number[symmetry_operation]])

        elif symmetry_operation in [5, 6, 7]:  # Rotations

            logger.pt_rot_info()
            n_rot = get_rot_n(logger)
            logger.pt_sym_rotation(n_rot, sym_number[symmetry_operation])
            molecule1.cn_rotation(sym_transf[sym_number[symmetry_operation]], n_rot)

        elif symmetry_operation == 8:

            molecule1.show_mol = True

        elif symmetry_operation == 10:  # Save current changes

            molecule1.make_save_point(reset_type='save'), molecule2.make_save_point(reset_type='save')
            logger.pt_saved_status()

        elif symmetry_operation == 20:  # Export structures

            export_structure(molecule1, molecule2, logger, settings)

        else:

            logger.pt_invalid_input()


###############################################################################
# HERE STARTS THE ACTUAL PROGRAM
###############################################################################

def run():
    """ Function to run the application """

    align = None

    # Read in settings from config file or use defaults
    settings = ac.settings()
    settings.parse_settings('settings.cfg')

    # Set up logger and print welcome screen
    logger = al.Logger(__aRMSD_version__, __aRMSD_release__)
    logger.check_modules(ac.has_np, ac.np_version, ap.has_vtk, ap.vtk_version,
                         ap.has_mpl, ap.mpl_version, ac.has_pybel, ac.pyb_version,
                         ac.has_uc, ac.uc_version)

    logger.pt_welcome()
    logger.pt_versions(__core_version__, __plot_version__, __log_version__)
    logger.pt_modules()

    if logger.has_requirements:

        logger.pt_start()

        # Import molecules
        molecule1, file_type = ac.parse_files(logger, 1, settings)
        molecule1.get_charge()
        molecule2, file_type = ac.parse_files(logger, 2, settings)
        molecule2.get_charge()
        logger.lg_files_loaded()

        # Check for standard deviations
        logger.chk_std_devs(molecule1, molecule2, settings)

        # Set plotting colors of the molecules, defaults are red and green
        molecule1.set_color(settings.col_model_rgb), molecule2.set_color(settings.col_refer_rgb)
        molecule1.set_color_disordered(settings.col_disord_rgb), molecule2.set_color_disordered(settings.col_disord_rgb)

        # Get properties, check for units
        molecule1.update_properties(settings, get_bonds=True), molecule2.update_properties(settings, get_bonds=True)
        molecule1.check_for_unit(logger, settings), molecule2.check_for_unit(logger, settings)

        # Save the current molecules in case something goes wrong during consistency attempt
        molecule1.calc_com(calc_for='molecule'), molecule2.calc_com(calc_for='molecule')
        molecule1.shift_com(calc_for='molecule'), molecule2.shift_com(calc_for='molecule')
        molecule1.make_save_point(reset_type='origin'), molecule2.make_save_point(reset_type='origin')
        logger.pt_saved_status()

        # Perform consistency check, update properties and shift center of mass to origin
        logger.pt_consistency_old()
        ac.make_consistent(molecule1, molecule2, logger, settings)
        molecule1.update_properties(settings, get_bonds=True), molecule2.update_properties(settings, get_bonds=True)
        molecule1.calc_com(calc_for='molecule'), molecule2.calc_com(calc_for='molecule')
        molecule1.shift_com(calc_for='molecule'), molecule2.shift_com(calc_for='molecule')

        if logger.consist:

            # Go in the symmetry and matching menu
            symmetry_and_matching(molecule1, molecule2, logger, settings)

            # Go into Kabsch menu
            align = kabsch_menu(molecule1, molecule2, logger, settings)

            return align

        else:

            logger.pt_program_termination()

            return align

    else:

        logger.pt_requirement_error()

        return align

if __name__ == '__main__':  # Run the program

    align = run()
