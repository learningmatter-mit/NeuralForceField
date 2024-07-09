"""
aRMSD log functions
(c) 2017 by Arne Wagner
"""

# Authors: Arne Wagner
# License: MIT

from __future__ import absolute_import, division, print_function
from builtins import range, input

try:

    import numpy as np

except ImportError:

    pass


class Logger(object):
    """ An object used for logging / plotting messages on screen """

    def __init__(self, __version__, year):
        """ Initializes messenger """

        self.version = __version__  # Program version
        self.year = year  # Year
        self.file_name = 'aRMSD_logfile.out'  # Name of the designated outfile
        self.file_mol1 = None  # File name of molecule 1
        self.file_mol2 = None  # File name of molecule 2
        self.name_mol1 = 'Model'  # Name of molecule 1
        self.name_mol2 = 'Reference'  # Name of molecule 2
        self.has_requirements = False  # If requirements are met
        self.has_cor_std_mol1 = False  # If molecule coordinates have standard deviations (molecule 1)
        self.has_cor_std_mol2 = False  # If molecule coordinates have standard deviations (molecule 2)
        self.use_std = True  # If standard deviations are to be used
        self.disorder_mol1 = False  # If atoms on identical positions were found (molecule 1)
        self.disorder_mol2 = False  # If atoms on identical positions were found (molecule 2)
        self.file_import = False  # If data import was successful
        self.chk_for_unit = False  # If units of the coordinates were checked
        self.chk_for_unit_warn = False  # If number of atoms is sufficient for unit check
        self.unit_transform = False  # If coordinate units were transformed (b2a)
        self.has_sub = False  # If a substructure has been defined
        self.cons_init_at_mol1 = None  # Initial Number of atoms in molecule 1
        self.cons_init_at_mol2 = None  # Initial Number of atoms in molecule 2
        self.cons_init_at_H_mol1 = None  # Initial Number of H-atoms in molecule 1
        self.cons_init_at_H_mol2 = None  # Initial Number of H-atoms in molecule 2
        self.rem_H_btc = False  # If hydrogens bound to carbons have been removed
        self.rem_H_btg14 = False  # If hydrogens bound to group-14 atoms have been removed
        self.rem_H_all = False  # If all hydrogen atoms have been removed
        self.can_rem_H_btc = True  # If hydrogens bound to carbons can be removed
        self.can_rem_H_btg14 = True  # If hydrogens bound to group-14 atoms can be removed
        self.user_choice_rem_all_H = False  # User choice to remove all H atoms
        self.user_choice_rem_btc_H = False  # User choice to remove all H atoms
        self.user_choice_rem_btg14_H = False  # User choice to remove all H atoms
        self.n_atoms = None  # Final number of atoms after consistency
        self.rot_to_std = None  # Rotation matrix for standard orientation
        self.use_groups = False  # If PSE groups are to be used in matching algorithm
        self.consist = False  # If the two molecular strucutres are consistent
        self.match_alg = 'distance'  # Matching algorithm
        self.match_solv = 'hungarian'  # Solver used in matching process
        self.is_matched = False  # If the molecules were matched
        self.was_saved = False  # If a status has been saved (checkpoint)
        self.xfs_energy = None  # Energy of X-ray source (None is unused)
        self.prop_bnd_dist_rmsd = None
        self.prop_bnd_dist_r_sq = None
        self.prop_bnd_dist_type_rmsd = None
        self.prop_bnd_dist_type_r_sq = None
        self.prop_ang_rmsd = None
        self.prop_ang_r_sq = None
        self.prop_tor_rmsd = None
        self.prop_tor_r_sq = None

        self.d_min = None  # d_min value for GARD calculation
        self.d_max = None  # d_max value for GARD calculation

        self.has_np = False  # If numpy is available
        self.np_version = None  # numpy version
        self.has_vtk = False  # If VTK is available
        self.vtk_version = None  # VTK version
        self.has_mpl = False  # If matplotlib is available
        self.mpl_version = None  # matplotlib version
        self.has_pybel = False  # If openbabel/pybel is available
        self.py_version = None  # openbabel version
        self.has_uc = False  # If uncertainties is available
        self.uc_version = None  # uncertainties version

        self.max_string_len = 60  # Maximum character length per line

    def get_numpy(self, has_np, np_version):

        self.has_np = has_np  # If numpy is available
        self.np_version = np_version  # numpy version

    def get_vtk(self, has_vtk, vtk_version):

        self.has_vtk = has_vtk  # If VTK is available
        self.vtk_version = vtk_version  # VTK version

    def get_mpl(self, has_mpl, mpl_version):

        self.has_mpl = has_mpl  # If matplotlib is available
        self.mpl_version = mpl_version  # matplotlib version

    def get_pybel(self, has_pybel, pyb_version):

        self.has_pybel = has_pybel  # If openbabel/pybel is available
        self.pyb_version = pyb_version  # openbabel/pybel version

    def get_uncertainties(self, has_uc, uc_version):

        self.has_uc = has_uc  # If uncertainties is available
        self.uc_version = uc_version  # uncertainties version

    def pt_no_mpl(self):

        print("\n> Matplotlib appears to be missing - but is required for 2D plots!")

    def check_modules(self, has_np, np_version, has_vtk, vtk_version,
                      has_mpl, mpl_version, has_pybel, pyb_version,
                      has_uc, uc_version):

        self.get_numpy(has_np, np_version)
        self.get_vtk(has_vtk, vtk_version)
        self.get_mpl(has_mpl, mpl_version)
        self.get_pybel(has_pybel, pyb_version)
        self.get_uncertainties(has_uc, uc_version)

        self.has_requirements = self.has_np and self.has_vtk

    def pt_modules(self):

        print("\nModule check:")
        print("- numpy         \t'" + str(self.np_version) + "'")
        print("- VTK           \t'" + str(self.vtk_version) + "'")
        print("- matplotlib    \t'" + str(self.mpl_version) + "'")
        print("- uncertainties \t'" + str(self.uc_version) + "'")
        print("- openbabel     \t'" + str(self.pyb_version) + "'")

    def format_value(self, value, n_digits):

        str_len = 12 if n_digits != 0 else 5

        ft_str_norm = '{:3.2f}'

        if n_digits != 0:
            
            ft_str_norm = '{:' + str(n_digits) + '.' + str(n_digits) + 'f}'
            ft_str_unce = '{:.1uS}'  # One digit for values with uncertainties

        if self.use_std:  # If standard deviations exist

            if value.std_dev == 0.0 or n_digits == 0:  # Different format for values without standard deviations

                add = str_len - len(ft_str_norm.format(value.nominal_value))

                if n_digits == 0 and value.nominal_value < 10.0:

                    return '0' + ft_str_norm.format(value.nominal_value) + ' ' * (add - 1)

                else:

                    return ft_str_norm.format(value.nominal_value) + ' ' * add

            else:

                add = str_len - len(ft_str_unce.format(value))

                return ft_str_unce.format(value) + ' ' * add

        elif n_digits == 0 and value < 10.0:

            add = str_len - len(ft_str_norm.format(value))

            return '0' + ft_str_norm.format(value) + ' ' * (add - 1)

        else:  # No ufloat values

            return ft_str_norm.format(value)

    def format_sym_idf(self, sym_idf1, sym_idf2):

        return ' ' * (6 - len(sym_idf1)) + sym_idf1 + ' -- ' + sym_idf2 + ' ' * (6 - len(sym_idf2))

    ###############################################################################
    # WRITE OUTFILE
    ###############################################################################


    def write_logfile(self, align, settings):

        def adj_str(string, prefix='\n\t', suffix='\t'):

            delta = self.max_string_len - len(string)

            return prefix + string + ' ' * delta + suffix

        def wt_general_info():

            output.write('===================================================================================================')
            output.write('\n                 aRMSD - automatic RMSD Calculator: Version ' +
                         str(self.version))
            output.write('\n===================================================================================================')
            output.write('\n                  A. Wagner, University of Heidelberg (' + str(self.year) + ')')
            output.write('\n\n\tA brief description of the program can be found in the manual and in:')
            output.write('\n\tA. Wagner, PhD thesis, University of Heidelberg, 2015.\n')
            output.write('\n---------------------------------------------------------------------------------------------------\n')
            output.write('\n*** Cite this program as:' +
                         '\n    A. Wagner, H.-J. Himmel, J. Chem. Inf. Model, 2017, 57, 428-438.')
            output.write('\n\n---------------------------------------------------------------------------------------------------')
            output.write(adj_str('*** Log file of the superposition between the structures ***', prefix='\n\n', suffix='\n'))
            output.write(adj_str('"' + str(self.name_mol1) + '"...', prefix='\n\t', suffix='\t') + str(self.file_mol1))
            output.write(adj_str('"' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') + str(self.file_mol2))

        def wt_consistency():

            output.write(adj_str('* Consistency establishment between the structures:', prefix='\n\n', suffix='\n'))
            output.write(adj_str('# The basic approach is to subsequently remove hydrogen atoms until', prefix='\n\t', suffix=''))
            output.write(adj_str('# the same number of atoms is found in both molecules', prefix='\n\t', suffix=''))
            output.write(adj_str('# If the number of atoms is identical and the atom types belong', prefix='\n\t', suffix=''))
            output.write(adj_str('# to the same group in the periodic table, the molecules are', prefix='\n\t', suffix=''))
            output.write(adj_str('# regarded as consistent', prefix='\n\t', suffix=''))

            if self.disorder_mol1:
                
                output.write(adj_str('   - Initial disorder was found in "', prefix='\n\n', suffix='') + str(self.name_mol2) + '"')
                output.write(adj_str('Disorder was resolved by the user...', prefix='\n\t', suffix=''))

            if self.disorder_mol2:
                
                output.write(adj_str('   - Initial disorder was found in "', prefix='\n\t', suffix='') + str(self.name_mol2) + '"')
                output.write(adj_str('Disorder was resolved by the user...', prefix='\n\t', suffix=''))

            if not self.disorder_mol1 and not self.disorder_mol2:

                output.write(adj_str('   - No disorder in the structures was found', prefix='\n\n', suffix='\n'))

            output.write(adj_str('Initial number of atoms in "' + str(self.name_mol1) + '"...', prefix='\n\t', suffix='\t') +
                         str(self.cons_init_at_mol1) + '\t(' + str(self.cons_init_at_H_mol1) + ' H atoms)')

            output.write(adj_str('Initial number of atoms in "' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') +
                         str(self.cons_init_at_mol2) + '\t(' + str(self.cons_init_at_H_mol2) + ' H atoms)')

            if self.rem_H_btc:

                output.write(adj_str('H atoms bound to carbon were removed...', prefix='\n\t', suffix=''))

            if self.rem_H_btg14:

                output.write(adj_str('H atoms bound to group-14 elements were removed...', prefix='\n\t', suffix=''))

            if self.rem_H_all:

                output.write(adj_str('All H atoms were removed...', prefix='\n\t', suffix=''))

            output.write(adj_str('   - Consistency between the structures was established', prefix='\n\n', suffix=''))

            output.write(adj_str('The number of atoms in "' + str(self.name_mol1) + '"...', prefix='\n\n\t', suffix='\t') +
                         str(self.cons_at_mol1) + '\t(' + str(self.cons_at_H_mol1) + ' H atoms)')
            output.write(adj_str('The number of atoms in "' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') +
                         str(self.cons_at_mol2) + '\t(' + str(self.cons_at_H_mol2) + ' H atoms)')

            if not self.user_choice_rem_all_H and not self.user_choice_rem_btc_H and not self.user_choice_rem_btg14_H:

                output.write(adj_str('   - No further modifications were done', prefix='\n\n', suffix=''))

            if self.user_choice_rem_all_H:

                output.write(adj_str('   - All hydrogen atoms were removed by the user', prefix='\n\n', suffix=''))

            elif self.user_choice_rem_btc_H:

                output.write(adj_str('   - All hydrogen atoms bound to carbon were removed by the user', prefix='\n\n', suffix=''))

            elif self.user_choice_rem_btg14_H:

                output.write(adj_str('   - All hydrogen atoms bound to group-14 elements were removed by the user', prefix='\n\n', suffix=''))

            output.write(adj_str('Final number of atoms...', prefix='\n\n\t', suffix='\t') + str(align.n_atoms) +
                         '\t(' + str(align.n_hydro) + ' H atoms)')

        def wt_std_orientation():

            output.write(adj_str('* Transformation of the molecules into "Standard Orientation":', prefix='\n\n', suffix='\n'))
            output.write(adj_str('# 1. The center of mass was shifted to the Cartesian origin', prefix='\n\t', suffix=''))
            output.write(adj_str('# 2. The moment of inertia tensor was constructed, diagonalized', prefix='\n\t', suffix=''))
            output.write(adj_str('#    and the eigenvectors rotated on the x, y and z axes', prefix='\n\t', suffix=''))

        def wt_match():

            output.write(adj_str('* Details of the matching process:', prefix='\n\n', suffix='\n'))
            output.write(adj_str('Structures were matched...', prefix='\n\t', suffix='\t') + str(self.is_matched))

            if self.is_matched:

                output.write(adj_str('Applied matching algorithm...', prefix='\n\t', suffix='\t') + str(self.match_alg))
                output.write(adj_str('Solver used for matching...', prefix='\n\t', suffix='\t') + str(self.match_solv))

                if self.use_groups:

                    output.write(adj_str('Solution of the matching problem...', prefix='\n\t', suffix='\t') + 'PSE groups')

                else:

                    output.write(adj_str('Solution of the matching problem...', prefix='\n\t', suffix='\t') + 'regular')

                output.write(adj_str('Number of highest deviations to be shown...', prefix='\n\t', suffix='\t') + str(self.n_dev))
                output.write(adj_str('The highest deviations were between the pairs...', prefix='\n\n\t', suffix='\t') + '[Angstrom]\n')
                
                [output.write('\n\t\t\t' + self.format_sym_idf(align.sym_idf_mol1[self.disord_pos[entry]], align.sym_idf_mol2[self.disord_pos[entry]]) + '\t\t\t\t\t\t\t' +
                              '{:6.5f}'.format(self.disord_rmsd[entry])) for entry in range(self.n_dev)]
                
                output.write(adj_str('The RMSD after matching was...', prefix='\n\n\t', suffix='\t') +
                             '{:6.5f}'.format(self.match_rmsd) + ' [Angstrom]')

        def wt_kabsch():

            # Contribution of individual atom types
            rmsd_perc = (align.rmsd_idv ** 2 / np.sum(align.rmsd_idv ** 2)) * 100

            output.write(adj_str('* Kabsch alignment:', prefix='\n\n', suffix='\n'))
            output.write(adj_str('# General settings', prefix='\n\t', suffix='\n'))
            output.write(adj_str('Substructures were defined...', prefix='\n\t', suffix='\t') + str(align.has_sub_rmsd))
            output.write(adj_str('Weighting function for Kabsch algorithm...', prefix='\n\t', suffix='\t') + str(align.wts_type))
            output.write(adj_str('Consideration of multi-center-contributions...', prefix='\n\t', suffix='\t') + str(align.calc_mcc))
            output.write(adj_str('# Differentiation criteria and color information', prefix='\n\n\t', suffix='\n'))
            output.write(adj_str('Number of colors for aRMSD plot...', prefix='\n\t', suffix='\t') + str(settings.n_col_aRMSD))
            output.write(adj_str('Maximum RMSD value for color projection...', prefix='\n\t', suffix='\t') + str(settings.max_RMSD_diff) + '  [Angstrom]')
            output.write(adj_str('Threshold for bond comparison...', prefix='\n\t', suffix='\t') + str(settings.thresh) + '  [Angstrom]')
            output.write(adj_str('Number of distance pairs above threshold...', prefix='\n\t', suffix='\t') + str(align.n_chd_bnd) + '  [Angstrom]')
            output.write(adj_str('Percentage of the colored intersections...', prefix='\n\t', suffix='\t') + str((1.0 - 2 * settings.n) * 100) + '  [%]')
            
            output.write(adj_str('Color for shorter bonds in "' + str(self.name_mol1) + '" wrt "' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') +
                         str(settings.col_short_hex) + '  [HEX]')
            output.write(adj_str('Color for longer bonds in "' + str(self.name_mol2) + '" wrt "' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') +
                         str(settings.col_long_hex) + '  [HEX]')
            output.write(adj_str('Number of bonds below threshold...', prefix='\n\t', suffix='\t') + str(align.n_chd_bnd))
            
            output.write(adj_str('Color of "' + str(self.name_mol1) + '"...', prefix='\n\t', suffix='\t') + str(settings.col_model_fin_hex) + '  [HEX]')
            output.write(adj_str('Color of "' + str(self.name_mol2) + '"...', prefix='\n\t', suffix='\t') + str(settings.col_refer_fin_hex) + '  [HEX]')
            
            output.write(adj_str('Final rotation matrix from "Standard Orientation"...', prefix='\n\n\t', suffix='\n'))

            output.write('\n\t           |' + '{:+06.8f}'.format(align.tot_rot_mat[0][0]) + '  ' +
                         '{:+06.8f}'.format(align.tot_rot_mat[0][1]) +
                         '  ' + '{:+06.8f}'.format(align.tot_rot_mat[0][2]) + '|')
            output.write('\n\t     U  =  |' + '{:+06.8f}'.format(align.tot_rot_mat[1][0]) + '  ' +
                         '{:+06.8f}'.format(align.tot_rot_mat[1][1]) +
                         '  ' + '{:+06.8f}'.format(align.tot_rot_mat[1][2]) + '|')
            output.write('\n\t           |' + '{:+06.8f}'.format(align.tot_rot_mat[2][0]) + '  ' +
                         '{:+06.8f}'.format(align.tot_rot_mat[2][1]) +
                         '  ' + '{:+06.8f}'.format(align.tot_rot_mat[2][2]) + '|')

            output.write(adj_str('# This matrix aligns "' + str(self.name_mol1) + '" with "' + str(self.name_mol2) + '"', prefix='\n\n\t', suffix=''))
            output.write(adj_str('# U already includes all custom symmetry operations!', prefix='\n\t', suffix=''))

            output.write(adj_str('* Quality of the Superposition:', prefix='\n\n', suffix='\n'))
            output.write(adj_str('d values for the GARD calculation...', prefix='\n\t', suffix='\t') + str(self.d_min) + ', ' + str(self.d_max))
            output.write(adj_str('Superposition R^2...', prefix='\n\t', suffix='\t') + self.format_value(align.r_sq, n_digits=5) +
                         '  [Dimensionless]')
            output.write(adj_str('Cosine similarity...', prefix='\n\t', suffix='\t') + self.format_value(align.cos_sim, n_digits=5) +
                         '  [Dimensionless]')
            output.write(adj_str('GARD score...', prefix='\n\t', suffix='\t') + self.format_value(align.gard, n_digits=5) +
                         '  [Dimensionless]')
            output.write(adj_str('RMSD...', prefix='\n\t', suffix='\t') + self.format_value(align.rmsd, n_digits=5) +
                         '  [Angstrom]')
            output.write(adj_str('   - Decomposition into different atom types', prefix='\n\n', suffix='\t\t') + 'Absolute [Angstrom] \tRelative [%]\n')
            
            [output.write('\n\t\t\t' + "{:4.4s}".format(align.at_types[entry]) + ' (#' + "{:3.0f}".format(align.occ[entry]) + ')\t\t\t\t\t\t\t\t' +
                          self.format_value(align.rmsd_idv[entry], n_digits=5) + '  \t\t\t(' +
                          self.format_value(rmsd_perc[entry], n_digits=0) + ')')
             for entry in range(align.n_atom_types)]

            output.write(adj_str('   - z-matrix properties', prefix='\n\n', suffix='\n'))
            output.write(adj_str('# z-matrices are created for both molecules using', prefix='\n\t', suffix=''))
            output.write(adj_str('# 3 N - 1 bond distances, 3 N - 2 bond angles and', prefix='\n\t', suffix=''))
            output.write(adj_str('# 3 N - 3 dihedral angles and the total RMSD and', prefix='\n\t', suffix=''))
            output.write(adj_str('# the relative contributions are calculated', prefix='\n\t', suffix='\n'))

            output.write(adj_str('RMSD...', prefix='\n\t', suffix='\t') + self.format_value(align.rmsd_z_matrix, n_digits=5) +
                         '  [Angstrom]')
            output.write(adj_str('Contribution of distances...', prefix='\n\t', suffix='\t') + self.format_value(align.c_dis * 100, n_digits=0) + '  [%]')
            output.write(adj_str('Contribution of angles...', prefix='\n\t', suffix='\t') + self.format_value(align.c_ang * 100, n_digits=0) + '  [%]')
            output.write(adj_str('Contribution of dihedral angles...', prefix='\n\t', suffix='\t') + self.format_value(align.c_tor * 100, n_digits=0) + '  [%]')

            if align.has_sub_rmsd:
                
                form1 = align.make_sum_formula(pos=align.pos_sub1)
                form2 = align.make_sum_formula(pos=align.pos_sub2)

                output.write(adj_str('   - Decomposition into different substructures', prefix='\n\n', suffix='\n'))

                output.write(adj_str('Substructure 1', prefix='\n\t', suffix='\t') + '[' + str(form1) + ']' + '  (' + str(len(align.pos_sub1)) + ' atoms)')
                
                output.write(adj_str('Superposition R^2...', prefix='\n\t', suffix='\t') + self.format_value(align.r_sq_sub1, n_digits=5) + '  [Dimensionless]')
                output.write(adj_str('Cosine similarity...', prefix='\n\t', suffix='\t') + self.format_value(align.cos_sim_sub1, n_digits=5) + '  [Dimensionless]')
                output.write(adj_str('RMSD...', prefix='\n\t', suffix='\t') + self.format_value(align.rmsd_sub1, n_digits=5) + '  [Angstrom]')
                output.write(adj_str('Contribution...', prefix='\n\t', suffix='\t') + self.format_value(align.c_sub1 * 100, n_digits=0) + '  [%]')

                output.write(adj_str('Substructure 2', prefix='\n\n\t', suffix='\t') + '[' + str(form2) + ']' + '  (' + str(len(align.pos_sub2)) + ' atoms)')
                
                output.write(adj_str('Superposition R^2...', prefix='\n\t', suffix='\t') + self.format_value(align.r_sq_sub2, n_digits=5) + '  [Dimensionless]')
                output.write(adj_str('Cosine similarity...', prefix='\n\t', suffix='\t') + self.format_value(align.cos_sim_sub2, n_digits=5) + '  [Dimensionless]')
                output.write(adj_str('RMSD...', prefix='\n\t', suffix='\t') + self.format_value(align.rmsd_sub2, n_digits=5) + '  [Angstrom]')
                output.write(adj_str('Contribution...', prefix='\n\t', suffix='\t') + self.format_value(align.c_sub2 * 100, n_digits=0) + '  [%]')


        def wt_prop(prop):

            return '{:6.5f}'.format(prop) if prop is not None else str(prop)

        def wt_struct():

            if align.has_stats:

                output.write(adj_str('* Evaluation of structural parameters:', prefix='\n\n', suffix='\n'))
                output.write(adj_str('# 1. The RMSE values are the root-mean-square errors', prefix='\n\t', suffix=''))
                output.write(adj_str('#    between the corresponding properties of the two structures', prefix='\n\t', suffix=''))
                output.write(adj_str('# 2. The R2 values are the the correlation coefficients', prefix='\n\t', suffix=''))
                output.write(adj_str('#    between the two data sets', prefix='\n\t', suffix=''))

                output.write(adj_str('Number of bonds...', prefix='\n\n\t', suffix='\t') + str(align.n_bonds))
                output.write(adj_str('R2 of linear correlation...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_bnd_dist_r_sq) + '  [Dimensionless]')
                output.write(adj_str('RMSE...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_bnd_dist_rmsd) + '  [Angstrom]')

                output.write(adj_str('Number of bond types...', prefix='\n\n\t', suffix='\t') + str(align.n_bnd_types))
                output.write(adj_str('R2 of linear correlation...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_bnd_dist_r_sq) + '  [Dimensionless]')
                output.write(adj_str('RMSE...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_bnd_dist_type_rmsd) + '  [Angstrom]')

                output.write(adj_str('Number of angles...', prefix='\n\n\t', suffix='\t') + str(align.n_angles))
                output.write(adj_str('R2 of linear correlation...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_ang_r_sq) + '  [Dimensionless]')
                output.write(adj_str('RMSE...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_ang_rmsd) + '  [Degrees]')

                output.write(adj_str('Number of dihedrals...', prefix='\n\n\t', suffix='\t') + str(align.n_torsions))
                output.write(adj_str('R2 of linear correlation...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_tor_r_sq) + '  [Dimensionless]')
                output.write(adj_str('RMSE...', prefix='\n\t', suffix='\t') + wt_prop(self.prop_tor_rmsd) + '  [Degrees]')

        def wt_eof():

            output.write(adj_str('*** End of log file ***', prefix='\n\n', suffix=''))

        output = open(self.file_name, 'w')  # Create a new file

        # Write all information aspects to file
        wt_general_info()
        wt_consistency()
        wt_std_orientation()
        wt_match()
        wt_kabsch()
        wt_struct()
        wt_eof()

        output.close()  # Close the outfile

        self.pt_logfile_written()  # Inform the user

    def pt_logfile_written(self):

        print("\n> A logfile (" + str(self.file_name) + ") has been written successfully!")

    ###############################################################################
    # HANDLERS FOR USER INPUT
    ###############################################################################

    def get_menu_choice(self, choices, question, return_type='int'):
        """ Handles user input in menu routine, returns a valid user choice based on 'return_type' """

        while True:  # Stay in loop until a valid user choice

            try:

                choice = eval(input(question))

                if return_type == 'int':

                    choice = int(choice)

                elif return_type == 'float':

                    choice = float(choice)

            except (NameError, SyntaxError, ValueError):

                # If input can not be converted: Raise error and set operation to ""
                choice = ""

            # If return type is HTML evaluate if user input is a valid HTML color
            if return_type == 'HTML':

                if self.html_check(choice):

                    break

                else:

                    self.pt_invalid_input()

            # If return type is float evaluate if float is in range of choices
            elif return_type == 'float':

                if choice >= min(choices) and choice <= max(choices):

                    break

                else:

                    self.pt_invalid_input()

            elif return_type == 'symOP':

                char_list = ['x', 'y', 'z', ' ', '.', ',', '/', '*', '-', '+',
                             '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

                symbol_ok = 'x' in choice and 'y' in choice and 'z' in choice
                comma_ok = choice.count(',') == 2
                char_ok = False not in [choice[pos] in char_list for pos in range(len(choice))]

                if False not in [symbol_ok, comma_ok, char_ok]:

                    break

                else:

                    self.pt_invalid_input()

            # Check if operation is a valid choice and exit loop if True
            elif choice in choices:

                if return_type == 'bool' and choice == 0:

                    choice = False

                elif return_type == 'bool' and choice == 1:

                    choice = True

                break

            # Otherwise pass and stay in loop
            else:

                self.pt_invalid_input()

        # Return value(s)
        return choice

    def html_check(self, color_string):
        """ Checks if string is a correct HTML color """

        def if_entry_in_dict(entry):

            html_dict = ['F', 'f', 'E', 'e', 'D', 'd', 'B', 'b', 'A',
                         'a', 'C', 'c', '0', '1', '2', '3', '4', '5',
                         '6', '7', '8', '9']

            return entry in html_dict

        if type(color_string) != str or len(color_string) != 7 or color_string[0] != '#':

            return False

        else:

            string_in_list = [if_entry_in_dict(color_string[entry]) for entry in range(1, len(color_string))]

            return False not in string_in_list

    def change_TF(self, dictionary, key):
        """ Changes True/false variables of a given key in a dictionary """

        if dictionary[key]:

            dictionary[key] = False

        else:

            dictionary[key] = True

        return dictionary

    ###############################################################################
    # START
    ###############################################################################

    def pt_welcome(self):

        print("\n\n=============================================================================")
        print("                            aRMSD: Version " + str(self.version))
        print("=============================================================================")
        print("                  A. Wagner, University of Heidelberg (" + str(self.year) + ")")
        print("-------------------------------- Description --------------------------------")
        print("Key features:")
        print("* Parses data from various file formats")
        print("* Establishes consistency and matches coordinate sequences of two molecules")
        print("* Aligns two molecular structures based on the Kabsch algorithm")
        print("* Supports different weighting functions for the superposition")
        print("* Supports error propagation for experimental structures")
        print("* Generates different visualization types of the superposition results")
        print("* Writes outfiles that can be passed to other programs")
        print("* ... more features and changes can be found in the documentation")
        print("  ... this project is hosted on GitHub: https://github.com/armsd/aRMSD")
        print("  ... documentation: http://armsd.rtfd.io")
        print("-----------------------------------------------------------------------------")
        print(
            '\n*** Cite this program as:' + 
            '\n    A. Wagner, H.-J. Himmel, J. Chem. Inf. Model, 2017, 57, 428-438.')

    def pt_versions(self, core_version, plot_version, log_version):

        print("\nRelease dates of the individual modules:")
        print("core module:    \t'" + str(core_version) + "'")
        print("plot module:    \t'" + str(plot_version) + "'")
        print("log  module:    \t'" + str(log_version) + "'")

    def pt_start(self):

        print("\n> Starting program ...")
        print("-----------------------------------------------------------------------------")

    ###############################################################################
    # FILE IMPORT
    ###############################################################################

    def pt_file_not_found(self):

        print("\n> File not found, please try again!")

    def pt_import_success(self, input_file, element_symbol):

        print("\n> '" + str(input_file) + "' has been loaded successfully! (#Atoms: " + str(len(element_symbol)) + ")")

    def lg_files_loaded(self):

        print("-----------------------------------------------------------------------------")
        print("... Files have been loaded!")

        self.file_import = True  # Log the successful import

    def pt_no_pybel(self):

        print("\n> ERROR: Openbabel is required and missing!")

    def pt_no_pybel_file(self):

        print("\n> ERROR: Openbabel does not recognize the file type, try a different file!")

    def chk_std_devs(self, molecule1, molecule2, settings):

        print("\n-----------------------------------------------------------------------------")
        print("> Checking for coordinate standard deviations...")

        self.has_cor_std_mol1 = np.sum(molecule1.cor_std) != 0.0  # Checks deviations for both molecules
        self.has_cor_std_mol2 = np.sum(molecule2.cor_std) != 0.0

        self.use_std = True in [self.has_cor_std_mol1, self.has_cor_std_mol2]  # If any standard deviations exist
        settings.use_std = self.use_std  # Copy information to settings

        if self.use_std:

            print("... Standard deviations were found and will be used!")

        else:

            print("... No standard deviations were found!")

        print("-----------------------------------------------------------------------------")

    ###############################################################################
    # Plotting
    ###############################################################################


    def pt_plotting(self):

        print("\n> Results are now shown ... close the pop-up window to continue!")

    def pt_plotting_screenshot(self):

        print("> Press the 's' button to save the scene as .png file or 'h' for help.\n")

    def pt_plotting_substructure_def(self):

        print("\n-----------------------------------------------------------------------------")
        print("========================== Substructure Definition  =========================")
        print("-----------------------------------------------------------------------------")
        print("\n> - Click on atoms to add or remove them from the designated substructure")
        print("\n> You have to select at least two atoms, but keep in mind that substructures")
        print("> with few atoms are not very meaningful. All unselected atoms will")
        print("> constitute the second substructure.\n")
        print("-----------------------------------------------------------------------------")

    def pt_plotting_deleted(self, n_del):

        if n_del > 0:
            print("\n> A total of " + str(n_del) + " atoms have been deleted!")

    def pt_aRMSD_plot_info(self):

        print("\n> - A click on one atom will show its RMSD contribution,")
        print(">   2/3 or 4 selected atoms will display information about the")
        print(">   respective distances, angles and dihedrals.\n")

    def pt_warning_bond_types(self):

        print("\n> WARNING: Only 2 bond types were found - consequently R2 is meaningless!")

    ###############################################################################
    # CONSISTENCY MESSAGES
    ###############################################################################

    def pt_consistency_old(self):

        print("\n-----------------------------------------------------------------------------")
        print("=============== Consistency Checks and Structural Modification ==============")
        print("-----------------------------------------------------------------------------")

    def pt_consistency_menu(self):

        print("\n-----------------------------------------------------------------------------")
        print("=============== Consistency Checks and Structural Modification ==============")
        print("-----------------------------------------------------------------------------")
        print("-10 Reset molecules to the original status")
        print("-5  Load the saved status (save point available: '" + str(self.was_saved) + "')")
        print("-2  Reset substructure")
        print("-1  Establish consistency based on current (sub)structures")
        print("-----------------------------------------------------------------------------")
        print("    A substructure has been defined       : '" + str(self.has_sub) + "'")
        print("    Consistency has been established      : '" + str(self.consist) + "'")
        print("    Group matching algorithm will be used : '" + str(self.use_groups) + "'")
        print("-----------------------------------------------------------------------------")
        print("0   ... exit the menu (point of no return)")
        print("1   ... show information about the two data sets")
        print("2   ... define substructures (!next release!)")
        print("3   ... remove selected atoms")
        print("8   ... show the molecules again")
        print("-----------------------------------------------------------------------------")
        print("10  ... save current changes")
        print("20  ... render the combined scene with VTK")
        print("21  ... export the scene/structures, change VTK settings")
        print("-----------------------------------------------------------------------------")

    def pt_diff_at_number(self):

        print("\n> Different number of atoms in the two structures:")

    def pt_possibilities(self, n_hydro, n_hydro_c, n_hydro_full):

        choices = []

        print("\n-----------------------------------------------------------------------------")
        print("    What should happen to the remaining " + str(n_hydro) + " H-atoms?")
        print("-----------------------------------------------------------------------------")
        print("    Info: The exclusion of H-atoms in RMSD calculations is recommended if")
        print("          they were not located and refined in the X-ray experiment")
        print("-----------------------------------------------------------------------------")

        if not self.rem_H_all:

            print("0   ... remove all hydrogen atoms (" + str(n_hydro) + ")")
            choices.append(0)

        if not self.rem_H_btc and self.can_rem_H_btc and n_hydro != n_hydro_c:

            print("1   ... remove all hydrogen atoms bound to carbon (" + str(n_hydro_c) + ")")
            choices.append(1)

        if not self.rem_H_btg14 and self.can_rem_H_btg14:

            print("2   ... remove all hydrogen atoms bound to group-14 elements (" + str(n_hydro_full) + ")")
            choices.append(2)

        print("3   ... keep all hydrogen atoms")
        print("-----------------------------------------------------------------------------")

        choices.append(3)

        return choices

    def pt_consistency_start(self, n_atoms1, n_atoms2):

        print("\n> Performing consistency checks ... (Number of atoms: " + str(n_atoms1) + ", " + str(n_atoms2) + ")")

    def lg_multiple_occupation_found(self, logg_for='molecule1'):

        print("\n> WARNING: Atoms on identical positions were found!")
        print("           Please carefully check your input files!")

        if logg_for == 'molecule1':

            self.disorder_mol1 = True
            name = self.name_mol1

        else:

            self.disorder_mol2 = True
            name = self.name_mol2

        print("           Disordered positions in molecule: '" + str(name) + "'\n")

    def pt_info_multiple_occupation(self, sym, idf, xyz, entry):

        print("-----------------------------------------------------------------------------")
        print("Entry\tSym-Idf\t\t\t[xyz]")
        [print("Pos:\t" + str(sym[idx]) + "-" + str(idf[idx]) + "\t\t" + str(xyz[idx])) for idx in entry]
        print("-----------------------------------------------------------------------------")

    def pt_number_h_atoms(self, n_hydro_mol1, n_hydro_mol2):

        print("  There are (" + str(n_hydro_mol1) + " & " + str(n_hydro_mol2) + ") H atoms in the molecules")

    def lg_rem_H_btc(self, n_hydro_mol1, n_hydro_mol2):

        self.rem_H_btc = True
        print("\n  Removing all hydrogen atoms bound to carbon ...")
        print("  All respective hydrogen atoms (" + str(n_hydro_mol1) + " & " + str(
            n_hydro_mol2) + ") have been removed!")

    def lg_rem_H_btg14(self, n_hydro_mol1, n_hydro_mol2):

        self.rem_H_btg14 = True
        print("\n  Removing all hydrogen atoms bound to group-14 elements ...")
        print("  All respective hydrogen atoms (" + str(n_hydro_mol1) + " & " + str(
            n_hydro_mol2) + ") have been removed!")

    def lg_rem_H_all(self, n_hydro_mol1, n_hydro_mol2):

        self.rem_H_all = True
        print("\n  Removing all hydrogen atoms ...")
        print("  All hydrogen atoms (" + str(n_hydro_mol1) + " & " + str(n_hydro_mol2) + ") have been removed!")

    def lg_group_algorithm(self):

        self.use_groups = True
        print("\n> The matching problem will be solved for PSE groups.")

    def lg_consistent(self):

        self.consist = True
        print("\n> The structures of both molecules are consistent.")

    def lg_substructure(self):

        self.has_sub = True
        print("\n> A proper substructure has been defined and will be used ...")

    def lg_wrong_substructure(self):

        self.has_sub = False
        print("\n> ERROR: Number of atoms must be identical in both substructures and >= 3")

    def lg_reset_substructure(self):

        self.has_sub = False
        print("\n> Reset substructures to the full coordinate sets.")

    def pt_consistency_failure(self, n_atoms1, n_atoms2):

        print("\n> ERROR: Number of atoms (" + str(n_atoms1) + " & " + str(n_atoms2) +
              ") is not identical - check your input files!")
        self.pt_exiting()

    def pt_consistency_error(self):

        print("\n> ERROR: Severe problem encountered - check your input files!")
        self.pt_exiting()

    def pt_no_consistency_done(self):

        print("\n> ERROR: Data sets need to be checked for consistency!")

    def pt_saved_status(self):

        self.was_saved = True
        print("\n> The current status was saved successfully!")

    def pt_loaded_status(self):

        self.was_saved = True
        print("\n> The last saved status was loaded successfully!")

    def pt_write_success(self, output_file):

        print("\n> Coordinates were written to '" + str(output_file) + "' !")

    def pt_interpol_write_success(self, align):

        print("\n> Interpolated coordinates were written to 'interp_'... outfiles!")
        print("\n    The reference structure (RMSD = 0.0) corresponds to file number 0")
        print("\n          The model structure corresponds to file number " +
              str(len(align.interp_rmsd) - 1) + "\n")
        
        [print('\tFile number: ' + str(entry).zfill(2) + ' \tRMSD...\t\t' +
               self.format_value(align.interp_rmsd[entry], n_digits=5)
               + '  [Angstrom]') for entry in range(len(align.interp_rmsd))]

    def pt_unsupported_atom(self, symbol):

        print("\n> ERROR: Unsupported atomic symbol ('" + str(symbol) + "') found!")
        print("         Check your files for dummy atoms, etc.!")

    def pt_unsupported_element(self, symbol):

        print("\n> ERROR: Unknown element symbol ('" + str(symbol) + "') encountered!")

    # COORDINATE UNITS
    # ------------------------------------------------------------------------------

    def lg_unit_check(self):

        self.chk_for_unit = True
        print("\n  Checking length unit of xyz coordinates ...")

    def lg_unit_check_warning(self):

        self.chk_for_unit_warn = True
        print("\n> WARNING: Number of atoms insufficient!\nAssuming [Angstrom] as unit ...")

    def pt_unit_is_a(self):

        print("  The coordinate unit is [Angstrom]")

    def lg_transform_coord_unit(self):

        self.unit_transform = True
        print("  Coordinates were transformed from [Bohr] to [Angstrom], updating bonds ...")

    # MISC
    # ------------------------------------------------------------------------------

    def pt_exiting(self):

        print("         Exiting program ...")

    def pt_render_first(self):

        print("\n> ERROR: You need to render the scene with VTK first!")

    def pt_kabsch_first(self):

        print("\n> ERROR: You need to perform the Kabsch algorithm first!")

    def pt_future_implementation(self):

        print("\n> ERROR: Will be implemented in the future!")

    def pt_swap_n_pairs(self):

        question = "\n>>  How many atom pairs are to be swapped? "

        return question

    def pt_swap_atoms(self, sym1, idf1, sym2, idf2):

        print("\n> Swapping atom " + str(sym1) + str(idf1) + " with " + str(sym2) + str(idf2))

    def pt_program_termination(self):

        print("\n-----------------------------------------------------------------------------")
        print("========================= Normal program termination ========================")
        print("-----------------------------------------------------------------------------")

    def pt_requirement_error(self):

        print("\n> ERROR: Requirements are not met, exiting program!")

    # FILE HANDLING
    # ------------------------------------------------------------------------------

    def pt_unsupported_file_type(self, filetypes):

        print("\n> ERROR: Unsupported file type - supported types are:")
        print("         "+', '.join(filetypes))

    def pt_no_ob_support(self):

        print("\n> ERROR: File type is not supported by openbabel!")

    def inp_file_export(self, example='myfile.xyz', prefix=None):

        if prefix is None:

            pass

        else:

            print("\nEnter the " + str(prefix) + " filename:")

        return input("\n> Enter a filename (e.g. " + str(example) + "): ")

    def wrt_comment(self, comment=None):

        if comment is None:

            return "# Created with aRMSD V. " + str(self.version) + "\n"

        else:

            return "# Created with aRMSD V. " + str(self.version) + str(comment) + "\n"

    # MATCHING AND SYMMETRY OPERATIONS
    # ------------------------------------------------------------------------------


    def pt_sym_expand(self, symOP):

        print("\n> Expanding coordinates by symmetry operation '" + str(symOP) + "'")

    def pt_standard_orientation(self):

        print("\n> The molecules were rotated into 'Standard Orientation' ...")

    def pt_match_reset(self, reset_type='save'):

        if reset_type == 'save':

            print("\n> Matching process will be reseted ...")

        elif reset_type == 'origin':

            print("\n> Consistency process will be reseted ...")

    def pt_sym_inversion(self):

        print("\n> Inversion of '" + str(self.name_mol1) + "' structure at the origin of the coordinate system ...")

    def pt_sym_reflection(self, plane):

        print("\n> Reflection of '" + str(self.name_mol1) + "' structure at the " + str(plane) + "-plane ...")

    def pt_rot_info(self):

        print("\nUse '-n' for counter-clockwise rotations\n(e.g. -20 for a rotation of -360/20 = -18 deg.)")

    def pt_sym_rotation(self, n, axis):

        if n < 0:

            print("\nA " + str(abs(n)) + "-fold ccw rotation (" + str(round(360.0 / n, 1)) + " deg.) around the " +
                  str(axis) + "-axis was requested.")

        else:

            print("\nA " + str(n) + "-fold cw rotation (" + str(round(360.0 / n, 1)) + " deg.) around the " +
                  str(axis) + "-axis was requested.")

        print("\n> Applying rotation to '" + str(self.name_mol1) + "' structure ...")

    def pt_wrong_n(self):

        print("\n> ERROR: n has been set to 1 (it can't be 0)!")

    def pt_wrong_n_dev(self):

        print("\n> ERROR: The number of deviations must be 1 < n_dev < n_atoms!")

    def pt_invalid_input(self):

        print("\n> ERROR: Input invalid or out of range, try again!")

    def pt_exit_sym_menu(self):

        print("\n> Exiting symmetry transformation menu ...")

    def pt_no_match(self):

        print("\nThe molecules were not matched, exit anyway? ('y' / 'n')")

    def pt_highest_deviations(self, molecule1, molecule2, settings):                

        print("\n-----------------------------------------------------------------------------\n")
        print("\tThe geometric RMSD of the current alignment is: " + "{:6.3f}".format(self.match_rmsd) + " A\n")
        print("\t\t The " + str(settings.n_dev) + " most disordered atom pairs are:")
        print("\t\tEntry\t\t      Pair\t\t  Distance / A ")

        [print("\t\t  " + str(settings.n_dev - entry) + "\t\t" + self.format_sym_idf(molecule1.sym_idf[molecule1.disord_pos[entry]],
                                                                                     molecule2.sym_idf[molecule2.disord_pos[entry]]) +
               "\t   " + "{:6.3f}".format(molecule2.disord_rmsd[entry])) for entry in range(settings.n_dev)]

    def pt_all_bonds(self, align):

        print("\n-----------------------------------------------------------------------------")

    def pt_no_proper_rmsd_sub(self):

        print("\n> ERROR: Proper substructures (with at least two atoms) were not defined!")

    def pt_max_dev_internal(self, align, settings):

        desc_dis, dis_mol1, dis_mol2, delta_dis = align.get_max_diff_prop(settings, prop='distance')
        desc_ang, ang_mol1, ang_mol2, delta_ang = align.get_max_diff_prop(settings, prop='angle')
        desc_tor, tor_mol1, tor_mol2, delta_tor = align.get_max_diff_prop(settings, prop='torsion')

        print("\n--------------- The Highest Deviations in Internal Coordinates --------------")
        print("The " + str(settings.n_max_diff) + " highest deviations are printed below")
        print("Entries are: Atoms, values in the Model and Reference, difference")
        print("-----------------------------------------------------------------------------")
        print(" >> Bonds (in Angstrom):")
        [print("    " + str(entry + 1) + ".  " + str(desc_dis[entry]) + " " + str(dis_mol1[entry]) + " " + str(
            dis_mol2[entry]) +
               "\tDiff. " + str(delta_dis[entry])) for entry in range(settings.n_max_diff)]
        print("-----------------------------------------------------------------------------")
        print(" >> Bond angles (in deg.):")
        [print("    " + str(entry + 1) + ".  " + str(desc_ang[entry]) + " " + str(ang_mol1[entry]) + " " + str(
            ang_mol2[entry]) +
               "\tDiff. " + str(delta_ang[entry])) for entry in range(settings.n_max_diff)]
        print("-----------------------------------------------------------------------------")
        print(" >> Dihedral angles (in deg.):")
        [print("    " + str(entry + 1) + ".  " + str(desc_tor[entry]) + " " + str(tor_mol1[entry]) + " " + str(
            tor_mol2[entry]) +
               "\tDiff. " + str(delta_tor[entry])) for entry in range(settings.n_max_diff)]
        print("-----------------------------------------------------------------------------")

    def pt_rmsd_results(self, align):

        # Contribution of individual atom types (based on MSD)
        rmsd_perc = (align.rmsd_idv ** 2 / np.sum(align.rmsd_idv ** 2)) * 100

        print("\n-----------------------------------------------------------------------------")
        print("====================== Quality of the Superposition =========================")
        print("-----------------------------------------------------------------------------")
        print("\n> The type of weighting function is: '" + str(align.wts_type) + "'")
        print("\n-------------------------- Similarity Descriptors ---------------------------")
        print("   >>>   Superposition R^2 :  " + self.format_value(align.r_sq, n_digits=5))
        print("   >>>   Cosine similarity :  " + self.format_value(align.cos_sim, n_digits=5))
        print("   >>>   GARD score        :  " + self.format_value(align.gard, n_digits=5))
        print("\n------------------------ Root-Mean-Square-Deviation -------------------------")
        print("   >>>   RMSD              :  " + self.format_value(align.rmsd, n_digits=5) + " Angstrom")
        print("\n   >>>   - Decomposition   :  Individual atom types  (total percentage)")
        [print("           " + "{:4.4s}".format(align.at_types[entry]) + " (#" +
               "{:3.0f}".format(align.occ[entry]) + ")     :  " +
               self.format_value(align.rmsd_idv[entry], n_digits=5) + " Angstrom  (" +
               self.format_value(rmsd_perc[entry], n_digits=0) + " %)") for entry in range(align.n_atom_types)]

        if align.has_sub_rmsd:

            form1, form2 = align.make_sum_formula(pos=align.pos_sub1), align.make_sum_formula(pos=align.pos_sub2)

            print("\n   >>>   - Decomposition   :  Substructure properties")
            print("-----------------------------------------------------------------------------")
            print("           Substructure 1  :  # Atoms: " + str(len(align.pos_sub1)) + "    [" + str(form1) + "]")
            print("                     RMSD  :  " + self.format_value(align.rmsd_sub1, n_digits=5) + " Angstrom  (" +
                  self.format_value(align.c_sub1 * 100.0, n_digits=0) + " %)")
            print("        Superposition R^2  :  " + self.format_value(align.r_sq_sub1, n_digits=5))
            print("        Cosine similarity  :  " + self.format_value(align.cos_sim_sub1, n_digits=5))
            print("-----------------------------------------------------------------------------")
            print("           Substructure 2  :  # Atoms: " + str(len(align.pos_sub2)) + "    [" + str(form2) + "]")
            print("                     RMSD  :  " + self.format_value(align.rmsd_sub2, n_digits=5) + " Angstrom  (" +
                  self.format_value(align.c_sub2 * 100.0, n_digits=0) + " %)")
            print("        Superposition R^2  :  " + self.format_value(align.r_sq_sub2, n_digits=5))
            print("        Cosine similarity  :  " + self.format_value(align.cos_sim_sub2, n_digits=5))

        elif not align.has_sub_rmsd and align.has_sub:

            print("\n> INFO: Reexecute Kabsch alignment to include the substructure decomposition!")

        print("\n--------------------------- Z-matrix properties -----------------------------")
        print("   >>>   RMSD              :  " + self.format_value(align.rmsd_z_matrix, n_digits=5))
        print("\n   >>>   - Decomposition   :  total percentage")
        print("             distances     :  " + self.format_value(align.c_dis * 100.0, n_digits=0) + " %")
        print("             angles        :  " + self.format_value(align.c_ang * 100.0, n_digits=0) + " %")
        print("             dihedrals     :  " + self.format_value(align.c_tor * 100.0, n_digits=0) + " %")

    def pt_data_set_info(self, molecule1, molecule2):

        print("\n-----------------------------------------------------------------------------")
        print("==================== Information about Molecular Data =======================")
        print("-----------------------------------------------------------------------------")
        print("-------  Molecule 1: 'Model'                 Molecule 2: 'Reference'  -------")
        print("   #Atoms = " + str(molecule1.n_atoms) + "                          #Atoms = " + str(molecule2.n_atoms))
        print("   #H-Atoms = " + str(molecule1.n_h_atoms) + "                          #H-Atoms = " + str(
            molecule2.n_h_atoms))
        print("-- Sym.-Idf.        --  [xyz] / A    |  Sym.-Idf.        -- [xyz] / A  --")

        if molecule1.n_atoms > molecule2.n_atoms:

            common_number = molecule2.n_atoms
            rest_number = molecule1.n_atoms - molecule2.n_atoms
            to_print = 'molecule1'

        elif molecule1.n_atoms < molecule2.n_atoms:

            common_number = molecule1.n_atoms
            rest_number = molecule2.n_atoms - molecule1.n_atoms
            to_print = 'molecule2'

        else:

            common_number = molecule2.n_atoms
            rest_number = 0

        [print("   " + str(molecule1.sym_idf[entry]) + "\t" + str(np.around(molecule1.cor[entry], 3)) + "\t\t" +
               str(molecule2.sym_idf[entry]) + "\t" + str(np.around(molecule2.cor[entry], 3)))
         for entry in range(common_number)]

        if rest_number != 0:

            if to_print == 'molecule1':

                [print("   " + str(molecule1.sym_idf[common_number + entry]) + "\t" +
                       str(np.around(molecule1.cor[common_number + entry], 3))) for entry in range(rest_number)]

            else:

                [print("   \t\t\t\t\t" + str(molecule2.sym_idf[common_number + entry]) + "\t" +
                       str(np.around(molecule2.cor[common_number + entry], 3))) for entry in range(rest_number)]

    def pt_x_ray_menu(self, n_atoms, symOPs, picker_type):

        print("\n-----------------------------------------------------------------------------")
        print("========================= X-ray Data Modification ===========================")
        print("-----------------------------------------------------------------------------")
        print("-10 Exit the menu")
        print("-5  Export structure to '.xyzs' file")
        print("-4  Export structure to '.xyz' file")
        print("-2  Change picker mode (current: '" + str(picker_type) + "')")
        print("-1  Show the X-ray structure again")
        print("-----------------------------------------------------------------------------")
        print("    Current number of atoms     : " + str(n_atoms))
        print("-----------------------------------------------------------------------------")
        [print(str(entry) + "   expand by operation\t\t'" + str(symOPs[entry]) + "'") for entry in range(len(symOPs))]
        print(str(entry + 1) + "   expand by custom operation ")
        print("-----------------------------------------------------------------------------")

        choices = [-10, -5, -4, -2, -1]
        choices.extend(range(len(symOPs) + 1))

        return choices

    def pt_match_menu(self, settings):

        print("\n-----------------------------------------------------------------------------")
        print("================ Symmetry Adjustments & Sequence Matching ===================")
        print("-----------------------------------------------------------------------------")
        print("-6  Set number of deviations which are highlighted in the plot (current = " + str(settings.n_dev) + ")")
        print("-5  Load the saved status (save point available: '" + str(self.was_saved) + "')")
        print("-4  Change plot settings")
        print("-3  Manually swap atoms in Model structure")
        print("-2  Change matching algorithm or solver")
        print("-1  Match molecular sequences based on current alignment")
        print("-----------------------------------------------------------------------------")
        print("    Current matching algorithm  : '" + str(self.match_alg) + "'")
        print("    Current matching solver     : '" + str(self.match_solv) + "'")
        print("    Structures were matched     : '" + str(self.is_matched) + "'")
        print("-----------------------------------------------------------------------------")
        print("0   ... exit the menu (no return)")
        print("1   ... inversion at the origin")
        print("2   ... reflection at the xy plane")
        print("3   ... reflection at the xz plane")
        print("4   ... reflection at the yz plane")
        print("5   ... rotation around the x axis")
        print("6   ... rotation around the y axis")
        print("7   ... rotation around the z axis")
        print("8   ... show the molecules again")
        print("-----------------------------------------------------------------------------")
        print("10  ... save current changes (status was saved: '" + str(self.was_saved) + "')")
        print("20  ... export structures")
        print("-----------------------------------------------------------------------------")

    def pt_change_algorithm(self, alg_type):

        if alg_type == 'solving':

            pt_alg = self.match_solv

        elif alg_type == 'matching':

            pt_alg = self.match_alg

        print("\n> Changed " + str(alg_type) + " algorithm to '" + str(pt_alg) + "'.")

    def pt_algorithm_menu(self, molecule):

        print("\n-----------------------------------------------------------------------------")
        print("======================= Matching Algorithm Submenu ==========================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-1  Show details of current solving algorithm ('" + str(self.match_solv) + "')")
        print("0   Show details of current matching algorithm ('" + str(self.match_alg) + "')")
        print("-----------------------------------------------------------------------------")
        print("1   ... use absolute distance between atoms ('distance')")
        print("2   ... use combination of absolut and relative distances ('combined')")
        print("3   ... use random permutations ('brute_force')")
        print("-----------------------------------------------------------------------------")
        print("4   ... use 'Hungarian' solver for the permutation matrix ('hungarian')")
        print("5   ... use 'aRMSD' solver for the permutation matrix ('standard')")
        print("-----------------------------------------------------------------------------")

    def pt_solve_algorithm_details(self):

        print("\n-----------------------------------------------------------------------------")
        print("Details about the algorithm '" + str(self.match_solv) + "':")

        if self.match_solv == 'hungarian':

            print("This 'Hungarian/Munkres' algorithm is the de facto default solution to")
            print("cost problem which is similar to the matching of two molecular structures.")
            print("It is quite fast and hence the default in 'aRMSD'.")

        elif self.match_solv == 'standard':

            print("The 'standard' solving algorithm is a simplified version of the 'Hungarian'")
            print("algorithm and was initially developed in the first versions of 'aRMSD'. ")
            print("It is not as fast as the implementation of the 'Hungarian/Munkres'")
            print("algorithm but it should be tried if the defaults fail.")

    def pt_match_algorithm_details(self):

        print("\n-----------------------------------------------------------------------------")
        print("Details about the algorithm '" + str(self.match_alg) + "':")

        if self.match_alg == 'distance':

            print("This algorithm uses the distances between the possible atom pairs for the")
            print("creation of the permutation matrix. Hence it requires an alignment of")
            print("sufficiently good quality - provided by the user through\nsymmetry transformations.")

        elif self.match_alg == 'combined':

            print("The 'combined' algorithm combine the distances between the possible atom")
            print("pairs and the relative positions of the atoms within the molecule to")
            print("create a reasonable permutation matrix. A sufficiently good alignment")
            print("drastically improves the matching results.")

        elif self.match_alg == 'brute_force':

            print("This is an experimental algorithm that tries to find the best solution to")
            print("the matching problem through all possible permutations. This will take a")
            print("lot of time - however: nothing is required from the user.")

        print("-----------------------------------------------------------------------------")

    def pt_export_structure_menu(self, min_rad, max_rad):

        print("\n-----------------------------------------------------------------------------")
        print("============================= Export Structures =============================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-1  Project atomic radii for export (current range: " + str(min_rad) + " to " + str(max_rad) + ")")
        print("-----------------------------------------------------------------------------")
        print("0   ... export data in two '.xyz' files")
        print("1   ... export data in two '.xyzs' files")
        print("2   ... export combined data in one '.xyzs' file")
        print("-----------------------------------------------------------------------------")

    def pt_export_kabsch_menu(self):

        print("\n-----------------------------------------------------------------------------")
        print("============================= Export Structures =============================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-----------------------------------------------------------------------------")
        print("0   ... export superposition in two '.xyz' files")
        print("1   ... export superposition in one '.xyzs' files")
        print("2   ... export aRMSD representation in one '.xyzs' file")
        print("-----------------------------------------------------------------------------")

    def pt_change_vtk_settings_menu(self, settings, molecule1, molecule2):

        print("\n-----------------------------------------------------------------------------")
        print("============================ Change VTK Settings ============================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-1  Change current plotting style")
        print("-----------------------------------------------------------------------------")
        print("    Current plotting style  : '" + str(settings.name) + "'")
        print("-----------------------------------------------------------------------------")
        print("0   ... draw labels (current = " + str(settings.draw_labels) + ")")
        print("1   ... change label type (current = " + str(settings.label_type) + ")")
        print("2   ... draw arrows (current = " + str(settings.draw_arrows) + ")")
        print("3   ... draw legend (current = " + str(settings.draw_legend) + ")")
        print("4   ... set global scale factor (current = " + str(settings.scale_glob) + ")")
        print("5   ... set atom scale factor (current = " + str(settings.scale_atom) + ")")
        print("6   ... set resolution (current = " + str(settings.res_atom) + ")")
        print("7   ... set color of '" + str(molecule1.name) + "' (current = " + str(settings.col_model_hex) + ")")
        print("8   ... set color of '" + str(molecule2.name) + "' (current = " + str(settings.col_refer_hex) + ")")
        print("9   ... use lightning (current = " + str(settings.use_light) + ")")
        print("10  ... set export magnification factor (current = " + str(settings.magnif_fact) + ")")
        print("-----------------------------------------------------------------------------")

    def pt_w_function_menu(self, align):

        print("\n-----------------------------------------------------------------------------")
        print("========================== Set Weighting Functions ==========================")
        print("-----------------------------------------------------------------------------")
        print("Info: For functions marked with a '*' the contributions from other atoms")
        print("      to each individual atom (multi-center-correction) can be calculated")
        print("      if requested.")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-----------------------------------------------------------------------------")
        print("    Current weighting function  : '" + str(align.wts_type) + "'")
        print("    Calculate mcc contribution  : '" + str(align.calc_mcc) + "'")
        print("-----------------------------------------------------------------------------")
        print("0   ... geometric / unweighted")
        print("1   ... x-ray scattering factors (*)")
        print("2   ... atomic masses")
        print("3   ... number of electrons")
        print("4   ... number of core electrons")
        print("5   ... spherical electron densities (*)")
        print("6   ... LDA electron densities (*)")
        print("-----------------------------------------------------------------------------")

    def pt_xsf_wrong_source(self):

        print("Unrecognized X-ray source, use: 'MoKa', 'CuKa', 'CoKa', 'FeKa', 'CrKa'")
        print("\n> Using 'MoKa' scattering factors (lambda = 0.071073 nm)")

    def pt_xsf_import_error(self):

        print("\n> ERROR: Scattering factor import failed, using prestored factors ...")

    def pt_kabsch_menu(self, align, settings):

        print("\n-----------------------------------------------------------------------------")
        print("============== Kabsch Algorithm, Statistics & Visualization  ================")
        print("-----------------------------------------------------------------------------")
        print("-10 Exit aRMSD")
        print("-8  Plot aRMSD color map")
        print("-7  Change general RMSD settings")
        print("-6  Add/remove bond")
        print("-4  Change plot settings")
        print("-3  Define two substructures (structures are defined: '" + str(align.has_sub) + "')")
        print("-2  Change weighting function")
        print("-1  Perform Kabsch alignment (required for all functions)")
        print("-----------------------------------------------------------------------------")
        print("    Current weighting function  : '" + str(align.wts_type) + "'")
        print("    Calculate mcc contribution  : '" + str(align.calc_mcc) + "'")
        print("    Kabsch alignment performed  : '" + str(align.has_kabsch) + "'")
        print("-----------------------------------------------------------------------------")
        print("0   ... visualize results in aRMSD representation")
        print("1   ... visualize structural superposition")
        print("2   ... perform statistic investigation of bond lengths and angles")
        print("3   ... show RMSD results")
        print("4   ... interpolate between the structures (cart., " + str(settings.n_steps_interp) + " steps)")
        print("5   ... generate outfile")
        print("20  ... export structural data")
        print("-----------------------------------------------------------------------------")

    def pt_change_rmsd_settings_menu(self, settings):

        print("\n-----------------------------------------------------------------------------")
        print("============================ Change RMSD Settings ===========================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-5  Use RYG coloring scheme for 'aRMSD representation' (current = " + str(settings.use_aRMSD_col) + ")")
        print("-----------------------------------------------------------------------------")
        print("0   ... set maximum RMSD value for color projection (current = " + str(settings.max_RMSD_diff) + ")")
        print("1   ... set the number of colors for the aRMSD representation (current = " + str(
            settings.n_col_aRMSD) + ")")
        print("2   ... set threshold for aRMSD bond comparison (current = " + str(settings.thresh) + ")")
        print("3   ... set basic color of aRMSD bonds (current = " + str(settings.col_bnd_glob_hex) + ")")
        print("4   ... set color of shortened bonds (current = " + str(settings.col_short_hex) + ")")
        print("5   ... set color of elongated bonds (current = " + str(settings.col_long_hex) + ")")
        print("6   ... set length of the bond intersection (current = " + str(1.0 - 2 * settings.n) + ")")
        print("7   ... set precision for the aRMSD picker (current = " + str(settings.calc_prec) + ")")
        print("8   ... set number of highest property deviations to be shown (current = " + str(
            settings.n_max_diff) + ")")
        print("9   ... set the number of points for structure interpolations (current = " + str(
            settings.n_steps_interp) + ")")
        print("-----------------------------------------------------------------------------")

    def pt_change_rmsd_vtk_settings_menu(self, settings, align):

        print("\n-----------------------------------------------------------------------------")
        print("============================ Change VTK Settings ============================")
        print("-----------------------------------------------------------------------------")
        print("-10 Return to upper menu")
        print("-----------------------------------------------------------------------------")
        print("0   ... draw labels (current = " + str(settings.draw_labels) + ")")
        print("1   ... change label type (current = " + str(settings.label_type) + ")")
        print("2   ... set global scale factor (current = " + str(settings.scale_glob) + ")")
        print("3   ... set resolution (current = " + str(settings.res_atom) + ")")
        print("4   ... set color of '" + str(align.name1) + "' (current = " + str(settings.col_model_fin_hex) + ")")
        print("5   ... set color of '" + str(align.name2) + "' (current = " + str(settings.col_refer_fin_hex) + ")")
        print("6   ... use lightning (current = " + str(settings.use_light) + ")")
        print("7   ... set export magnification factor (current = " + str(settings.magnif_fact) + ")")
        print("8   ... draw color bar (current = " + str(settings.draw_col_map) + ")")
        print("-----------------------------------------------------------------------------")

    def pt_kabsch_alignment(self, w_function_type):

        print("\nNow performing Kabsch alignment of weighted coordinates")
        print("> The type of weighting function is: " + str(w_function_type))

    def pt_rot_matrix(self, rot_matrix):

        print("\nThe rotation matrix for the optimal alignment (from Standard Orientation) is:\n")
        print("\t           |" + "{:+06.8f}".format(rot_matrix[0][0]) + "  " + "{:+06.8f}".format(rot_matrix[0][1]) +
              "  " + "{:+06.8f}".format(rot_matrix[0][2]) + "|")
        print("\t     U  =  |" + "{:+06.8f}".format(rot_matrix[1][0]) + "  " + "{:+06.8f}".format(rot_matrix[1][1]) +
              "  " + "{:+06.8f}".format(rot_matrix[1][2]) + "|")
        print("\t           |" + "{:+06.8f}".format(rot_matrix[2][0]) + "  " + "{:+06.8f}".format(rot_matrix[2][1]) +
              "  " + "{:+06.8f}".format(rot_matrix[2][2]) + "|")

    def pt_bond_added(self, align, idx1, idx2):

        print(
            "\n> A bond between [" + str(align.sym_idf[idx1]) + " -- " + str(align.sym_idf[idx2]) + "] has been added!")

    def pt_bond_removed(self, align, idx1, idx2):

        print("\n> The bond between [" + str(align.sym_idf[idx1]) + " -- " + str(
            align.sym_idf[idx2]) + "] has been removed!")

    def pt_wrong_indices(self, align):

        print("\n> ERROR: The given bond identifiers are out of range (1 - " + str(align.n_atoms + 1) + ")!")
