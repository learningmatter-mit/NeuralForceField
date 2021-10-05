import numpy as np
import io

try:
    from alog import Logger
    from acore import settings
    import acore as ac
except ModuleNotFoundError:
    print("You need to install the group's fork of aRMSD and put it in your path "
          "https://github.mit.edu/MLMat/aRMSD")

VERSION, YEAR = "0.9.4", "2017"

def write_coord(coord):
    """ Adjusts whitespace for coordinates """

    return "{:06.8f}".format(coord) if coord < 0.0 else " " + "{:06.8f}".format(coord)


def data_to_xyz(sym, cor):
    """ Transform Molecule attributes symbol and coordinates to xyz as a string"""

    xyz_string = str(len(sym)) + '\n' + '\n'
    for atom in range(len(sym)):
        xyz_string = xyz_string + str(sym[atom]) + '\t' + str(write_coord(cor[atom][0])) + '\t' + str(
            write_coord(cor[atom][1])) + '\t' + str(write_coord(cor[atom][2])) + '\n'

    return xyz_string

def kabsch(rxn,
           indexedproductgeom_raw,
           reactantgeom_raw,
           rid,
           pid):
    
    try:
        settings = ac.settings()
        settings.parse_settings('settings.cfg')
    except NameError:
        pass
    logger = Logger(VERSION, YEAR)

    dt = np.dtype([('RXN_IDX', 'int'),
                   ('RCT_ID', 'int'),
                   ('PRO_ID', 'int'),
                   ('RMSD', 'float64'),
                   ('REACTANT_COM', 'float64', (1, 3)),
                   ('PRODUCT_COM', 'float64', (1, 3)),
                   ('ROT_MAT', 'float64', (3, 3)),
                   ('FINAL_RXT_XYZ', 'U10000'),
                   ('FINAL_PDT_XYZ', 'U10000')])
    row_data = np.array([(0, 0, 0, 0.5, (2, 1, 1), (1, 1, 1), np.identity(
        3), '', '')], dtype=dt)  # Initialization with dummy values

    row_data['RCT_ID'] = rid
    row_data['PRO_ID'] = pid

    # idp is indexed product
    # Convert to array of strings separated by "\n"
    indexedproductgeom = io.StringIO(indexedproductgeom_raw)
    idp_data = indexedproductgeom.readlines()

    idp_mol_name = "product"
    idp_element_symbol, idp_element_xyz = ac.read_xyz_file(logger, idp_data)
    idp_element_xyz_std = None
    # Create a molecule object
    molecule1 = ac.Molecule(idp_mol_name, idp_element_symbol,
                            idp_element_xyz, idp_element_xyz_std)
    molecule1.get_charge()
    molecule1.get_mass()
    molecule1.calc_com(calc_for='molecule')
    row_data['PRODUCT_COM'] = molecule1.com

    molecule1.shift_com(calc_for='molecule')  # molecule translated by -COM

    # rxt is the reactant
    # Convert to array of strings separated by "\n"
    reactantgeom = io.StringIO(reactantgeom_raw)
    rxt_data = reactantgeom.readlines()
    rxt_mol_name = "reactant"
    rxt_element_symbol, rxt_element_xyz = ac.read_xyz_file(logger, rxt_data)
    rxt_element_xyz_std = None
    # Create a molecule object
    molecule2 = ac.Molecule(rxt_mol_name, rxt_element_symbol,
                            rxt_element_xyz, rxt_element_xyz_std)
    molecule2.get_charge()
    molecule2.get_mass()
    molecule2.calc_com(calc_for='molecule')
    row_data['REACTANT_COM'] = molecule2.com
    molecule2.shift_com(calc_for='molecule')  # molecule translated by -COM

    # Update molecule objects
    molecule1.update_properties(settings, get_bonds=True)
    molecule2.update_properties(settings, get_bonds=True)

    # kabsch alogorithm
    align = ac.Kabsch(molecule1, molecule2, settings)
    align.init_coords(molecule1, molecule2)
    align.get_symbols()
    align.get_weights(logger)
    align.kabsch_rot_matrix()  # Rotate first coordinate array
    row_data['ROT_MAT'] = align.rot_mat
    row_data['RMSD'] = ac.fast_rmsd(align.cor_mol1_kbs, align.cor_mol2_kbs)
    row_data['FINAL_PDT_XYZ'] = data_to_xyz(align.sym_mol1, align.cor_mol1_kbs)
    row_data['FINAL_RXT_XYZ'] = data_to_xyz(align.sym_mol2, align.cor_mol2_kbs)
    row_data['RXN_IDX'] = rxn

    return row_data
