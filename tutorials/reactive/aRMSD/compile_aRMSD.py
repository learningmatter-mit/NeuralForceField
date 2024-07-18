"""
Script for an easy installation/compilation of aRMSD from command line
License: MIT
(c) 2016 by Arne Wagner

*Recent changes:
- Installer writes PyInstaller .spec file on-the-fly
- Added support for openbabel (writes hook if none exists)
- PyInstaller imports/excludes can be modified in the functions
- Added optional openbabel flag for compilation
"""


from __future__ import print_function
from distutils.sysconfig import get_python_lib
import os
import sys
import shutil
import subprocess
import shlex
from codecs import open
import time

inst_version = '1.4'  # Version of the installer


def pyinstaller_data(name, platform, obf_files):
    """ Sets up a PyInstaller dictionary with all parameters that will be written to the .spec file """

    # Excludes and hidden imports
    #data_excl = ['_ssl', '_hashlib', 'PySide', '_gtkagg', '_tkagg', '_wxagg', '_qt5agg',
    #             'bsddb', 'curses', 'pywin.debugger', 'pywin.debugger.dbgcon',
    #             'pywin.dialogs', 'tcl', 'Tkconstants', 'Tkinter', 'wx', '_Qt5Agg', '_webagg']

    data_excl = []
    
    hiddenimp = ['matplotlib', 'vtk', 'uncertainties']

    if obf_files is not None:  # Add pybel to hiddenimports

        hiddenimp.append('openbabel')

    # Extra data and binaries for PyInstaller
    ext_bin = []

    ext_dat = []

    # Setup dictionary and return it
    pyinst_dict = {'name': name, 'platform': platform, 'hiddenimports': hiddenimp, 'data_excludes': data_excl,
                   'binaries': ext_bin, 'extra_datas': ext_dat}

    return pyinst_dict


def analyze_arguments(arguments):
    """ Checks given arguments and passes correct ones to the compilation script """

    accepted_arg_prefix = ['--use_openbabel', '--use_cython', '--cython_compiler', '--overwrite']

    def _split(arg):

        pos = arg.find('=')
        prefix = arg[:pos]
        suffix = arg[pos+1:]

        return (None, None) if prefix not in accepted_arg_prefix else (prefix, suffix)

    # Default compiler arguments
    use_openbabel = False
    use_cython = False
    cython_compiler = 'msvc'
    overwrite = False

    if len(arguments) != 0:  # Arguments are given

        for entry in arguments:

            data = _split(entry)

            if data[0] == '--use_cython':

                use_cython = data[1]

            elif data[0] == '--cython_compiler':

                cython_compiler = data[1]

            elif data[0] == '--use_openbabel':

                use_openbabel = data[1]

            elif data[0] == '--overwrite':

                overwrite = data[1]

    return use_openbabel, use_cython, cython_compiler, overwrite


def check_for_ext(pyx_file_path, ext, default):
    """ Checks if a .pyx/.pyd file exists and returns the extension """

    return ext if os.path.isfile(pyx_file_path+ext) else default


def check_for_pyd_so(file_path):
    """ Checks if a file with .pyd or .so extension exists """

    return True if os.path.isfile(file_path+'.pyd') or os.path.isfile(file_path+'.so') else False


def get_current_version(armsd_dir):
    """ Returns the name of the executable (this part is copied from the spec file) """

    # Determine the version of aRMSD and append it to the file name
    contents = open(armsd_dir+'\\aRMSD.py').readlines()

    for index, line in enumerate(contents):

        if '__aRMSD_version__' in line and len(line.split()) == 3:

            version = eval(line.split()[-1])

        if 'is_compiled' in line and len(line.split()) > 7:  # Let the program now that it is compiled

            contents[index].replace('False', 'True')

    # Setup platform and program name
    platform, name = '', 'aRMSD'

    # Determine platform and architecture
    # First: Operating system
    if sys.platform == 'win32':

        platform = 'Win'

    elif sys.platform == 'darwin':

        platform = 'Mac'

    elif sys.platform == 'linux2':

        platform = 'Lin'

    else:

        platform = 'Os'

    # Second: 32 or 63 bit
    if sys.maxsize > 2 ** 32:

        platform += '64'

    else:
        
        platform += '32'

    name += '_{}_{}'.format(version, platform)

    return name, platform


def has_module(mod, site_packages_path):
    """ Checks for a module folder in the site pacakges path """

    return os.path.isdir(site_packages_path+'\\'+mod)


def copy_obfiles(build_dir, site_packages_path):
    """ Copies .obf files to build folder """

    # List of .obf files (file format support), add or remove obf files accordingly
    obf_files = ['formats_cairo.obf', 'formats_common.obf', 'formats_compchem.obf',
                 'formats_misc.obf', 'formats_utility.obf', 'formats_xml.obf']

    babel_dir = site_packages_path+'\\openbabel'  # Path of the .obf files

    # Copy the files from the openbabel path to the build directory, return the files names
    [shutil.copyfile(babel_dir+'\\'+entry, build_dir+'\\'+entry) for entry in obf_files]

    return obf_files


def write_ob_hook(site_packages_path, overwrite):
    """ Writes a working pyinstaller hook for openbabel if there is none """

    hook_path = site_packages_path+'\\PyInstaller\\hooks'  # Path of the PyInstaller hooks
    babel_data = site_packages_path+'\\openbabel\\data'  # Path of the openbabel data files

    if not os.path.isfile(hook_path+'\\hook-openbabel.py') or overwrite:  # Don't overwrite files

        data_files = os.listdir(babel_data)  # All files in the directory

        # If these files are not included openbabel will give a warning and fall back to the internal data
        dat_names = ['space-groups', 'element', 'types', 'resdata', 'bondtyp', 'aromatic', 'atomtyp']

        datas = []

        for entry in range(len(data_files)):

            if data_files[entry].split('.')[0] in dat_names:  # If a file in dat_names is found, add it to datas

                datas.append((site_packages_path+'\\openbabel\\data\\'+data_files[entry], 'openbabel\\data'))

        os.chdir(hook_path)

        # Write hook file
        with open('hook-openbabel.py', 'w') as outfile:

            outfile.write("""
# This hook has been created by aRMSD
# It may not work for the compilation of other executables

from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs('openbabel')
datas """+"= "+str(datas))

            print('>> An openbabel hook for PyInstaller has been created!')

            outfile.close()

    else:

        print('>> A preexisting openbabel hook was found and will be used!')


def write_spec_file(build_dir, pyinst_dict, obf_files):
    """ Writes a .spec file for PyInstaller """

    def _write_obf(obf_files, build_dir):

        return_string = 'a.binaries'        

        if obf_files is not None:

            if len(obf_files) == 1:  # Only one .obf files

                return_string += " + [('"+obf_files[0]+"', "+repr(build_dir+'\\'+obf_files[0])+", 'BINARY')],"

            else:

                for entry in range(len(obf_files) - 1):

                    return_string += " + [('"+obf_files[entry]+"', "+repr(build_dir+'\\'+obf_files[entry])+", 'BINARY')]"

		return_string += " + [('"+obf_files[-1]+"', "+repr(build_dir+'\\'+obf_files[-1])+", 'BINARY')],"

	else:

            return_string += ','

	return return_string

    os.chdir(build_dir)  # Change to build directory and create a new file

    spec_file = 'aRMSD.spec'

    obf_str = _write_obf(obf_files, build_dir)  # Write additional binary string for .spec file

    # Write temporary setup file
    with open(spec_file, 'w') as outfile:
        
        outfile.write("""
# Automatically created aRMSD 'spec' file for a PyInstaller based compilation
# This file deletes itself after the installation.

# Authors: Arne Wagner
# License: MIT

block_cipher = None

import os

folder = os.getcwd()  # Get current working directory

binaries = """+str(pyinst_dict['binaries'])+"""

extra_datas = """+str(pyinst_dict['extra_datas'])+"""

exclude_datas = """+str(pyinst_dict['data_excludes'])+"\n\n"+"""hiddenimports = """+str(pyinst_dict['hiddenimports'])+"""


a = Analysis(['aRMSD.py'],
             pathex = [folder],
             binaries = binaries,
             datas = extra_datas,
             hiddenimports = hiddenimports,
             hookspath = [],
             runtime_hooks = [],
             excludes = [],
             win_no_prefer_redirects = False,
             win_private_assemblies = False,
             cipher = block_cipher)

# Setup platform and program name """+"\n"+"platform, name = '"+pyinst_dict['platform']+"', '"+pyinst_dict['name']+"'\n")

        outfile.write("""
# Exclude some binaries
#a.binaries = [x for x in a.binaries if not x[0].startswith("zmq")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("IPython")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("docutils")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("pytz")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("wx")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("libQtWebKit")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("libQtDesigner")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("PySide")]
#a.binaries = [x for x in a.binaries if not x[0].startswith("libtk")]

# Exclude selected data
for exclude_data in exclude_datas:

    a.datas = [x for x in a.datas if exclude_data not in x[0]]
    
# Setup pyz
pyz = PYZ(a.pure, a.zipped_data,
          cipher = block_cipher)

exe = EXE(pyz,
          a.scripts,\n          """+_write_obf(obf_files, build_dir)+"""
          a.zipfiles,
          a.datas,
          name = name,
          debug = False,
          strip = False,
          upx = True,
          console = True,
          icon = folder+"""+r"'\\aRMSD_icon.ico')"+"""

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip = False,
               upx = True,
               name = name)

""")

    outfile.close()

    return spec_file


def package_cython_modules(build_dir, list_of_files, cython_compiler):
    """ Compiles .pyx/.py files to .pyd/.so files inplace """

    os.chdir(build_dir)  # Change to build directory and create a new file

    setup_file = 'cythonize_modules.py'
    
    # Write temporary setup file
    with open(setup_file, 'w') as outfile:
        
        outfile.write("""
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

setup(name = 'acore', ext_modules = cythonize('"""+list_of_files[0]+"""'),)
setup(name = 'aplot', ext_modules = cythonize('"""+list_of_files[1]+"""'),)
setup(name = 'alog', ext_modules = cythonize('"""+list_of_files[2]+"""'),)
""")

    t0 = time.clock()  # Start time

    # Cythonize modules and compile them to .pyd/.so libraries
    subprocess.call(r'python {setup} build_ext --inplace --compiler={cython_compiler}'.format(setup=setup_file,
                                                                                              cython_compiler=cython_compiler))
    t1 = time.clock()  # End time

    print('\n>> Modules were successfully compiled by Cython!')
    print('Compilation time: '+str(round((t1 - t0) / 60.0, 1))+' min')


def run_compilation(use_openbabel, use_cython, cython_compiler, overwrite):
    """ Runs the pyinstaller compilation with the given flags for cython and the c compiler """

    print('\n   *** This the official installer for aRMSD (Installer version: '+inst_version+') ***')
    print('==============================================================================')
    print('It will create a standalone executable using PyInstaller.')
    print('\nNote: The compilation process will take some time (see below),')
    print('      open/close additional windows and create temporary files.')

    if not use_cython:  # Estimates the compilation time based on tests - but may be different on other machines

        print('\n\t -- Estimated total compilation time: 30 min --')

    else:

        print('\n\t --Estimated total compilation time: 35 min --')

    print('------------------------------------------------------------------------------')
    print('Info: You can customize the build by adjusting the')
    print('      aRMSD.spec and compile_aRMSD.py files')
    print('------------------------------------------------------------------------------')

    # Determine site packages path
    site_packages_path = get_python_lib()

    # Check for PyInstaller and openbabel
    has_pyinst = has_module('pyinstaller', site_packages_path)
    has_obabel = has_module('openbabel', site_packages_path)
    
    obf_files = None  # Will be checked and updated if .obf files are copyied

    if has_pyinst:

        # Names of the build folder and the core modules (without extensions)
        build_folder_name = 'build'
        name_core, name_log, name_plot = 'acore', 'alog', 'aplot'

        basic_dir = os.getcwd()  # Determine the initial working directory
        armsd_dir = basic_dir+'\\armsd'  # aRMSD folder in the working directory
        build_dir = basic_dir+'\\'+build_folder_name  # Build folder directory

        # Check for pre-compiled .pyd files
        comp_core = check_for_pyd_so(armsd_dir + '\\' + name_core)
        comp_plot = check_for_pyd_so(armsd_dir + '\\' + name_plot)
        comp_log = check_for_pyd_so(armsd_dir + '\\' + name_log)

        if True in [comp_core, comp_plot, comp_log]:  # If a single pre-compiled module exists, don't compile

            print('\n>> Pre-compiled modules found...')
            use_cython = False

        # Check if .pyx files of the three modules exist
        ext_core = check_for_ext(armsd_dir+'\\'+name_core, '.pyx', '.py')
        ext_plot = check_for_ext(armsd_dir+'\\'+name_plot, '.pyx', '.py')
        ext_log = check_for_ext(armsd_dir+'\\'+name_log, '.pyx', '.py')

        print('\n>> Installer was called as...')
        print('\npython compile_aRMSD.py --use_openbabel='+str(use_openbabel)+' --use_cython='+str(use_cython)+
              ' --cython_compiler='+str(cython_compiler)+' --overwrite='+str(overwrite))

        print('\n>> Creating temporary directory... '+build_folder_name)

        if os.path.isdir(build_folder_name):  # Remove build folder if it exists

            shutil.rmtree(build_dir)
            print('\n>> Build directory already exists... it will be removed!')

        os.makedirs(build_folder_name)  # Make temporary build directory

        os.chdir(build_folder_name)  # Change to build folder

        # Copy core files to build directory
        shutil.copyfile(armsd_dir+'\\'+name_core+ext_core, build_dir+'\\'+name_core+ext_core)
        shutil.copyfile(armsd_dir+'\\'+name_plot+ext_plot, build_dir+'\\'+name_plot+ext_plot)
        shutil.copyfile(armsd_dir+'\\'+name_log+ext_log, build_dir+'\\'+name_log+ext_log)

        if overwrite:

            print('\n>> INFO: All existing files (hooks, etc.) will be overwritten')

        if use_openbabel and has_obabel:  # Copy obenbabel files

            print('\n>> Copying openbabel files...')            
            obf_files = copy_obfiles(build_dir, site_packages_path)
            write_ob_hook(site_packages_path, overwrite)

        elif use_openbabel and not has_obabel:

            print('\n>> ERROR: Openbabel was not found on your system, will continue without it!')

        else:

            print('\n>> INFO: Openbabel will not be used!')

        print('\n>> Copying core modules...')
        print('\t... '+name_core+ext_core)
        print('\t... '+name_plot+ext_plot)
        print('\t... '+name_log+ext_log)

        if use_cython:  # Cythonize pyx files or py files

            print('\n>> Attempting to use Cython in the compilation')

            try:

                # Import required modules
                from setuptools import setup
                from setuptools import Extension
                from Cython.Build import cythonize

                print('\n>> Cython and setuptools found, starting compilation...')
                will_use_cython = True

            except ImportError:  # Something went wrong, most likely no Cython installation

                print('\n>> ERROR: Will continue without cythonization!')
                will_use_cython = False

            if will_use_cython:

                print('\npython cythonize_modules.py build_ext --inplace --compiler='+cython_compiler)

                # Combine modules in list and compile to libraries
                sourcefiles = [name_core+ext_core, name_plot+ext_plot, name_log+ext_log]
                package_cython_modules(build_dir, sourcefiles, cython_compiler)

                # Remove .pyx/.py and .c files - the program will be automatically compiled with the cythonized files
                os.remove(name_core+ext_core)
                os.remove(name_core+'.c')
                os.remove(name_plot+ext_plot)
                os.remove(name_plot+'.c')
                os.remove(name_log+ext_log)
                os.remove(name_log+'.c')

        else:

            print('\n>> INFO: Cython will not be used!')

        print('\n>> Copying main program files...')

        # Gets the file name of the created executable
        file_name_dir, platform = get_current_version(armsd_dir)

        # Copy main file and icon to build directory
        shutil.copyfile(armsd_dir+'\\aRMSD.py', build_dir+'\\aRMSD.py')
        shutil.copyfile(basic_dir+'\\aRMSD_icon.ico', build_dir+'\\aRMSD_icon.ico')

        # Load PyInstaller information (modules can be adjusted in the respective function)
        pyinst_dict = pyinstaller_data(file_name_dir, sys.platform, obf_files)

        # Write .spec file for compilation
        spec_file = write_spec_file(build_dir, pyinst_dict, obf_files)
        pyinstaller_cmd = 'pyinstaller --onefile '+spec_file

        print('\n>> Calling PyInstaller...')
        print('\n'+build_dir+'> '+pyinstaller_cmd)

        t0 = time.clock()  # Start time

        # Compile files with PyInstaller - this should work on every system
        pyinstaller_args = shlex.split(pyinstaller_cmd+' '+spec_file)
        subprocess.call(pyinstaller_args)

        t1 = time.clock()  # End time

        # Copy executable to 'armsd' folder and delete all temporary files
        os.chdir(basic_dir)
        shutil.rmtree(build_dir+'\\dist\\'+file_name_dir)
        prg_file_name = os.listdir(build_dir+'\\dist')[0]  # List file (only one should be there) in distribution directory
        shutil.copyfile(build_dir+'\\dist\\'+prg_file_name, armsd_dir+'\\'+prg_file_name)
        shutil.rmtree(build_dir)

        # Echo successful creation, print compilation time
        print('Executable -- '+prg_file_name+' -- was created successfully!')
        print('Compilation time: '+str(round((t1 - t0) / 60.0, 1))+' min')
        print('Cleaning up files and directories')

        print('\nClean up complete, executable has been moved to:\n'+armsd_dir)
        print('\n>> Compilation complete!')
        print('\n-----------------------------------------------------------------------------')
        print('In order to use the executable, copy...')
        print('settings.cfg, the xsf folder and '+prg_file_name)
        print('to any directory of your choice and start the program.')
        print('It is recommended to call aRMSD from command line')
        print('e.g. Path\\to\\exe> '+prg_file_name)
        print('to catch potential errors. The start of the program may take a few seconds!')

    else:

        print('\n>> ERROR: PyInstaller was not found, install the package and run again!')
        print('--> from command line: pip install pyinstaller')


if __name__ == '__main__':  # Run the program

    arguments = sys.argv[1:]  # Get arguments

    use_openbabel, use_cython, cython_compiler, overwrite = analyze_arguments(arguments)  # Check arguments and set variables

    run_compilation(use_openbabel, use_cython, cython_compiler, overwrite)
