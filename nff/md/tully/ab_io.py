import os
import json
from jinja2 import Template
from rdkit import Chem
import time
import numpy as np
import copy

from chemconfigs.parsers.qchem import (get_cis_grads,
                                       get_nacv,
                                       get_sf_energies)

from nff.utils.misc import bash_command


CONFIG_DIRS = {
    "bhhlyp_6-31gs_sf_tddft_engrad_qchem":
    "qchem/bhhlyp_6-31gs_sf_tddft_engrad",

    "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
    "qchem/bhhlyp_6-31gs_sf_tddft_nacv"
}

SPIN_FLIP_CONFIGS = ["bhhlyp_6-31gs_sf_tddft_engrad_qchem",
                     "bhhlyp_6-31gs_sf_tddft_engrad_qchem_pcm",
                     "bhhlyp_6-31gs_sf_tddft_nacv_qchem",
                     "bhhlyp_6-31gs_sf_tddft_nacv_qchem_pcm"]

PERIODICTABLE = Chem.GetPeriodicTable()


def render(temp_text,
           jobspec,
           write_path):

    template = Template(temp_text)
    inp = template.render(jobspec=jobspec)

    with open(write_path, 'w') as f_open:
        f_open.write(inp)


def get_files(config, jobspec):
    platform = jobspec['details'].get("platform")
    dic = config
    if platform is not None:
        dic = config[platform]

    files = [dic['job_template_filename'],
             *dic['extra_template_filenames']]

    if config['name'] == "bhhlyp_6-31gs_sf_tddft_engrad_qchem":
        rm_file = 'qchem_bhhlyp_6-31gs_sf_tddft_engrad.inp'
        if rm_file in files:
            files.remove(rm_file)

    return files


def render_config(config_name,
                  config_dir,
                  config,
                  jobspec,
                  job_dir,
                  num_parallel,
                  run_parallel=True):

    files = get_files(config, jobspec)

    # use 1 / num_parallel * total number of cores
    # in this job
    this_jobspec = copy.deepcopy(jobspec)

    if run_parallel:
        nprocs = this_jobspec['details']['nprocs']
        this_jobspec['details']['nprocs'] = int(nprocs / num_parallel)

    for file in files:
        temp_path = os.path.join(config_dir, file)
        write_path = os.path.join(job_dir, file)

        with open(temp_path, 'r') as f:
            temp_text = f.read()

        render(temp_text=temp_text,
               jobspec=this_jobspec,
               write_path=write_path)

    info_path = os.path.join(job_dir, 'job_info.json')
    with open(info_path, 'w') as f:
        json.dump(this_jobspec, f, indent=4)


def translate_dir(direc):
    if '$HOME' in direc:
        direc = direc.replace("$HOME",
                              os.environ["HOME"])

    return direc


def load_config(config_name,
                htvs_dir):

    config_dir_name = CONFIG_DIRS[config_name]
    config_dir = os.path.join(translate_dir(htvs_dir),
                              'chemconfigs',
                              config_dir_name)

    config_path = os.path.join(config_dir,
                               'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config, config_dir


def render_all(config_name,
               jobspec,
               job_dir,
               num_parallel):

    htvs_dir = jobspec['details']['htvs']
    config, config_dir = load_config(config_name=config_name,
                                     htvs_dir=htvs_dir)

    render_config(config_name=config_name,
                  config_dir=config_dir,
                  config=config,
                  jobspec=jobspec,
                  job_dir=job_dir,
                  num_parallel=num_parallel)


def get_coords(nxyz):
    coords = []
    for l in nxyz:
        this_coord = {"element": PERIODICTABLE.GetElementSymbol(int(l[0])),
                      "x": l[1],
                      "y": l[2],
                      "z": l[3]}
        coords.append(this_coord)

    return coords


def init_jobspec(nxyz,
                 details,
                 charge):

    coords = get_coords(nxyz)
    jobspec = {'details': details,
               'coords': coords,
               'charge': charge}

    return jobspec


def sf_grad_jobspec(jobspec,
                    surf):

    jobspec['details'].update({'grad_roots': [int(surf)],
                               'num_grad_roots': 1})

    return jobspec


def sf_nacv_jobspec(jobspec,
                    singlet_path,
                    num_states):

    with open(singlet_path, 'r') as f:
        singlets = json.load(f)

    coupled_states = singlets[:num_states]
    details = jobspec['details']
    details.update({"coupled_states": coupled_states})

    return jobspec


def run_job(config_name,
            jobspec,
            job_dir,
            num_parallel):

    render_all(config_name=config_name,
               jobspec=jobspec,
               job_dir=job_dir,
               num_parallel=num_parallel)

    cmd = f"cd {job_dir} && bash job.sh && rm *fchk"
    p = bash_command(cmd)

    return p


def bhhlyp_6_31gs_sf_tddft_engrad_qchem(nxyz,
                                        details,
                                        charge,
                                        surf,
                                        job_dir,
                                        num_parallel):

    jobspec = init_jobspec(nxyz=nxyz[0],
                           details=details,
                           charge=charge)
    jobspec = sf_grad_jobspec(jobspec=jobspec,
                              surf=surf)

    config_name = 'bhhlyp_6-31gs_sf_tddft_engrad_qchem'

    grad_dir = os.path.join(job_dir, 'grad')
    if not os.path.isdir(grad_dir):
        os.makedirs(grad_dir)

    # copy job_info.json
    p = run_job(config_name=config_name,
                jobspec=jobspec,
                job_dir=grad_dir,
                num_parallel=num_parallel)

    return p


def get_singlet_path(job_dir):
    singlet_path = os.path.join(job_dir, 'grad', 'singlets.json')
    return singlet_path


def bhhlyp_6_31gs_sf_tddft_nacv_qchem(nxyz,
                                      details,
                                      charge,
                                      num_states,
                                      job_dir,
                                      num_parallel):

    singlet_path = get_singlet_path(job_dir)
    exists = False
    while not exists:
        exists = os.path.isfile(singlet_path)
        time.sleep(5)

    jobspec = init_jobspec(nxyz=nxyz[0],
                           details=details,
                           charge=charge)
    jobspec = sf_nacv_jobspec(jobspec=jobspec,
                              singlet_path=singlet_path,
                              num_states=num_states)

    config_name = "bhhlyp_6-31gs_sf_tddft_nacv_qchem"
    nacv_dir = os.path.join(job_dir, 'nacv')
    if not os.path.isdir(nacv_dir):
        os.makedirs(nacv_dir)

    p = run_job(config_name=config_name,
                jobspec=jobspec,
                job_dir=nacv_dir,
                num_parallel=num_parallel)

    return p


def run_sf(job_dir,
           nxyz,
           charge,
           num_states,
           surf,
           grad_details,
           nacv_details,
           grad_config,
           nacv_config,
           calc_nacv=True):

    procs = []
    proc_names = []

    num_parallel = 2 if calc_nacv else 1

    if grad_config == 'bhhlyp_6-31gs_sf_tddft_engrad_qchem':
        p = bhhlyp_6_31gs_sf_tddft_engrad_qchem(nxyz=nxyz,
                                                details=grad_details,
                                                charge=charge,
                                                surf=surf,
                                                job_dir=job_dir,
                                                num_parallel=num_parallel)

        procs.append(p)
        proc_names.append("Q-Chem engrad")

    else:
        raise NotImplementedError

    if calc_nacv:
        if nacv_config == "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
            p = bhhlyp_6_31gs_sf_tddft_nacv_qchem(nxyz=nxyz,
                                                  details=nacv_details,
                                                  charge=charge,
                                                  num_states=num_states,
                                                  job_dir=job_dir,
                                                  num_parallel=num_parallel)

            procs.append(p)
            proc_names.append("Q-Chem NACV")

        else:
            raise NotImplementedError

    for i, p in enumerate(procs):
        exit_code = p.wait()
        if exit_code != 0:
            msg = f"{proc_names[i]} returned an error"
            raise Exception(msg)


def parse_sf_grads(job_dir):
    path = os.path.join(job_dir, 'singlet_grad.out')
    with open(path, 'r') as f:
        lines = f.readlines()
    output_dics = get_cis_grads(lines)

    return output_dics


def parse_sf_ens(job_dir):
    path = os.path.join(job_dir, 'singlet_energy.out')
    with open(path, 'r') as f:
        lines = f.readlines()
    output_dics = get_sf_energies(lines)

    return output_dics


def parse_sf_nacv(job_dir,
                  conifg_name):

    if conifg_name == "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
        out_name = 'qchem_bhhlyp_6-31gs_sf_tddft_nacv'
    else:
        raise NotImplementedError

    path = os.path.join(job_dir, f'{out_name}.out')
    with open(path, 'r') as f:
        lines = f.readlines()
    output_dics = get_nacv(lines)

    return output_dics


def check_sf(grad_config,
             nacv_config):
    configs = [grad_config, nacv_config]
    is_sf = any([config in SPIN_FLIP_CONFIGS for config in configs])

    return is_sf


def parse_sf(job_dir,
             nacv_config,
             calc_nacv=True):

    nacv_dir = os.path.join(job_dir, 'nacv')
    grad_dir = os.path.join(job_dir, 'grad')

    en_dics = parse_sf_ens(job_dir=grad_dir)
    grad_dics = parse_sf_grads(job_dir=grad_dir)
    if calc_nacv:
        nacv_dic = parse_sf_nacv(job_dir=nacv_dir,
                                 conifg_name=nacv_config)
    else:
        nacv_dic = {}

    return en_dics, grad_dics, nacv_dic


def en_to_arr(results,
              en_dics,
              singlets):

    for dic in en_dics:
        state = dic['state']
        if state not in singlets:
            continue

        idx = singlets.index(state)
        key = f"energy_{idx}"
        en = dic['energy']
        results[key] = np.array([en])

    return results


def grad_to_arr(results,
                grad_dics,
                singlets):

    combined_grad = {}
    for dic in grad_dics:
        combined_grad.update(dic)

    for abs_state, grad in combined_grad.items():
        idx = singlets.index(abs_state)
        key = f'energy_{idx}_grad'

        grad = np.array(grad)
        shape = grad.shape
        results[key] = grad.reshape(1, *shape)

    return results


def nacv_to_arr(results,
                nacv_dic,
                singlets):

    translation = {"deriv_nacv_etf": 'nacv'}
    keys = ['deriv_nacv_etf', 'force_nacv']

    for key in keys:
        if key not in nacv_dic:
            continue
        sub_dic = nacv_dic[key]
        for start_state, sub_sub in sub_dic.items():
            for end_state, nacv in sub_sub.items():
                singlet_start = singlets.index(start_state)
                singlet_end = singlets.index(end_state)

                translate_base = translation.get(key, key)
                results_key = (f"{translate_base}_{singlet_start}"
                               f"{singlet_end}")

                nacv = np.array(nacv)
                shape = nacv.shape
                results[results_key] = nacv.reshape(1, *shape)

    return results


def combine_results(singlets,
                    en_dics,
                    grad_dics,
                    nacv_dic):

    results = {}
    results = en_to_arr(results=results,
                        en_dics=en_dics,
                        singlets=singlets)
    results = grad_to_arr(results=results,
                          grad_dics=grad_dics,
                          singlets=singlets)
    results = nacv_to_arr(results=results,
                          nacv_dic=nacv_dic,
                          singlets=singlets)

    return results


def parse(job_dir,
          grad_config,
          nacv_config,
          calc_nacv=True):

    is_sf = check_sf(grad_config=grad_config,
                     nacv_config=nacv_config)

    if is_sf:
        en_dics, grad_dics, nacv_dic = parse_sf(job_dir=job_dir,
                                                nacv_config=nacv_config,
                                                calc_nacv=calc_nacv)

        singlet_path = get_singlet_path(job_dir)
        with open(singlet_path, 'r') as f:
            singlets = json.load(f)

        results = combine_results(singlets=singlets,
                                  en_dics=en_dics,
                                  grad_dics=grad_dics,
                                  nacv_dic=nacv_dic)

    else:
        raise NotImplementedError

    return results


def get_results(nxyz,
                charge,
                num_states,
                surf,
                job_dir,
                grad_config,
                nacv_config,
                grad_details,
                nacv_details,
                calc_nacv=True):

    is_sf = check_sf(grad_config=grad_config,
                     nacv_config=nacv_config)

    if is_sf:
        run_sf(job_dir=job_dir,
               nxyz=nxyz,
               charge=charge,
               surf=surf,
               num_states=num_states,
               grad_details=grad_details,
               nacv_details=nacv_details,
               grad_config=grad_config,
               nacv_config=nacv_config,
               calc_nacv=calc_nacv)

    else:
        raise NotImplementedError

    results = parse(job_dir=job_dir,
                    grad_config=grad_config,
                    nacv_config=nacv_config,
                    calc_nacv=calc_nacv)

    return results
