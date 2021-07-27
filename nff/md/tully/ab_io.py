import os
import json
from jinja2 import Template
from rdkit import Chem
import time
import numpy as np

from chemconfigs.parsers.qchem import get_cis_grad, get_nacv

from nff.utils.misc import bash_command


CONFIG_DIRS = {"bhhlyp_6_31gs_sf_tddft_engrad_qchem":
               "qchem/bhhlyp_6_31gs_sf_tddft_engrad",

               "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
               "qchem/bhhlyp_6-31gs_sf_tddft_nacv"

               }

SPIN_FLIP_CONFIGS = ["bhhlyp_6-31gs_sf_tddft_engrad_qchem",
                     "bhhlyp_6-31gs_sf_tddft_engrad_qchem_pcm",
                     "bhhlyp_6-31gs_sf_tddft_nacv_qchem",
                     "bhhlyp_6-31gs_sf_tddft_nacv_qchem_pcm"]

PERIODICTABLE = Chem.GetPeriodicTable()


def render(temp_text,
           params,
           write_path):

    template = Template(temp_text)
    inp = template.render(**params)

    with open(write_path, 'w') as f_open:
        f_open.write(inp)


def get_files(config, jobspec):
    platform = jobspec['details'].get("platform")
    if platform is not None:
        files = config[platform]['extra_template_filenames']
    else:
        files = config['extra_template_filenames']

    return files


def render_config(config_name,
                  config_dir,
                  config,
                  jobspec,
                  job_dir):

    files = get_files(config, jobspec)
    for file in files:
        temp_path = os.path.join(config_dir, file)
        write_path = os.path.join(job_dir, file)

        with open(temp_path, 'r') as f:
            temp_text = f.read()

        render(temp_text=temp_text,
               params=jobspec,
               write_path=write_path)


def load_config(config_name,
                htvs_dir):

    config_dir_name = CONFIG_DIRS[config_name]
    config_dir = os.path.join(htvs_dir,
                              'chemconfigs',
                              config_dir_name)

    config_path = os.path.join(config_dir,
                               'config.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config, config_dir


def render_all(config_name,
               jobspec,
               job_dir):

    htvs_dir = jobspec.details['htvs']
    config, config_dir = load_config(config_name=config_name,
                                     htvs_dir=htvs_dir)

    render_config(config_name=config_name,
                  config_dir=config_dir,
                  config=config,
                  jobspec=jobspec,
                  job_dir=job_dir)


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

    jobspec['details'].update({'grad_roots': [surf]})

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
            job_dir):

    render_all(config_name=config_name,
               jobspec=jobspec,
               job_dir=job_dir)

    cmd = f"cd {job_dir} && bash job.sh"
    p = bash_command(cmd)

    return p


def bhhlyp_6_31gs_sf_tddft_engrad_qchem(nxyz,
                                        details,
                                        charge,
                                        surf,
                                        job_dir):

    jobspec = init_jobspec(nxyz=nxyz,
                           details=details,
                           charge=charge)
    jobspec = sf_grad_jobspec(jobspec=jobspec,
                              surf=surf)

    config_name = 'bhhlyp_6-31gs_sf_tddft_engrad_qchem'
    p = run_job(config_name=config_name,
                jobspec=jobspec,
                job_dir=job_dir)

    return p


def bhhlyp_6_31gs_sf_tddft_nacv_qchem(nxyz,
                                      details,
                                      charge,
                                      num_states,
                                      job_dir):

    singlet_path = os.path.join(job_dir, 'singlets.json')
    exists = False
    while not exists:
        exists = os.path.isfile(singlet_path)
        time.sleep(20)

    jobspec = init_jobspec(nxyz=nxyz,
                           details=details,
                           charge=charge)
    jobspec = sf_nacv_jobspec(jobspec=jobspec,
                              singlet_path=singlet_path,
                              num_states=num_states)

    config_name = "bhhlyp_6-31gs_sf_tddft_nacv_qchem"
    p = run_job(config_name=config_name,
                jobspec=jobspec,
                job_dir=job_dir)

    return p


def run_sf(job_dir,
           nxyz,
           charge,
           num_states,
           grad_details,
           nacv_details,
           grad_config,
           nacv_config):

    procs = []
    if grad_config == 'bhhlyp_6-31gs_sf_tddft_engrad_qchem':
        p = bhhlyp_6_31gs_sf_tddft_engrad_qchem(details=grad_details,
                                                charge=charge,
                                                num_states=num_states,
                                                job_dir=job_dir)
        procs.append(p)

    else:
        raise NotImplementedError

    if nacv_config == "bhhlyp_6-31gs_sf_tddft_nacv_qchem":
        p = bhhlyp_6_31gs_sf_tddft_nacv_qchem(nxyz=nxyz,
                                              details=nacv_details,
                                              charge=charge,
                                              num_states=num_states,
                                              job_dir=job_dir)
        procs.append(p)
    else:
        raise NotImplementedError

    for p in procs:
        p.join()


def parse_sf_grads(job_dir):
    path = os.path.join(job_dir, 'singlet_grad.out')
    with open(path, 'r') as f:
        lines = f.readlines()
    output_dics = get_cis_grad(lines)

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
             nacv_config):

    grad_dics = parse_sf_grads(job_dir=job_dir)
    nacv_dic = parse_sf_nacv(job_dir=job_dir,
                             conifg_name=nacv_config)

    return grad_dics, nacv_dic


def grad_to_arr(results,
                grad_dics):

    combined_grad = {}
    for dic in grad_dics:
        combined_grad.update(dic)

    grad_states = sorted(list(combined_grad.keys()),
                         key=lambda x: int(x))

    for i, state in enumerate(grad_states):
        key = f'energy_{i}_grad'
        grad = np.array(combined_grad[state])
        shape = grad.shape
        results[key] = grad.reshape(1, *shape)

    return results


def nacv_to_arr(results,
                nacv_dic,
                singlets):

    translation = {"deriv_nacv_etf": 'nacv'}
    base_keys = ['deriv_nacv_etf', 'force_nacv']

    for base_key in base_keys:
        keys = [i for i in nacv_dic.keys()
                if base_key in i]

        for key in keys:
            start_state = int(key.split("_")[-2])
            end_state = int(key.split("_")[-1])

            singlet_start = singlets.index(start_state)
            singlet_end = singlets.index(end_state)

            translate_base = translation.get(key, key)
            results_key = f"{translate_base}_{singlet_start}_{singlet_end}"

            nacv = np.array(nacv_dic[key])
            shape = nacv.shape
            results[results_key] = nacv.reshape(1, *shape)

    return results


def combine_results(singlets,
                    grad_dics,
                    nacv_dic):

    results = {}
    results = grad_to_arr(results=results,
                          grad_dics=grad_dics)
    results = nacv_to_arr(results=results,
                          nacv_dic=nacv_dic,
                          singlets=singlets)

    return results


def parse(job_dir,
          grad_config,
          nacv_config):

    is_sf = check_sf(grad_config=grad_config,
                     nacv_config=nacv_config)

    if is_sf:
        grad_dics, nacv_dic = parse_sf(job_dir=job_dir,
                                       nacv_config=nacv_config)

        singlet_path = os.path.join(job_dir, 'singlets.json')
        with open(singlet_path, 'r') as f:
            singlets = json.load(f)

        results = combine_results(grad_dics=grad_dics,
                                  nacv_dic=nacv_dic,
                                  singlets=singlets)

    else:
        raise NotImplementedError

    return results


def get_results(nxyz,
                charge,
                old_U,
                num_states,
                surf,
                job_dir,
                grad_config,
                nacv_config,
                grad_details,
                nacv_details):

    is_sf = check_sf(grad_config=grad_config,
                     nacv_config=nacv_config)

    if is_sf:
        run_sf(job_dir=job_dir,
               nxyz=nxyz,
               charge=charge,
               num_states=num_states,
               grad_details=grad_details,
               nacv_details=nacv_details,
               grad_config=grad_config,
               nacv_config=nacv_config)

    else:
        raise NotImplementedError

    results = parse(job_dir=job_dir,
                    grad_config=grad_config,
                    nacv_config=nacv_config)

    return results
