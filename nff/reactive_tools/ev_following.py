import torch

from nff.io.ase import NeuralFF, AtomsBatch
from nff.reactive_tools.utils import (neural_hessian_ase, neural_energy_ase,
                                      neural_force_ase)
from nff.utils.constants import EV_TO_AU, BOHR_RADIUS
from barriers.utils.vib import hessian_and_modes
from ase.units import Bohr, mol, kcal
from ase import Atoms

CONVG_LINE = "Optimization converged!"


def get_hessian(atoms,
                device):

    mode_results = hessian_and_modes(atoms)

    # use the full Hessian, not the one with
    # translation and rotation projected out

    # hessian = mode_results["hess_proj"]
    hessian = mode_results["hessianmatrix"]
    hessian = torch.Tensor(hessian).to(device)
    hessian /= (EV_TO_AU * BOHR_RADIUS ** 2)

    return hessian


def powell_update(hessian_old,
                  h,
                  gradient_old,
                  gradient_new):

    V = gradient_new - gradient_old - torch.mv(hessian_old, h)
    update = (
        torch.mm(V.reshape(-1, 1),
                 h.reshape(1, -1)) +
        torch.mm(h.reshape(-1, 1),
                 V.reshape(1, -1)) -
        (torch.dot(V, h) / torch.dot(h, h) *
         torch.mm(h.reshape(-1, 1), h.reshape(1, -1)))
    ) / torch.dot(h, h)

    powell_hessian = hessian_old + update

    return powell_hessian.detach()


def eigvec_following(ev_atoms,
                     step,
                     maxstepsize,
                     device,
                     method,
                     hessian_old=None,
                     gradient_old=None,
                     h_old=None):

    Ndim = 3
    old_xyz = torch.Tensor(ev_atoms.get_positions()
                           ).reshape(1, -1, Ndim).to(device)

    if(method == 'NR' or step == 0):
        hessian = get_hessian(atoms=ev_atoms,
                              device=device)

    neural_energy_ase(ev_atoms)
    neural_gradient = -1 * torch.Tensor(neural_force_ase(ev_atoms)).to(device)
    grad = neural_gradient.reshape(-1)

    if (method == 'Powell' and step != 0):
        hessian = powell_update(hessian_old, h_old, gradient_old, grad)

    # eigenvectors are stored in a transposed form
    eigenvalues, eigvecs = torch.eig(hessian, eigenvectors=True)

    # Ordering eigenvalues and eigenvectors in ascending order
    idx = eigenvalues[:, 0].argsort()
    eigenvalues = eigenvalues[idx][:, 0]
    # print(eigenvalues)

    eigvecs = eigvecs[:, idx]
    eigvecs_t = torch.t(eigvecs)
    # print(eigvecs_t)

    F = torch.mv(eigvecs_t, grad).reshape(-1, 1)

    matrix_p = torch.Tensor([[eigenvalues[0], F[0]], [F[0], 0]])
    lambda_p = torch.symeig(matrix_p, eigenvectors=False)[0][1]

    matrix_n = torch.zeros(Ndim * len(old_xyz[0]),
                           Ndim * len(old_xyz[0]))

    for i in range(Ndim * len(old_xyz[0]) - 1):
        matrix_n[i][i] = eigenvalues[i + 1]
        matrix_n[i][Ndim * len(old_xyz[0]) - 1] = F[i + 1]
        matrix_n[Ndim * len(old_xyz[0]) - 1][i] = F[i + 1]

    lambda_n = torch.symeig(matrix_n, eigenvectors=False)[0][0]

    lambda_n = lambda_n.new_full(
        (Ndim * len(old_xyz[0]) - 1,), lambda_n.item()).to(device)

    h_p = -1.0 * F[0] * eigvecs_t[0] / (eigenvalues[0] - lambda_p)
    h_n = -1.0 * F[1:] * eigvecs_t[1:] / \
        ((eigenvalues[1:] - lambda_n).reshape(-1, 1))

    h = torch.add(h_p, torch.sum(h_n, dim=0)
                  ).reshape(-1, len(old_xyz[0]), Ndim)

    step_size = h.norm()
    if step_size <= maxstepsize:
        new_xyz = old_xyz + h
    else:
        new_xyz = old_xyz + (h / (step_size / maxstepsize))

    output = (new_xyz.detach(), grad.detach(),
              hessian.detach(), h.reshape(-1).detach())

    return output


def get_calc_kwargs(calc_kwargs,
                    device,
                    nff_dir):

    if calc_kwargs is None:
        calc_kwargs = {}
    if device is not None:
        calc_kwargs["device"] = device
    if nff_dir is not None:
        calc_kwargs['model_path'] = nff_dir

    return calc_kwargs


def ev_run(ev_atoms,
           nff_dir=None,
           maxstepsize=0.15,
           maxstep=1000,
           convergence=0.03,
           device=None,
           method='Powell',
           calc_kwargs=None,
           nbr_update_period=1):

    rmslist = []
    maxlist = []

    calc_kwargs = get_calc_kwargs(calc_kwargs=calc_kwargs,
                                  device=device,
                                  nff_dir=nff_dir)
    nff = NeuralFF.from_file(**calc_kwargs)
    ev_atoms.set_calculator(nff)

    for step in range(maxstep):

        if step % nbr_update_period == 0:
            ev_atoms.update_nbr_list()

        if step == 0:
            args = []
        else:
            args = [hessian, grad, h]

        xyz, grad, hessian, h = eigvec_following(ev_atoms,
                                                 step,
                                                 maxstepsize,
                                                 device,
                                                 method,
                                                 *args)

        if step == 0:
            xyz_all = xyz
        else:
            xyz_all = torch.cat((xyz_all, xyz), dim=0)

        rmslist.append(grad.pow(2).sqrt().mean())
        maxlist.append(grad.pow(2).sqrt().max())
        print("RMS: {}, MAX: {}".format(
            grad.pow(2).sqrt().mean(), grad.pow(2).sqrt().max()))

        if grad.pow(2).sqrt().max() < convergence:
            print(CONVG_LINE)
            break

        positions = xyz.reshape(-1, 3).cpu().numpy()
        ev_atoms.set_positions(positions)

    # so that we're returning the xyz that has gradient `grad`, not whatever
    # the xyz of the next step would be
    xyz = torch.Tensor(ev_atoms.get_positions()).reshape(1, -1, 3)

    output = xyz, grad, xyz_all, rmslist, maxlist

    return output
