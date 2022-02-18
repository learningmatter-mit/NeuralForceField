import torch
import torch.nn as nn
from .spookynet import SpookyNet
from typing import List, Tuple, Optional


class SpookyNetEnsemble(nn.Module):
    """
    Ensemble of SpookyNet models.

    Arguments:
        models (list of str):
            File paths from which to load the parameters of the individual
            models (they are passed to the load_from argument of the SpookyNet
            class).
    """

    def __init__(self, models: List[str] = []) -> None:
        """ Initializes the SpookyNetEnsemble class. """
        super(SpookyNetEnsemble, self).__init__()
        assert len(models) > 1
        self.models = nn.ModuleList([SpookyNet(load_from=model) for model in models])
        for model in self.models:
            model.module_keep_prob = 1.0

    @property
    def dtype(self) -> torch.dtype:
        """ Return torch.dtype of parameters (input tensors must match). """
        return self.models[0].dtype

    @property
    def device(self) -> torch.device:
        """ Return torch.device of parameters (input tensors must match). """
        return self.models[0].device

    def train(self, mode=True) -> None:
        """
        Turn on training mode. This is just for compatibility, the models should
        be trained individually and only evaluated as ensemble.
        """
        super(SpookyNetEnsemble, self).train(mode=mode)
        for model in self.models:
            model.train(mode)

    def eval(self) -> None:
        """ Turn on evaluation mode (smaller memory footprint)."""
        super(SpookyNetEnsemble, self).eval()
        for model in self.models:
            model.eval()

    @torch.jit.export
    def atomic_properties(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Computes atomic properties by calling methods from the SpookyNet class.

        (see documentation of atomic_properties in SpookyNet class)
        """
        (
            N,
            cutoff_values,
            rij,
            sr_rij,
            pij,
            dij,
            sr_idx_i,
            sr_idx_j,
            mask,
        ) = self.models[0]._atomic_properties_static(
            Z=Z,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )
        (f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = ([], [], [], [], [], [], [], [])
        for model in self.models:
            (
                f_,
                ea_,
                qa_,
                ea_rep_,
                ea_ele_,
                ea_vdw_,
                pa_,
                c6_,
            ) = model._atomic_properties_dynamic(
                N=N,
                Q=Q,
                S=S,
                Z=Z,
                R=R,
                cutoff_values=cutoff_values,
                rij=rij,
                idx_i=idx_i,
                idx_j=idx_j,
                sr_rij=sr_rij,
                pij=pij,
                dij=dij,
                sr_idx_i=sr_idx_i,
                sr_idx_j=sr_idx_j,
                cell=cell,
                num_batch=num_batch,
                batch_seg=batch_seg,
                mask=mask,
            )
            f.append(f_)
            ea.append(ea_)
            qa.append(qa_)
            ea_rep.append(ea_rep_)
            ea_ele.append(ea_ele_)
            ea_vdw.append(ea_vdw_)
            pa.append(pa_)
            c6.append(c6_)
        return (f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    def _mean_std_from_list(
        self, x: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a list of tensors, computes their mean and standard deviation.
        Only used internally.
        """
        x = torch.stack(x)
        return (torch.mean(x, dim=0), torch.std(x, dim=0))

    @torch.jit.export
    def energy(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the potential energy.

        (see documentation of energy in SpookyNet class)

        Returns:
            Mean and standard deviation of the predictions of the ensemble.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = Z.new_zeros(Z.size(0))
        (f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.atomic_properties(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )
        energy = []
        for ea_, ea_rep_, ea_ele_, ea_vdw_ in zip(ea, ea_rep, ea_ele, ea_vdw):
            esum = ea_ + ea_rep_ + ea_ele_ + ea_vdw_
            energy.append(ea_.new_zeros(num_batch).index_add_(0, batch_seg, esum))
        energy = self._mean_std_from_list(energy)
        f = self._mean_std_from_list(f)
        ea = self._mean_std_from_list(ea)
        qa = self._mean_std_from_list(qa)
        ea_rep = self._mean_std_from_list(ea_rep)
        ea_ele = self._mean_std_from_list(ea_ele)
        ea_vdw = self._mean_std_from_list(ea_vdw)
        pa = self._mean_std_from_list(pa)
        c6 = self._mean_std_from_list(c6)
        return (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    @torch.jit.export
    def energy_and_forces(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        create_graph: bool = False,
        calculate_forces_std: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the potential energy and forces.

        (see documentation of energy_and_forces in SpookyNet class)

        Returns:
            Mean and standard deviation of the predictions of the ensemble.
        """
        (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.energy(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
        )

        # calculating the derivative of std gives the same as first taking the
        # derivative and then doing std but the values may be negative => abs
        # fixes this
        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad_mean = torch.autograd.grad(
                [torch.sum(energy[0])],
                [R],
                retain_graph=True,
                create_graph=create_graph,
            )[0]
            if grad_mean is not None:  # necessary for torch.jit compatibility
                forces_mean = -grad_mean
            else:
                forces_mean = torch.zeros_like(R)
            if calculate_forces_std:
                grad_std = torch.autograd.grad(
                    [torch.sum(energy[1])], [R], create_graph=create_graph
                )[0]
                if grad_std is not None:  # necessary for torch.jit compatibility
                    forces_std = torch.abs(grad_std)
                else:
                    forces_std = torch.zeros_like(R)
            else:
                forces_std = torch.zeros_like(R)
        else:
            forces_mean = torch.zeros_like(R)
            forces_std = torch.zeros_like(R)
        forces = (forces_mean, forces_std)
        return (energy, forces, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    @torch.jit.export
    def energy_and_forces_and_hessian(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        calculate_forces_std: bool = False,
        calculate_hessian_std: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the potential energy, forces and the hessian.

        (see documentation of energy_and_forces_and_hessian in SpookyNet class)

        Returns:
            Mean and standard deviation of the predictions of the ensemble.
        """
        (
            energy,
            forces,
            f,
            ea,
            qa,
            ea_rep,
            ea_ele,
            ea_vdw,
            pa,
            c6,
        ) = self.energy_and_forces(
            Z=Z,
            Q=Q,
            S=S,
            R=R,
            idx_i=idx_i,
            idx_j=idx_j,
            cell=cell,
            cell_offsets=cell_offsets,
            num_batch=num_batch,
            batch_seg=batch_seg,
            create_graph=True,
            calculate_forces_std=calculate_forces_std,
        )

        grad_mean = -forces[0].view(-1)
        grad_std = -forces[1].view(-1)
        s = grad_mean.size(0)
        hessian_mean = energy[0].new_zeros((s, s))
        hessian_std = energy[1].new_zeros((s, s))
        if idx_i.numel() > 0:
            for idx in range(s):  # loop through entries of the hessian
                # retain graph when the index is smaller than the max index,
                # else computation fails
                tmp = torch.autograd.grad(
                    [grad_mean[idx]], [R], retain_graph=(idx < s)
                )[0]
                if tmp is not None:  # necessary for torch.jit compatibility
                    hessian_mean[idx] = tmp.view(-1)
                if calculate_hessian_std and calculate_forces_std:
                    tmp = torch.autograd.grad(
                        [grad_std[idx]], [R], retain_graph=(idx < s)
                    )[0]
                    if tmp is not None:  # necessary for torch.jit compatibility
                        hessian_std[idx] = tmp.view(-1)
        hessian = (hessian_mean, hessian_std)
        return (energy, forces, hessian, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)

    def forward(
        self,
        Z: torch.Tensor,
        Q: torch.Tensor,
        S: torch.Tensor,
        R: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        num_batch: int = 1,
        batch_seg: Optional[torch.Tensor] = None,
        create_graph: bool = True,
        use_forces: bool = True,
        use_dipole: bool = True,
        calculate_forces_std: bool = False,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Computes the total energy, forces, dipole moment vectors, and all atomic
        properties.

        (see documentation of forward in SpookyNet class)

        Returns:
            Mean and standard deviation of the predictions of the ensemble.
        """
        if batch_seg is None:
            batch_seg = Z.new_zeros(Z.size(0))
        if use_forces:
            (
                energy,
                forces,
                f,
                ea,
                qa,
                ea_rep,
                ea_ele,
                ea_vdw,
                pa,
                c6,
            ) = self.energy_and_forces(
                Z=Z,
                Q=Q,
                S=S,
                R=R,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offsets=cell_offsets,
                num_batch=num_batch,
                batch_seg=batch_seg,
                create_graph=create_graph,
                calculate_forces_std=calculate_forces_std,
            )
        else:
            (energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6) = self.energy(
                Z=Z,
                Q=Q,
                S=S,
                R=R,
                idx_i=idx_i,
                idx_j=idx_j,
                cell=cell,
                cell_offsets=cell_offsets,
                num_batch=num_batch,
                batch_seg=batch_seg,
            )
            forces = (torch.zeros_like(R), torch.zeros_like(R))
        if use_dipole:
            dipole_mean = (
                qa[0]
                .new_zeros((num_batch, 3))
                .index_add_(0, batch_seg, qa[0].view(-1, 1) * R)
            )
            dipole_std = (
                qa[1]
                .new_zeros((num_batch, 3))
                .index_add_(0, batch_seg, qa[1].view(-1, 1) * R)
            )
            dipole = (dipole_mean, torch.abs(dipole_std))
        else:
            dipole = (qa[0].new_zeros((num_batch, 3)), qa[1].new_zeros((num_batch, 3)))

        return (energy, forces, dipole, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6)
