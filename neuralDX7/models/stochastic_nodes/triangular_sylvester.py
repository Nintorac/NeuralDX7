#%%
from collections import namedtuple
from itertools import count
import torch
from torch import nn
from neuralDX7.models.stochastic_nodes import NormalNode

"""
This code modified from the reference implementation provided by the authors
https://github.com/riannevdberg/sylvester-flows
"""



class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.h = nn.Tanh()

        # diag_idx = torch.arange(0, z_size).long()
        # self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        # diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r1 = torch.diagonal(r1, 0, -1, -2)
        # diag_r2 = r2[:, self.diag_idx, self.diag_idx]
        diag_r2 = torch.diagonal(r2, 0, -1, -2)

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk

        r2qzb = z_per @ r2.transpose(2, 1) + b
        z = self.h(r2qzb) @ r1.transpose(2, 1)

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = (diag_j.abs()+1e-8).log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

class TriangularSylvesterFlow(nn.Module):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, in_features, latent_dim, num_flows):

        super().__init__()
        # Initialize log-det-jacobian to zero
        self.log_det_j = 0.

        # Flow parameters
        self.num_flows = num_flows
        self.latent_dim = latent_dim

        # permuting indices corresponding to Q=P (permutation matrix) for every other flow
        flip_idx = torch.arange(latent_dim - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        # self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * latent_dim)
        self.q_z = NormalNode(in_features, latent_dim)
        self._flow_params = nn.Linear(in_features,
                self.num_flows * latent_dim * latent_dim + \
                self.num_flows * latent_dim + \
                self.num_flows * latent_dim + \
                self.num_flows * latent_dim
        )
        self.flows = nn.ModuleList([
            TriangularSylvester(latent_dim) for k in range(self.num_flows)
        ])

    def flow_params(self, h):
        """
        Parameterise the base distribution, sample and flow
        """

        batch_size = h.size(0)

        params = self._flow_params(h)
        params = params.reshape(batch_size, self.num_flows, self.latent_dim, -1)
        params = params.transpose(0,1) # batch x flows x z x z  -> flows x batch x z x z
        
        diag1 = torch.tanh(params[...,0])
        diag2 = torch.tanh(params[...,1])
        b = params[...,2].unsqueeze(-2)
        full_d = params[...,3:]

        r1 = torch.triu(full_d, diagonal=1)
        r2 = torch.triu(full_d.transpose(-1, -2), diagonal=1)
        r1 = diag1.diag_embed(0) + r1
        r2 = diag2.diag_embed(0) + r2

        return r1, r2, b

    def forward(self, h, z=None, flow=True):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        Flow = namedtuple('Flow', ('q_z', 'log_det', 'z_0', 'z_k', 'flow'))

        q_z = self.q_z(h)
        z_0 = z_k = q_z.rsample() if z is None else z

        if not flow:
            return Flow(q_z, None, z_0, None)

        r1, r2, b = self.flow_params(h)

        # Sample z_0
        def flow_f(z_k):
            log_det_j = 0.

            # Normalizing flows
            for k, flow_k, r1_k, r2_k, b_k in zip(count(), self.flows, r1, r2, b):

                if k % 2 == 1:
                    # Alternate with reorderering z for triangular flow
                    permute_z = self.flip_idx
                else:
                    permute_z = None

                z_k, log_det_jacobian = flow_k(z_k, r1_k, r2_k, b_k, permute_z, sum_ldj=True)

                log_det_j += log_det_jacobian
            
            return z_k, log_det_j
        z_k, log_det_j = flow_f(z_0)
        return Flow(q_z, log_det_j, z_0, z_k, flow_f)


if __name__=="__main__":
    
    num_ortho_vecs = z_size = 6
    batch_size = 12
    in_features = 64
    
    h = torch.randn(batch_size, in_features)
    # zk = torch.randn(batch_size, z_size)
    # r1 = torch.randn(batch_size, num_ortho_vecs, num_ortho_vecs)
    # r2 = torch.randn(batch_size, num_ortho_vecs, num_ortho_vecs)
    # b = torch.randn(batch_size, 1, z_size)

    f = TriangularSylvesterFlow(in_features, z_size, 3)

    f(h)

# %%
