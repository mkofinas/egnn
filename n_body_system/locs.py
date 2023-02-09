import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter


"""
Geometric primitives, rotations, cartesian to spherical
"""


def rotation_matrix(ndim, theta, phi=None, psi=None, /):
    """
    theta, phi, psi: yaw, pitch, roll

    NOTE: We assume that each angle is has the shape [dims] x 1
    """
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    if ndim == 2:
        R = torch.stack([torch.cat([cos_theta, -sin_theta], -1),
                         torch.cat([sin_theta, cos_theta], -1)], -2)
        return R
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    R = torch.stack([
            torch.cat([cos_phi * cos_theta, -sin_theta, sin_phi * cos_theta], -1),
            torch.cat([cos_phi * sin_theta, cos_theta, sin_phi * sin_theta], -1),
            torch.cat([-sin_phi, torch.zeros_like(cos_theta), cos_phi], -1)], -2)
    return R


def cart_to_n_spherical(x, symmetric_theta=False):
    """Transform Cartesian to n-Spherical Coordinates

    NOTE: Not tested thoroughly for n > 3

    Math convention, theta: azimuth angle, angle in x-y plane

    x: torch.Tensor, [dims] x D
    return rho, theta, phi
    """
    ndim = x.size(-1)

    rho = torch.norm(x, p=2, dim=-1, keepdim=True)

    theta = torch.atan2(x[..., [1]], x[..., [0]])
    if not symmetric_theta:
        theta = theta + (theta < 0).type_as(theta) * (2 * np.pi)

    if ndim == 2:
        return rho, theta

    cum_sqr = (rho if ndim == 3
               else torch.sqrt(torch.cumsum(torch.flip(x ** 2, [-1]), dim=-1))[..., 2:])
    EPS = 1e-7
    phi = torch.acos(
        torch.clamp(x[..., 2:] / (cum_sqr + EPS), min=-1.0, max=1.0)
    )

    return rho, theta, phi


def velocity_to_rotation_matrix(vel):
    num_dims = vel.size(-1)
    orientations = cart_to_n_spherical(vel)[1:]
    R = rotation_matrix(num_dims, *orientations)
    return R


def gram_schmidt(vel, acc):
    """Gram-Schmidt orthogonalization"""
    # normalize
    e1 = F.normalize(vel, dim=-1)
    # orthogonalize
    u2 = acc - torch.sum(e1 * acc, dim=-1, keepdim=True) * e1
    # normalize
    e2 = F.normalize(u2, dim=-1)
    # cross product
    e3 = torch.cross(e1, e2)

    frame1 = torch.stack([e1, e2, e3], dim=-1)
    return frame1


def rotation_matrix_to_euler(R, num_dims, normalize=True):
    """Convert rotation matrix to euler angles

    In 3 dimensions, we follow the ZYX convention
    """
    if num_dims == 2:
        euler = torch.atan2(R[..., 1, [0]], R[..., 0, [0]])
    else:
        euler = torch.stack([
            torch.atan2(R[..., 1, 0], R[..., 0, 0]),
            torch.asin(-R[..., 2, 0]),
            torch.atan2(R[..., 2, 1], R[..., 2, 2]),
        ], -1)

    if normalize:
        euler = euler / np.pi
    return euler


def rotate(x, R):
    return torch.einsum('...ij,...j->...i', R, x)


"""
Transformation from global to local coordinates and vice versa
"""


class Localizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def set_edge_index(self, send_edges, recv_edges):
        self.send_edges = send_edges
        self.recv_edges = recv_edges

    def sender_receiver_features(self, x):
        x_j = x[self.send_edges]
        x_i = x[self.recv_edges]
        return x_j, x_i

    def canonicalize_inputs(self, inputs):
        vel = inputs[..., self.num_dims:2*self.num_dims]
        # acc = inputs[..., 2*self.num_dims:]
        # R = gram_schmidt(vel, acc)
        R = velocity_to_rotation_matrix(vel)
        Rinv = R.transpose(-1, -2)

        canon_vel = rotate(vel, Rinv)
        canon_inputs = torch.cat([torch.zeros_like(canon_vel), canon_vel], dim=-1)

        return canon_inputs, R

    def create_edge_attr(self, x):
        x_j, x_i = self.sender_receiver_features(x)

        # R = gram_schmidt(x_i[..., self.num_dims:2*self.num_dims],
                         # x_i[..., 2*self.num_dims:])
        # We approximate orientations via the velocity vector
        R = velocity_to_rotation_matrix(x_i[..., self.num_dims:2*self.num_dims])
        R_inv = R.transpose(-1, -2)

        # Positions
        relative_positions = x_j[..., :self.num_dims] - x_i[..., :self.num_dims]
        rotated_relative_positions = rotate(relative_positions, R_inv)

        # Orientations
        # send_R = gram_schmidt(x_j[..., self.num_dims:2*self.num_dims],
                              # x_j[..., 2*self.num_dims:])
        send_R = velocity_to_rotation_matrix(x_j[..., self.num_dims:2*self.num_dims])
        rotated_orientations = R_inv @ send_R
        rotated_euler = rotation_matrix_to_euler(rotated_orientations, self.num_dims)

        # Rotated relative positions in spherical coordinates
        node_distance = torch.norm(relative_positions, p=2, dim=-1, keepdim=True)
        spherical_relative_positions = torch.cat(
            cart_to_n_spherical(rotated_relative_positions, symmetric_theta=True)[1:], -1)

        # Velocities
        rotated_velocities = rotate(x_j[..., self.num_dims:2*self.num_dims], R_inv)

        edge_attr = torch.cat([
            rotated_relative_positions,
            rotated_euler,
            node_distance,
            spherical_relative_positions,
            rotated_velocities,
        ], -1)
        return edge_attr

    def forward(self, x, edges):
        self.set_edge_index(*edges)
        rel_feat, R = self.canonicalize_inputs(x)
        edge_attr = self.create_edge_attr(x)

        edge_attr = torch.cat([edge_attr, rel_feat[self.recv_edges]], -1)
        return rel_feat, R, edge_attr


class Globalizer(nn.Module):
    def __init__(self, num_dims: int = 2):
        super().__init__()
        self.num_dims = num_dims

    def forward(self, x, R):
        return torch.cat(
            [rotate(xi, R) for xi in x.split(self.num_dims, dim=-1)], -1)


"""
LoCS Neural Network
"""
class LoCS(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims,
                 device='cuda', num_layers=4):
        super().__init__()
        self.gnn = GNN(input_size, hidden_size, dropout_prob, num_dims,
                       num_layers=num_layers)
        self.localizer = Localizer(num_dims)
        self.globalizer = Globalizer(num_dims)
        self.num_dims = num_dims
        self.to(device)

    def forward(self, h, x, edges, vel, edge_attr_orig):
        """inputs shape: [batch_size, num_objects, input_size]"""
        inputs = torch.cat([x, vel], dim=-1)
        # Global to Local
        rel_feat, Rinv, edge_attr = self.localizer(inputs, edges)
        edge_attr = torch.cat([edge_attr, edge_attr_orig], dim=-1)

        # GNN
        pred = self.gnn(rel_feat, edge_attr, edges)
        # Local to Global
        pred = self.globalizer(pred, Rinv)

        # Predict position/velocity difference and integrate
        outputs = x + pred
        return outputs


class GNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob, num_dims,
                 num_layers=4):
        super().__init__()
        self.num_dims = num_dims
        out_size = num_dims

        self.num_orientations = self.num_dims * (self.num_dims - 1) // 2
        # Relative features include: positions, orientations, positions in
        # spherical coordinates, and velocities
        self.num_relative_features = 3 * self.num_dims + self.num_orientations

        initial_edge_features = 2
        num_edge_features = self.num_relative_features + input_size + initial_edge_features

        self.gnn = nn.Sequential(
            GNNLayer(input_size, hidden_size, only_edge_attr=True,
                     num_edge_features=num_edge_features),
            *[GNNLayer(hidden_size, hidden_size) for _ in range(num_layers-1)],
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size * num_objects, input_size]
        """
        for layer in self.gnn:
            x, edge_attr = layer(x, edge_attr, edges)

        # Output MLP
        pred = self.out_mlp(x)
        return pred


class GNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, only_edge_attr=False, num_edge_features=0):
        super().__init__()

        # Neural Network Layers
        self.only_edge_attr = only_edge_attr
        num_edge_features = num_edge_features if only_edge_attr else 3 * hidden_size
        self.message_fn = nn.Sequential(
            nn.Linear(num_edge_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

        self.res = (nn.Linear(input_size, hidden_size) if input_size != hidden_size
                    else nn.Identity())

        self.update_fn = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.SiLU(),
            nn.Linear(2*hidden_size, hidden_size),
        )

    def forward(self, x, edge_attr, edges):
        """
        inputs shape: [batch_size, num_objects, input_size]
        """
        send_edges, recv_edges = edges
        if not self.only_edge_attr:
            edge_attr = torch.cat([x[send_edges], x[recv_edges], edge_attr], dim=-1)

        edge_attr = self.message_fn(edge_attr)

        x = self.res(x) + scatter(
            edge_attr, recv_edges.to(x.device), dim=0,
            reduce='mean').contiguous()

        x = x + self.update_fn(x)

        return x, edge_attr
