from typing import Dict, Union, List, Tuple

import torch
import numpy as np
import networkx as nx
import trimesh
from trimesh.creation import icosphere
from scipy.spatial.transform import Rotation as Rot
from pytransform3d.rotations import matrix_from_axis_angle


def get_affine(data):
    """
    Get or construct the affine matrix of the image, it can be used to correct
    spacing, orientation or execute spatial transforms.
    Args:
    data: an ITK image object loaded from an image file or props dictionary.

    """
    if isinstance(data, Dict):
        direction = np.array(data["sitk_stuff"]["direction"]).reshape(3, 3)
        spacing = np.array(data["sitk_stuff"]["spacing"])
        origin = np.array(data["sitk_stuff"]["origin"])

    else:
        direction = np.array(data.GetDirection()).reshape(3, 3)
        spacing = np.asarray(data.GetSpacing())
        origin = np.asarray(data.GetOrigin())

    sr = min(max(direction.shape[0], 1), 3)
    affine: np.ndarray = np.eye(sr + 1)
    affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
    affine[:sr, -1] = origin[:sr]
    return torch.from_numpy(affine).float(), torch.from_numpy(spacing).float()


def cart2spher(cart_coords: Union[torch.Tensor, np.ndarray]):
    """ Transform an Nx3 matrix of (unit length) Cartesian coordinates
    into an Nx3 matrix with [r, \u03D5, \u03B8] spherical coordinates.
    normalize cartesian coordinates. Note that r=1, \u03D5 in [0, pi],
    \u03B8 in [0, 2pi]. """

    if torch.is_tensor(cart_coords):
        cart_coords /= torch.linalg.norm(cart_coords, dim=1, keepdim=True)
        coords_spherical = torch.ones_like(cart_coords)
        # theta = arctan(y/x), r=1, range: [-pi, pi]
        theta = torch.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        # phi = arccos(z), r=1, range [0, pi]
        phi = torch.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).long()  # theta range [0, 2 * pi]
    else:
        cart_coords = cart_coords / \
            np.expand_dims(np.linalg.norm(cart_coords, axis=1), 1)
        coords_spherical = np.ones_like(cart_coords)
        theta = np.arctan2(cart_coords[:, 1], cart_coords[:, 0])
        phi = np.arccos(cart_coords[:, 2])
        theta += np.pi * 2 * (theta < 0).astype(int)

    coords_spherical[:, 1] = phi
    coords_spherical[:, 2] = theta
    return coords_spherical


def spher2cart(spher_coords: Union[torch.Tensor, np.ndarray]):
    """ Transform an N*3 matrix of (unit radius) Spherical coordinates into an
    N*3 matrix with [x, y, z] Cartesian coordinates. """

    if isinstance(spher_coords, torch.Tensor):
        cart_coords = torch.zeros_like(spher_coords)
        x = spher_coords[:, 0] * \
            torch.sin(spher_coords[:, 1]) * torch.cos(spher_coords[:, 2])
        y = spher_coords[:, 0] * \
            torch.sin(spher_coords[:, 1]) * torch.sin(spher_coords[:, 2])
        z = spher_coords[:, 0] * torch.cos(spher_coords[:, 1])

    else:
        cart_coords = np.zeros_like(spher_coords)
        x = spher_coords[:, 0] * \
            np.sin(spher_coords[:, 1]) * np.cos(spher_coords[:, 2])
        y = spher_coords[:, 0] * \
            np.sin(spher_coords[:, 1]) * np.sin(spher_coords[:, 2])
        z = spher_coords[:, 0] * np.cos(spher_coords[:, 1])

    cart_coords[:, 0] = x
    cart_coords[:, 1] = y
    cart_coords[:, 2] = z
    return cart_coords


def transform_points(
    points: torch.Tensor | np.ndarray,
    affine: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:

    if isinstance(points, torch.Tensor) and isinstance(affine, torch.Tensor):
        points = torch.cat(
            [points, torch.ones([points.shape[0], 1]).to(points)], dim=1).T
        return (affine.float() @ points.float()).T[:, :-1]

    elif isinstance(points, np.ndarray) and isinstance(affine, np.ndarray):
        points = np.concatenate(
            [points, np.ones([points.shape[0], 1])], axis=1).T
        return (affine @ points).T[:, :-1]


def get_rotation_matrix(vector: torch.Tensor):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    Adapted from: https://stackoverflow.com/questions/63287960/python-rotate-plane-set-of-points-to-match-new-normal-vector-using-scipy-spat
    """
    vector = vector / torch.linalg.norm(vector)
    axis = np.cross(np.array([0, 0, 1]), vector.numpy())

    # determine angle between new normal and z-axis
    dot_product = np.dot(np.array([0, 0, 1]), vector.numpy())
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)
    M = Rot.from_matrix(R).as_matrix()
    return torch.from_numpy(M).float()


def nearestPointsIndices3D(src: torch.Tensor, tar: torch.Tensor) -> Tuple[List, List]:
    """ Compute the indices of points on tar which is nearest to points in src.
    Both src and tar should be 3D position tensor with shape [K, 3]
     """

    if src.shape[-1] != tar.shape[-1]:
        raise ValueError("Dimension mismatch.")
    if src.shape[-1] != 3:
        raise ValueError("Input points should be 3D.")

    src_expand = src.unsqueeze(1)  # [K1, 1, 3]
    tar_expand = tar.unsqueeze(0)  # [1, K2, 3]
    dist = torch.linalg.norm(
        (src_expand - tar_expand).float(),
        ord=2,
        dim=-1,
        keepdim=False
    )  # [K1, K2]
    inds = torch.argmin(dist, dim=-1, keepdim=True)
    min_dists = torch.min(dist, dim=-1, keepdim=True).values
    inds = inds.squeeze(-1).tolist()
    min_dists = min_dists.squeeze(-1).tolist()

    return inds, min_dists


class IcoSphere(object):
    """ This is the sphere object, that is used for obtaining the right data for the GEM-CNN
    also contains the image/affine/centerline/annotations of the patient. """

    def __init__(self, subdivisions: int = 3):  # nverts = 642, 162, 42

        self.sphere = icosphere(subdivisions=subdivisions)  # r=1
        self.sphereverts = cart2spher(
            self.sphere.vertices)  # Spherical coordinates
        self.cartverts = self.sphere.vertices  # Cartesian coordinates

    def get_rays(self, npoints, ray_length, center=np.array([[0, 0, 0]])):
        """
        transform the Nx3 matrix containing the spherical coordinates of the vertices into image coordinates
        of all the points on the rays.

        Args:
            npoints: number of points on the ray
            ray_length:  real-world length of ray (in mms/cms)
            center: center of sphere, in world-coordinates

        Returns: (Nxraylength) x 3 matrix containing cartesian coordinates

        """

        rays = np.linspace(0, ray_length, npoints)
        sphereverts_long = self.sphereverts.repeat(npoints, axis=0)

        # Num of points on single sphere * number of spheres as 1st dim
        cart_coords = np.ones([self.sphereverts.shape[0] * npoints, 3])
        cart_coords[:, 0] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(
                sphereverts_long[:, 1]) * np.cos(sphereverts_long[:, 2])
        )  # x = r * sin(phi) * cos(theta)
        cart_coords[:, 1] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(
                sphereverts_long[:, 1]) * np.sin(sphereverts_long[:, 2])
        )  # y = r * sin(phi) * sin(theta)
        cart_coords[:, 2] = np.tile(
            rays, self.sphereverts.shape[0]) * np.cos(sphereverts_long[:, 1])  # z = r * cos(phi)

        return cart_coords + center  # [x, y, z] in world-coordinates

    def make_heatmap(self, directions, alpha, r):
        """
        make a discrete heatmap for phi,thetas given the objective tangent
        Args:
            directions: objective directions in spherical coordinates
            alpha: e^alpha*t (as in Sironi et al.)
            r: radius of Gaussian peak

        Returns: n_verts * 1 array, indicating objective directions as a Gaussian function on the sphere
         use 'continuous' sense of direction so true value of peak might differ!
        """
        heatmap = np.zeros([len(directions), self.sphereverts.shape[0]])
        for i, direction in enumerate(directions):
            great_circle_dist = np.arccos(
                np.sin(self.sphereverts[:, 1] - np.pi /
                       2) * np.sin(direction[1] - np.pi / 2)
                + np.cos(self.sphereverts[:, 1] - np.pi / 2)
                * np.cos(direction[1] - np.pi / 2)
                * np.cos(np.abs(self.sphereverts[:, 2] - direction[2]))
            )

            heatmap[i, :] = np.clip(
                (np.exp(alpha * (1 - great_circle_dist / r)) - 1) *
                (great_circle_dist < r).astype(int),
                0,
                np.exp(alpha),
            )
        return np.expand_dims(np.max(heatmap, axis=0), 1)


class IHPSphere(object):
    """ This is the sphere object that is used for obtaining image data for
    GCN, based on Icosahedral Hexagonal-Pentagonal Grids. """

    def __init__(self, subdivision: int = 0):

        self.icosphere = icosphere(subdivisions=subdivision)
        self.ihp: nx.Graph = self.icosphere_dual(self.icosphere)
        self.cartverts = np.array([
            self.ihp.nodes[n]["pos"]
            for n in sorted(list(self.ihp.nodes))
        ]).reshape(-1, 3)
        self.sphereverts = cart2spher(self.cartverts.copy())

    @staticmethod
    def icosphere_dual(ico_sphere: trimesh.Trimesh) -> nx.Graph:

        # New nodes are triangle centers in IcoSphere. New edges are face
        # adjacencies in IcoSphere.
        centers: np.ndarray = ico_sphere.triangles_center  # [N, 3]
        edges: np.ndarray = ico_sphere.face_adjacency  # [M, 2]

        g = nx.Graph()
        for idx, center in enumerate(centers):
            center = center / np.linalg.norm(center)
            g.add_node(idx, pos=center.tolist())
        g.add_edges_from([tuple(edge) for edge in edges.tolist()])

        return g

    def get_rays(self, npoints, ray_length, center: np.ndarray = np.array([[0, 0, 0]])):
        """
        transform the Nx3 matrix containing the spherical coordinates of the vertices into image coordinates
        of all the points on the rays.

        Args:
            npoints: number of points on the ray
            ray_length:  real-world length of ray (in mms/cms)
            center: center of sphere, in world-coordinates

        Returns: (Nxraylength) x 3 matrix containing cartesian coordinates

        """

        rays = np.linspace(0, ray_length, npoints)
        sphereverts_long = self.sphereverts.repeat(npoints, axis=0)

        # Num of points on single sphere * number of spheres as 1st dim
        cart_coords = np.ones([self.sphereverts.shape[0] * npoints, 3])
        cart_coords[:, 0] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(
                sphereverts_long[:, 1]) * np.cos(sphereverts_long[:, 2])
        )  # x = r * sin(phi) * cos(theta)
        cart_coords[:, 1] = (
            np.tile(rays, self.sphereverts.shape[0]) * np.sin(
                sphereverts_long[:, 1]) * np.sin(sphereverts_long[:, 2])
        )  # y = r * sin(phi) * sin(theta)
        cart_coords[:, 2] = np.tile(
            rays, self.sphereverts.shape[0]) * np.cos(sphereverts_long[:, 1])  # z = r * cos(phi)

        return cart_coords + center  # [x, y, z] in world-coordinates

    @property
    def graph(self) -> nx.Graph:

        return self.ihp.copy()

    @property
    def edge_index(self) -> np.ndarray:
        # NOTE: For undirected graph edge_index in torch_geometric, the edge
        # indices matrices must contain both directions.
        undirected_edges = np.array(list(self.ihp.edges())).T
        edge_index = np.concatenate(
            [undirected_edges, undirected_edges[[1, 0]]], axis=1)

        return edge_index


class RectanglePatch(object):
    """
    A geometrical object representing a rectangular 3D image patch.
    """

    def __init__(
        self,
        patch_size: List[int],
        target_spacing: List[float],
    ):

        self.d, self.w, self.h = patch_size
        self.sd, self.sh, self.sw = target_spacing

    def create_patch(
        self,
        center=torch.tensor([[0, 0, 0]]),
        rotMatrix=torch.eye(4, dtype=torch.float32),
    ):
        """
        Create patch coordinates for sampling. The patch is zero-centered.
        """

        patch_size = (self.d, self.h, self.w)  # (D, H, W)
        inds = np.unravel_index(np.arange(np.prod(patch_size)), patch_size)

        ds = (inds[0] - (self.d - 1) / 2) * self.sd
        hs = (inds[1] - (self.h - 1) / 2) * self.sh
        ws = (inds[2] - (self.w - 1) / 2) * self.sw

        phy_coords = np.concatenate(  # (N, 3)
            (ds.reshape(-1, 1), hs.reshape(-1, 1), ws.reshape(-1, 1)), axis=-1)
        phy_coords = torch.from_numpy(phy_coords).to(center.device)

        # Rotate the coordinates
        phy_coords = transform_points(phy_coords.view(-1, 3), rotMatrix)

        return center + phy_coords


class FibonacciSphere(object):
    """
    A zero-centered sphere with its coordinates drawn using Fibonacci sequence.
    """

    def __init__(self, npoints: int, r: float = 1.0):

        self.npoints = npoints
        self.r = r

    def get_coords(self):

        golden_ratio = (1 + 5 ** 0.5) / 2
        i = np.arange(0, self.npoints)
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * i / self.npoints)

        x = self.r * np.cos(theta) * np.sin(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(phi)

        sphere = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=-1)
        return sphere
