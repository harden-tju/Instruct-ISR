```python
import numpy as np
from scipy.spatial import cKDTree


class ImplicitFeatureEncoding:
    """
    Implicit Feature Encoding (IFE) for point clouds.

    This implementation follows the original C++ implementation:
    - Radius-based neighborhood search
    - Local canonicalization using SVD
    - Optional fixed-Z-axis canonicalization
    - Nearest-neighbor distance encoding
    - Distance normalization by the sample sphere radius

    Parameters
    ----------
    neighbor_radius : float
        Radius used for local neighborhood searching.
    fix_z_axis : bool
        If True, only the XY plane is canonicalized while the Z axis
        remains fixed. If False, full 3D canonicalization is performed.
    """

    def __init__(self, neighbor_radius=0.15, fix_z_axis=True):
        self.cloud_input = None
        self.sample_sphere = None

        self.neighbor_radius = neighbor_radius
        self.sample_radius = None
        self.sample_size = None
        self.fix_z_axis = fix_z_axis

        self.kdtree = None

    def set_input_cloud(self, cloud):
        """
        Set the input point cloud.

        Parameters
        ----------
        cloud : ndarray of shape (N, 3)
            Input point cloud.
        """
        cloud = np.asarray(cloud, dtype=np.float32)

        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("Input cloud must have shape (N, 3).")

        self.cloud_input = cloud
        self.kdtree = cKDTree(cloud)

    def set_sample_sphere(self, sample_sphere):
        """
        Set the predefined sample sphere.

        Parameters
        ----------
        sample_sphere : ndarray of shape (M, 3)
            Sample points centered at the origin.
        """
        sample_sphere = np.asarray(sample_sphere, dtype=np.float32)

        if sample_sphere.ndim != 2 or sample_sphere.shape[1] != 3:
            raise ValueError("Sample sphere must have shape (M, 3).")

        self.sample_sphere = sample_sphere
        self.sample_size = len(sample_sphere)

        squared_distances = np.sum(sample_sphere ** 2, axis=1)
        self.sample_radius = np.sqrt(np.max(squared_distances))

    def set_neighbor_radius(self, neighbor_radius):
        """Set the neighborhood search radius."""
        self.neighbor_radius = float(neighbor_radius)

    def fix_z_axis_enabled(self, fix):
        """Enable or disable fixed-Z-axis canonicalization."""
        self.fix_z_axis = bool(fix)

    def get_neighbor_radius(self):
        return self.neighbor_radius

    def get_sample_radius(self):
        return self.sample_radius

    def get_sample_size(self):
        return self.sample_size

    def _canonicalize_sample_sphere(self, neighbors, query_index):
        """
        Canonicalize the sample sphere according to the local geometry.

        This corresponds to canonicalizeSampleSphere() in the original
        C++ implementation.
        """
        query_point = self.cloud_input[query_index]

        # ------------------------------------------------------------
        # Fixed-Z-axis mode
        # ------------------------------------------------------------
        if self.fix_z_axis:

            local_2d = (
                self.cloud_input[neighbors, :2]
                - query_point[:2]
            )

            # Highest point is used as the orientation anchor
            highest_idx = neighbors[
                np.argmax(self.cloud_input[neighbors, 2])
            ]

            anchor = (
                self.cloud_input[highest_idx, :2]
                - query_point[:2]
            )

            # 2D SVD-based canonicalization
            _, _, vh = np.linalg.svd(
                local_2d,
                full_matrices=False
            )

            v_2d = vh.T

            # Flip axes toward the anchor
            for axis in range(2):
                if np.dot(anchor, v_2d[:, axis]) < 0:
                    v_2d[:, axis] *= -1

            # Construct 3D transformation matrix
            v_3d = np.eye(3, dtype=np.float32)
            v_3d[:2, :2] = v_2d

        # ------------------------------------------------------------
        # Full 3D canonicalization
        # ------------------------------------------------------------
        else:

            local_3d = (
                self.cloud_input[neighbors]
                - query_point
            )

            # Farthest point is used as orientation anchor
            distances = np.sum(local_3d ** 2, axis=1)
            farthest_idx = neighbors[np.argmax(distances)]

            anchor = (
                self.cloud_input[farthest_idx]
                - query_point
            )

            # 3D SVD-based canonicalization
            _, _, vh = np.linalg.svd(
                local_3d,
                full_matrices=False
            )

            v_3d = vh.T

            # Flip axes toward the anchor
            for axis in range(3):
                if np.dot(anchor, v_3d[:, axis]) < 0:
                    v_3d[:, axis] *= -1

        # ------------------------------------------------------------
        # Transform sample sphere to canonical pose
        #
        # Original C++:
        # pt_vector * v_3d.transpose()
        #
        # For row vectors in NumPy:
        # pt_vector @ v_3d.T
        # ------------------------------------------------------------
        canonical_sample = (
            self.sample_sphere @ v_3d.T
        )

        return canonical_sample

    def calculate_implicit_features(self):
        """
        Calculate implicit features for every point.

        Returns
        -------
        implicit_features : ndarray of shape (N, M)
            M-dimensional implicit feature vector for each input point.
        """

        if self.cloud_input is None:
            raise RuntimeError("Input cloud has not been set.")

        if self.sample_sphere is None:
            raise RuntimeError("Sample sphere has not been set.")

        if self.kdtree is None:
            self.kdtree = cKDTree(self.cloud_input)

        num_points = len(self.cloud_input)
        implicit_features = np.zeros(
            (num_points, self.sample_size),
            dtype=np.float32
        )

        # Radius-based neighborhood search
        neighborhoods = self.kdtree.query_ball_point(
            self.cloud_input,
            r=self.neighbor_radius
        )

        for i in range(num_points):

            neighbors = np.asarray(
                neighborhoods[i],
                dtype=np.int64
            )

            # --------------------------------------------------------
            # Local canonicalization
            # --------------------------------------------------------
            #
            # The original implementation uses the sample sphere
            # directly when fewer than four neighboring points exist.
            # Note that the query point itself is included.
            # --------------------------------------------------------
            if len(neighbors) < 4:
                canonical_sample = self.sample_sphere.copy()
            else:
                canonical_sample = (
                    self._canonicalize_sample_sphere(
                        neighbors,
                        i
                    )
                )

            # --------------------------------------------------------
            # Align sample sphere with the query point
            # --------------------------------------------------------
            sample_points = (
                canonical_sample
                + self.cloud_input[i]
            )

            # --------------------------------------------------------
            # Nearest-neighbor distance encoding
            # --------------------------------------------------------
            distances, _ = self.kdtree.query(
                sample_points,
                k=1
            )

            # Normalize by sample sphere radius
            implicit_features[i] = (
                distances / self.sample_radius
            )

        return implicit_features
```
