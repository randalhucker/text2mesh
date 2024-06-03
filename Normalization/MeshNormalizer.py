from mesh import Mesh
from . import Normalizer


class MeshNormalizer:
    """A class to normalize the vertices of a mesh to a unit sphere."""

    def __init__(self, mesh: Mesh):
        """A constructor for the MeshNormalizer class.

        Args:
            mesh (Mesh): The mesh to normalize.
        """
        self._mesh = mesh  # Original copy of the mesh
        self.normalizer = Normalizer.get_bounding_sphere_normalizer(
            self._mesh.vertices
        )  # Normalizer to normalize the vertices of the mesh

    def __call__(self):
        """A method to normalize the vertices of the mesh to a unit sphere.

        Returns:
            Mesh: The normalized mesh.
        """
        self._mesh.vertices = self.normalizer(
            self._mesh.vertices
        )  # Normalize the vertices of the mesh
        return self._mesh  # Return the normalized mesh
