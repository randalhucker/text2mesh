import copy

import numpy as np
import PIL
import torch

import kaolin as kal

import utils
from utils import device


class Mesh:
    def __init__(self, obj_path: str, color=torch.tensor([0.0, 0.0, 1.0])):
        """A constructor for the Mesh class. Initializes the mesh object by importing the mesh from the given file path.

        Args:
            obj_path (str): The path to the mesh file.
            color (torch.Tensor, optional): The color of the mesh. Defaults to torch.tensor([0.0, 0.0, 1.0]).

        Raises:
            ValueError: If the extension of the mesh file is not implemented.
        """
        # Initialize the Mesh object by importing the mesh from the given file path
        if ".obj" in obj_path:
            mesh = kal.io.obj.import_mesh(obj_path, with_normals=True)
        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")

        # Store the mesh vertices and faces on the GPU
        self.vertices: torch.Tensor = mesh.vertices.to(device)
        self.faces: torch.Tensor = mesh.faces.to(device)

        self.vertex_normals: torch.Tensor | None = None
        self.face_normals: torch.Tensor | None = None
        self.texture_map: torch.Tensor | None = None
        self.face_uvs: torch.Tensor | None = None

        if ".obj" in obj_path:
            # if mesh.uvs.numel() > 0:
            #     uvs = mesh.uvs.unsqueeze(0).to(device)
            #     face_uvs_idx = mesh.face_uvs_idx.to(device)
            #     self.face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()

            # Handle the case where the mesh has vertex normals
            if mesh.vertex_normals is not None:
                # Store the vertex normals on the GPU
                self.vertex_normals = mesh.vertex_normals.to(device).float()

                # Normalize the vertex normals
                self.vertex_normals = torch.nn.functional.normalize(self.vertex_normals)

            # Handle the case where the mesh has face normals
            if mesh.face_normals is not None:
                # Store the face normals on the GPU
                self.face_normals = mesh.face_normals.to(device).float()

                # Normalize the face normals
                self.face_normals = torch.nn.functional.normalize(self.face_normals)

        # Set the initial color of the mesh
        self.set_mesh_color(color)

    def standardize_mesh(self, inplace=False) -> "Mesh":
        """A method to standardize the mesh by centering the vertices and scaling the mesh to fit in a unit cube.

        Args:
            inplace (bool, optional): A flag to indicate whether to standardize the mesh in place. Defaults to False.

        Returns:
            Mesh: The standardized mesh.
        """
        mesh = self if inplace else copy.deepcopy(self)
        return utils.standardize_mesh(mesh)

    def normalize_mesh(self, inplace=False) -> "Mesh":
        """A method to normalize the mesh by scaling the mesh to fit in a unit sphere.

        Args:
            inplace (bool, optional): A flag to indicate whether to normalize the mesh in place. Defaults to False.

        Returns:
            Mesh: The normalized mesh.
        """
        mesh = self if inplace else copy.deepcopy(self)
        return utils.normalize_mesh(mesh)

    def update_vertex(self, verts: torch.Tensor, inplace=False):
        """A method to update the vertices of the mesh.

        Args:
            verts (torch.Tensor): The new vertices for the mesh.
            inplace (bool, optional): A flag to indicate whether to update the vertices in place. Defaults to False.

        Returns:
            Mesh: The mesh with the updated vertices.
        """
        mesh = self if inplace else copy.deepcopy(self)
        mesh.vertices = verts
        return mesh

    def set_mesh_color(self, color: torch.Tensor):
        """A method to set the color of the mesh.

        Args:
            color (torch.Tensor): The color of the mesh. The color should be a tensor of shape (3,) representing the RGB values of the color.
        """
        self.texture_map = utils.get_texture_map_from_color(self, color)
        self.face_attributes = utils.get_face_attributes_from_color(self, color)

    def set_image_texture(
        self, texture_map: torch.Tensor | str, inplace=True
    ) -> "Mesh":
        """A method to set the texture map of the mesh.

        Args:
            texture_map (str or torch.Tensor): The texture map of the mesh. The texture map can be a path to an image file or a tensor of shape (1, C, H, W) representing the texture map.
            inplace (bool, optional): A flag to indicate whether to set the texture map in place. Defaults to True.

        Returns:
            Mesh: The mesh with the updated texture map.
        """
        mesh = self if inplace else copy.deepcopy(self)

        if isinstance(texture_map, str):
            texture_map = PIL.Image.open(texture_map)  # Load the image file
            texture_map = (
                np.array(texture_map, dtype=np.float) / 255.0
            )  # Normalize the image to [0, 1]
            texture_map = (
                torch.tensor(
                    texture_map, dtype=torch.float
                )  # Convert the image to a tensor
                .to(device)  # Move the tensor to the GPU
                .permute(2, 0, 1)  # Permute the dimensions of the tensor
                .unsqueeze(0)  # Add a batch dimension
            )

        mesh.texture_map = texture_map
        return mesh

    def divide(self, inplace=True) -> "Mesh":
        """A method to divide the mesh into smaller triangles.

        Args:
            inplace (bool, optional): A flag to indicate whether to divide the mesh in place. Defaults to True.

        Returns:
            Mesh: The mesh with the divided faces.
        """
        mesh = self if inplace else copy.deepcopy(self)
        new_vertices, new_faces, new_face_uvs = utils.add_vertices(mesh)
        mesh.vertices = new_vertices
        mesh.faces = new_faces
        mesh.face_uvs = new_face_uvs
        return mesh

    def export(self, file: str, color: np.ndarray | None = None):
        """Export the mesh to a file in the OBJ format.

        Args:
            file (str): The path to the file where the mesh will be saved.
            color (np.ndarray, optional): Optional array of vertex colors with shape (num_vertices, 3).
        """
        with open(file, "w+") as f:
            # Write vertices and optional colors to the file
            for vi, v in enumerate(self.vertices):
                if color is None:
                    # Write vertex position if color is not provided
                    f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
                else:
                    # Write vertex position and color if provided
                    f.write(
                        "v %f %f %f %f %f %f\n"
                        % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2])
                    )
                if self.vertex_normals is not None:
                    # Write vertex normals if they exist
                    f.write(
                        "vn %f %f %f\n"
                        % (
                            self.vertex_normals[vi, 0],
                            self.vertex_normals[vi, 1],
                            self.vertex_normals[vi, 2],
                        )
                    )

            # Write faces to the file
            for face in self.faces:
                # Faces in OBJ format are 1-indexed, hence adding 1 to each index
                f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))
