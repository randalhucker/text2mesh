from pathlib import Path
from typing import Tuple, Optional, List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import kaolin as kal

if TYPE_CHECKING:
    from mesh import Mesh

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device: torch.device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device: torch.device = torch.device("cpu")


def get_camera_from_view(elev: float, azim: float, r=3.0) -> torch.Tensor:
    """Generate a camera transformation matrix given elevation and azimuth angles.

    Args:
        elev (float): Elevation angle in radians.
        azim (float): Azimuth angle in radians.
        r (float, optional): Radius (distance from the origin). Defaults to 3.0.

    Returns:
        torch.Tensor: Camera transformation matrix.
    """
    # Compute the camera position in Cartesian coordinates
    x = r * torch.cos(azim) * torch.sin(elev)
    y = r * torch.sin(azim) * torch.sin(elev)
    z = r * torch.cos(elev)

    # Create a position tensor
    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos  # Camera looks towards the origin
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)  # Up direction vector

    # Generate the camera transformation matrix
    camera_proj = kal.render.camera.generate_transformation_matrix(
        pos, look_at, direction
    )
    return camera_proj


def get_camera_from_view2(elev: float, azim: float, r=3.0) -> torch.Tensor:
    """Generate a camera transformation matrix given elevation and azimuth angles, with a different Cartesian coordinate computation.

    Args:
        elev (float): Elevation angle in radians.
        azim (float): Azimuth angle in radians.
        r (float, optional): Radius (distance from the origin). Defaults to 3.0.

    Returns:
        torch.Tensor: Camera transformation matrix.
    """
    # Compute the camera position in Cartesian coordinates (different formula)
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)

    # Create a position tensor
    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
    direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    # Generate the camera transformation matrix
    camera_proj = kal.render.camera.generate_transformation_matrix(
        pos, look_at, direction
    )
    return camera_proj


def get_homogenous_coordinates(V: torch.Tensor):
    """Add a column of ones to the vertices.

    Args:
        V (torch.Tensor): The vertices.

    Returns:
        torch.Tensor: The vertices with a column of ones added.
    """
    N, D = V.shape
    bottom = torch.ones(N, device=device).unsqueeze(1)
    return torch.cat([V, bottom], dim=1)


def apply_affine(verts: torch.Tensor, A: torch.Tensor):
    """Apply an affine transformation to a set of vertices.
    An affine transformation is a linear transformation followed by a translation.

    Args:
        verts (torch.Tensor): The vertices to transform.
        A (torch.Tensor): The affine transformation matrix.

    Returns:
        torch.Tensor: The transformed vertices.
    """
    verts = verts.to(
        device
    )  # Ensure vertices are on the same device as the affine matrix
    verts = get_homogenous_coordinates(verts)
    A = torch.cat(
        [A, torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).unsqueeze(0)], dim=0
    )
    transformed_verts = A @ verts.T  # Apply the affine transformation
    transformed_verts = transformed_verts[:-1]  # Remove the last row
    return transformed_verts.T  # Return the transformed vertices


def standardize_mesh(mesh: "Mesh") -> "Mesh":
    """A method to standardize the mesh by centering the vertices and scaling them to a unit sphere.

    Args:
        mesh (Mesh): The mesh to standardize.

    Returns:
        Mesh: The standardized mesh.
    """
    verts = mesh.vertices
    center = verts.mean(dim=0)
    verts -= center
    scale = torch.std(torch.norm(verts, p=2, dim=1))
    verts /= scale
    mesh.vertices = verts
    return mesh


def normalize_mesh(mesh: "Mesh") -> "Mesh":
    """A method to normalize the mesh by scaling the mesh to fit in a unit sphere.

    Args:
        mesh (Mesh): The mesh to normalize.

    Returns:
        Mesh: The normalized mesh.
    """
    verts = mesh.vertices

    # Compute center of bounding box
    # center = torch.mean(torch.column_stack([torch.max(verts, dim=0)[0], torch.min(verts, dim=0)[0]]))
    center = verts.mean(dim=0)  # center of mass
    verts = verts - center  # center mesh
    scale = torch.max(torch.norm(verts, p=2, dim=1))  # scale mesh to fit in unit sphere
    verts = verts / scale  # normalize mesh
    mesh.vertices = verts  # update mesh vertices
    return mesh


def get_texture_map_from_color(color: torch.Tensor, H=224, W=224):
    """Get a texture map from a color. The texture map is a 2D image that is used to texture the mesh. The texture map is a tensor of shape (1, 3, H, W).

    Args:
        color (torch.Tensor): The color of the texture map.
        H (int, optional): The height of the texture map. Defaults to 224.
        W (int, optional): The width of the texture map. Defaults to 224.

    Returns:
        torch.Tensor: The texture map.
    """
    # initialize texture map with zeros and move to device
    texture_map = torch.zeros(1, H, W, 3).to(device)

    # set texture map color to input color
    texture_map[:, :, :] = color

    # return texture map with permuted dimensions
    return texture_map.permute(0, 3, 1, 2)


def get_face_attributes_from_color(mesh: "Mesh", color: torch.Tensor) -> torch.Tensor:
    """A method to get face attributes from a color.

    Args:
        mesh (Mesh): The mesh to get face attributes from.
        color (torch.Tensor): The color of the face attributes.

    Returns:
        torch.Tensor: The face attributes with the input color.
    """
    # get number of faces in mesh
    num_faces = mesh.faces.shape[0]

    # initialize face attributes with zeros and move to device
    face_attributes = torch.zeros(1, num_faces, 3, 3).to(device)

    # set face attributes color to input color
    face_attributes[:, :, :] = color
    return face_attributes


def sample_bary(faces: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Sample new vertices using random barycentric coordinates for each face.

    Args:
        faces (torch.Tensor): Tensor of face indices with shape (num_faces, 3).
        vertices (torch.Tensor): Tensor of vertex positions with shape (num_vertices, 3).

    Returns:
        torch.Tensor: Tensor of new vertices with shape (num_vertices + num_faces, 3).
    """
    num_faces = faces.shape[0]  # Number of faces
    num_vertices = vertices.shape[0]  # Number of vertices

    # Get random barycentric coordinates for each face
    A = torch.randn(num_faces)  # Random values for A
    B = torch.randn(num_faces) * (1 - A)  # Random values for B ensuring A + B < 1
    C = 1 - (A + B)  # Values for C ensuring A + B + C = 1

    bary = torch.vstack([A, B, C]).to(
        device
    )  # Stack A, B, and C to create barycentric coordinates

    # Initialize tensors for new vertices and new UVs
    new_vertices = torch.zeros(num_faces, 3).to(device)
    new_uvs = torch.zeros(num_faces, 2).to(device)

    # Get the vertices of each face
    face_verts = kal.ops.mesh.index_vertices_by_faces(vertices.unsqueeze(0), faces)

    # Compute the new vertex positions using the barycentric coordinates
    for f in range(num_faces):
        new_vertices[f] = bary[:, f] @ face_verts[:, f]

    # Concatenate the new vertices with the existing vertices
    new_vertices = torch.cat([vertices, new_vertices])
    return new_vertices


def add_vertices(
    mesh: "Mesh",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Add vertices to the mesh by sampling barycentric coordinates for each face.

    Args:
        mesh (Mesh): The mesh to add vertices to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: New vertices, new faces, new vertex normals, and new face UVs (if present).
    """
    faces = mesh.faces
    vertices = mesh.vertices
    num_faces = faces.shape[0]
    num_vertices = vertices.shape[0]

    # Get random barycentric coordinates for each face TODO: improve sampling
    # Create a tensor of random numbers from a normal distribution
    A = torch.randn(num_faces)
    # Create a tensor of random numbers from a normal distribution and multiply by (1 - A) to ensure that the sum of A and B is less than 1
    B = torch.randn(num_faces) * (1 - A)
    # Create a tensor of random numbers such that the sum of A, B, and C is equal to 1
    C = 1 - (A + B)
    # Stack A, B, and C to create a tensor of shape (3, num_faces)
    bary = torch.vstack([A, B, C]).to(device)

    # Initialize tensors for new vertices and new UVs
    new_vertices = torch.zeros(num_faces, 3).to(device)
    new_uvs = torch.zeros(num_faces, 2).to(device)
    # Get the vertices of each face
    face_verts = kal.ops.mesh.index_vertices_by_faces(vertices.unsqueeze(0), faces)
    face_uvs = mesh.face_uvs

    # Compute the coordinates of new vertices and new UVs (if present)
    for f in range(num_faces):
        # Compute the new vertex by taking the dot product of the barycentric coordinates and the vertices of the face
        new_vertices[f] = bary[:, f] @ face_verts[:, f]
        if face_uvs is not None:
            # Compute the new UV by taking the dot product of the barycentric coordinates and the UVs of the face
            new_uvs[f] = bary[:, f] @ face_uvs[:, f]

    # Append new vertices to the existing vertices
    new_vertices = torch.cat([vertices, new_vertices])
    new_faces: List[torch.Tensor] = []
    new_face_uvs: List[torch.Tensor] = []
    new_vertex_normals: List[torch.Tensor] = []
    # Create new faces by splitting each face into three faces
    for i in range(num_faces):
        old_face = faces[i]  # Get the vertices of the old face
        a, b, c = old_face[0], old_face[1], old_face[2]
        d = num_vertices + i
        new_faces.append(torch.tensor([a, b, d]).to(device))
        new_faces.append(torch.tensor([a, d, c]).to(device))
        new_faces.append(torch.tensor([d, b, c]).to(device))

        # Update face UVs if present
        if face_uvs is not None:
            old_face_uvs = face_uvs[0, i]  # Get the UVs of the old face
            a, b, c = old_face_uvs[0], old_face_uvs[1], old_face_uvs[2]
            d = new_uvs[i]
            new_face_uvs.append(torch.vstack([a, b, d]))
            new_face_uvs.append(torch.vstack([a, d, c]))
            new_face_uvs.append(torch.vstack([d, b, c]))

        # Update vertex normals if present
        if mesh.face_normals is not None:
            new_vertex_normals.append(mesh.face_normals[i])
        else:
            e1 = vertices[b] - vertices[a]
            e2 = vertices[c] - vertices[a]
            norm = torch.cross(e1, e2)  # Calculate the normal vector
            norm /= torch.norm(norm)  # Normalize the normal

            # Double check sign against existing vertex normals
            if torch.dot(norm, mesh.vertex_normals[a]) < 0:
                norm = -norm

            new_vertex_normals.append(norm)

    # Concatenate new vertex normals with existing vertex normals
    vertex_normals = torch.cat([mesh.vertex_normals, torch.stack(new_vertex_normals)])

    # Reshape new face UVs if present
    if face_uvs is not None:
        new_face_uvs = (
            torch.vstack(new_face_uvs).unsqueeze(0).view(1, 3 * num_faces, 3, 2)
        )

    # Concatenate new faces to a single tensor
    new_faces = torch.vstack(new_faces)

    return new_vertices, new_faces, vertex_normals, new_face_uvs


def get_rgb_per_vertex(
    vertices: torch.Tensor, faces: torch.Tensor, face_rgbs: torch.Tensor
) -> torch.Tensor:
    """Assign RGB colors to each vertex based on the face colors.

    Args:
        vertices (torch.Tensor): Tensor of vertex positions with shape (num_vertex, 3).
        faces (torch.Tensor): Tensor of face indices with shape (num_faces, 3).
        face_rgbs (torch.Tensor): Tensor of RGB colors for each face with shape (num_faces, 3).

    Returns:
        torch.Tensor: Tensor of RGB colors for each vertex with shape (num_vertex, 3).
    """
    num_vertex = vertices.shape[0]  # Get the number of vertices
    num_faces = faces.shape[0]  # Get the number of faces
    vertex_color = torch.zeros(num_vertex, 3)  # Initialize tensor to hold vertex colors

    # Assign face color to each vertex in the mesh
    for v in range(num_vertex):
        for f in range(num_faces):
            face = num_faces[f]  # Get the current face
            if v in face:  # Check if the vertex is in the face
                vertex_color[v] = face_rgbs[f]  # Assign the face color to the vertex

    return face_rgbs


def get_barycentric(p: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Calculate barycentric coordinates for a set of points relative to a set of triangles (faces).

    Args:
        p (torch.Tensor): Tensor of points with shape (num_points, 3).
        faces (torch.Tensor): Tensor of triangle vertices with shape (num_points, 3, 3).

    Returns:
        torch.Tensor: Barycentric coordinates for each point with shape (num_points, 3).
    """
    # Extract vertices of the triangles
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]

    # Calculate the vectors from vertex a to vertices b, c, and point p
    v0, v1, v2 = b - a, c - a, p - a

    # Calculate dot products for the vectors
    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)

    # Calculate the denominator of the barycentric coordinates
    denom = d00 * d11 - d01 * d01

    # Calculate the barycentric coordinates v and w
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom

    # Calculate the barycentric coordinate u
    u = 1 - (w + v)

    # Stack and transpose the barycentric coordinates to get the final result
    return torch.vstack([u, v, w]).T


def get_uv_assignment(num_faces: int):
    """Generate a UV map for the given number of faces.

    Args:
        num_faces (int): Number of faces in the mesh.

    Returns:
        torch.Tensor: UV map with shape (1, num_faces, 3, 2).
    """
    # Calculate the dimension of the UV grid (M x M) to fit all faces
    M = int(np.ceil(np.sqrt(num_faces)))

    # Initialize a tensor for the UV map with zeros
    uv_map = torch.zeros(1, num_faces, 3, 2).to(device)

    px, py = 0, 0  # Initialize the starting positions for UV coordinates
    count = 0  # Initialize the face count

    for _ in range(M):
        px = 0  # Reset the x-coordinate for each row
        for _ in range(M):
            # Assign UV coordinates to current face
            uv_map[:, count] = torch.tensor([[px, py], [px + 1, py], [px + 1, py + 1]])
            px += 2  # Move to the next position in the x-direction
            count += 1  # Increment the face count

            # If all faces are assigned, normalize the UV map and return it
            if count >= num_faces:
                # Get the maximum width and height of the UV map
                hw = torch.max(uv_map.view(-1, 2), dim=0)[0]
                uv_map = (uv_map - hw / 2.0) / (hw / 2)  # Normalize the UV map
                return uv_map
        py += 2  # Move to the next position in the y-direction


def get_texture_visual(res: int, nt: callable, mesh: "Mesh") -> torch.Tensor:
    """Generate a texture visualization for the mesh.

    Args:
        res (int): The resolution of the texture image.
        nt (Callable): A function that maps 3D coordinates to RGB colors.
        mesh (Mesh): The mesh to visualize.

    Returns:
        torch.Tensor: The generated texture image with shape (3, res, res).
    """
    # Get vertices of each face
    faces_vt = kal.ops.mesh.index_vertices_by_faces(
        mesh.vertices.unsqueeze(0), mesh.faces
    ).squeeze(0)

    # Generate a grid of UV coordinates
    # To not include the endpoint, generate res+1 points and take the first res
    uv = torch.cartesian_prod(
        torch.linspace(-1, 1, res + 1)[:-1], torch.linspace(-1, 1, res + 1)
    )[:-1].to(device)

    # Initialize an image tensor with zeros
    image = torch.zeros(res, res, 3).to(device)

    # Permute the dimensions of the image tensor to (3, res, res)
    image = image.permute(2, 0, 1)

    # Get the number of faces in the mesh
    num_faces = mesh.faces.shape[0]

    # Get the UV map for the faces
    uv_map = get_uv_assignment(num_faces).squeeze(0)

    # Define tensors for zero and one
    zero = torch.tensor([0.0, 0.0, 0.0]).to(device)
    one = torch.tensor([1.0, 1.0, 1.0]).to(device)

    # Iterate over each face to compute the texture
    for face in range(num_faces):
        # Get barycentric coordinates for the UV grid points
        bary = get_barycentric(uv, uv_map[face].repeat(len(uv), 1, 1))

        # Create masks to determine which points are inside the triangle
        maskA = torch.logical_and(bary[:, 0] >= 0.0, bary[:, 0] <= 1.0)
        maskB = torch.logical_and(bary[:, 1] >= 0.0, bary[:, 1] <= 1.0)
        maskC = torch.logical_and(bary[:, 2] >= 0.0, bary[:, 2] <= 1.0)
        mask = torch.logical_and(maskA, maskB)
        mask = torch.logical_and(maskC, mask)

        # Get points that are inside the triangle
        inside_triangle = bary[mask]
        inside_triangle_uv = inside_triangle @ uv_map[face]
        inside_triangle_xyz = inside_triangle @ faces_vt[face]
        inside_triangle_rgb = nt(inside_triangle_xyz)

        # Map UV coordinates to pixel coordinates
        pixels = (inside_triangle_uv + 1.0) / 2.0
        pixels = pixels * res
        pixels = torch.floor(pixels).type(torch.int64)

        # Assign RGB colors to the corresponding pixels in the image
        image[:, pixels[:, 0], pixels[:, 1]] = inside_triangle_rgb.T

    return image


def getRotMat(axis: np.ndarray, theta: float) -> np.ndarray:
    """Get the rotation matrix for rotating a vector around an axis through the origin.

    Args:
        axis (np.ndarray): A normalized vector representing the axis of rotation.
        theta (float): The angle of rotation in radians.

    Returns:
        np.ndarray: The 3x3 rotation matrix.
    """
    import math

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)

    # Create the cross-product matrix for the axis vector
    cprod = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Calculate the rotation matrix using the Rodrigues' rotation formula
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    rot = (
        math.cos(theta) * np.identity(3)  # cos(theta) * I
        + math.sin(theta) * cprod  # sin(theta) * [axis]_x (cross-product matrix)
        + (1 - math.cos(theta))
        * np.outer(axis, axis)  # (1 - cos(theta)) * axis * axis.T
    )
    return rot


def trimMesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map vertices and a subset of faces to 0-indexed vertices, keeping only relevant vertices.

    Args:
        vertices (np.ndarray): Array of vertex positions with shape (num_vertices, 3).
        faces (np.ndarray): Array of face indices with shape (num_faces, 3).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Trimmed vertices and faces with shape (num_relevant_vertices, 3) and (num_faces, 3) respectively.
    """
    # Find the unique vertices that are used in the faces
    unique_v = np.sort(np.unique(faces.flatten()))

    # Create a mapping from old vertex indices to new 0-indexed vertex indices
    v_val = np.arange(len(unique_v))
    v_map = dict(zip(unique_v, v_val))

    # Remap the faces to use the new 0-indexed vertex indices
    new_faces = np.array([v_map[i] for i in faces.flatten()]).reshape(
        faces.shape[0], faces.shape[1]
    )

    # Extract the relevant vertices based on the unique vertex indices
    new_v = vertices[unique_v]

    return new_v, new_faces


# ================== VISUALIZATION =======================
def extract_from_gl_viewmat(gl_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract camera location and target position from a GL view matrix.

    Args:
        gl_mat (np.ndarray): A 4x4 view transformation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Camera location and target position.
    """
    # Reshape the input matrix to ensure it is 4x4
    gl_mat = gl_mat.reshape(4, 4)

    # Extract the right (s), up (u), and forward (f) vectors from the matrix
    s = gl_mat[0, :3]  # Right vector
    u = gl_mat[1, :3]  # Up vector
    f = -1 * gl_mat[2, :3]  # Forward vector (negated)

    # Extract the camera coordinates from the first three entries of the last column
    coord = gl_mat[:3, 3]

    # Compute the camera location by transforming the coordinates
    camera_location = np.array([-s, -u, f]).T @ coord

    # Compute the target position by moving along the forward vector from the camera location
    target = camera_location + f * 10  # Any scale can be used

    return camera_location, target


def psScreenshot(
    vertices: np.ndarray,
    faces: np.ndarray,
    axis: np.ndarray,
    angles: List[float],
    save_path: str,
    name: str = "mesh",
    frame_folder: str = "frames",
    scalars: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    defined_on: str = "faces",
    highlight_faces: Optional[np.ndarray] = None,
    highlight_color: List[int] = [1, 0, 0],
    highlight_radius: Optional[float] = None,
    cmap: Optional[str] = None,
    sminmax: Optional[Tuple] = None,
    cpos: Optional[np.ndarray] = None,
    clook: Optional[np.ndarray] = None,
    save_video: bool = False,
    save_base: bool = False,
    ground_plane: str = "tile_reflection",
    debug: bool = False,
    edge_color: List[int] = [0, 0, 0],
    edge_width: int = 1,
    material: Optional[str] = None,
):
    """
    Generate and save screenshots of a mesh rotated around a given axis.

    Args:
        vertices (np.ndarray): Array of vertex positions.
        faces (np.ndarray): Array of face indices.
        axis (np.ndarray): Axis of rotation.
        angles (list): List of rotation angles in radians.
        save_path (str): Path to save the screenshots.
        name (str, optional): Name prefix for the saved screenshots. Defaults to "mesh".
        frame_folder (str, optional): Subfolder to save individual frames. Defaults to "frames".
        scalars (np.ndarray, optional): Scalars to visualize on the mesh.
        colors (np.ndarray, optional): Colors to visualize on the mesh.
        defined_on (str, optional): Specifies if the scalars/colors are defined on 'faces' or 'vertices'. Defaults to "faces".
        highlight_faces (np.ndarray, optional): Faces to highlight.
        highlight_color (list, optional): Color to use for highlighting faces. Defaults to [1, 0, 0].
        highlight_radius (float, optional): Radius for the highlight curves.
        cmap (str, optional): Colormap for the scalar visualization.
        sminmax (tuple, optional): Min and max values for scalar visualization.
        cpos (np.ndarray, optional): Camera position.
        clook (np.ndarray, optional): Camera look-at position.
        save_video (bool, optional): Whether to save the frames as a video. Defaults to False.
        save_base (bool, optional): Whether to save the base mesh without rotation. Defaults to False.
        ground_plane (str, optional): Ground plane mode for Polyscope. Defaults to "tile_reflection".
        debug (bool, optional): Whether to show the Polyscope viewer for debugging. Defaults to False.
        edge_color (list, optional): Edge color for the mesh. Defaults to [0, 0, 0].
        edge_width (int, optional): Edge width for the mesh. Defaults to 1.
        material (str, optional): Material for the mesh.

    """
    import polyscope as ps

    # Initialize Polyscope
    ps.init()
    ps.set_ground_plane_mode(ground_plane)

    # Create the frame folder path
    frame_path = f"{save_path}/{frame_folder}"

    # Save the base mesh without rotation if specified
    if save_base == True:
        ps_mesh = ps.register_surface_mesh(
            "mesh",
            vertices,
            faces,
            enabled=True,
            edge_color=edge_color,
            edge_width=edge_width,
            material=material,
        )
        ps.screenshot(f"{frame_path}/{name}.png")
        ps.remove_all_structures()

    # Create the frame folder if it doesn't exist
    Path(frame_path).mkdir(parents=True, exist_ok=True)

    # Convert 2D vertices to 3D by appending Z-axis if necessary
    if vertices.shape[1] == 2:
        vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)

    # Iterate over each angle to generate rotated views
    for i in range(len(angles)):
        rot = getRotMat(axis, angles[i])  # Get the rotation matrix
        rot_verts = np.transpose(rot @ np.transpose(vertices))  # Apply rotation

        # Register the rotated mesh with Polyscope
        ps_mesh = ps.register_surface_mesh(
            "mesh",
            rot_verts,
            faces,
            enabled=True,
            edge_color=edge_color,
            edge_width=edge_width,
            material=material,
        )

        # Add scalar quantity if provided
        if scalars is not None:
            ps_mesh.add_scalar_quantity(
                f"scalar",
                scalars,
                defined_on=defined_on,
                cmap=cmap,
                enabled=True,
                vminmax=sminmax,
            )

        # Add color quantity if provided
        if colors is not None:
            ps_mesh.add_color_quantity(
                f"color", colors, defined_on=defined_on, enabled=True
            )

        # Highlight specified faces if provided
        if highlight_faces is not None:
            curve_v, new_f = trimMesh(rot_verts, faces[highlight_faces, :])
            curve_edges = []
            for face in new_f:
                curve_edges.extend(
                    [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]]
                )
            curve_edges = np.array(curve_edges)
            ps_curve = ps.register_curve_network(
                "curve",
                curve_v,
                curve_edges,
                color=highlight_color,
                radius=highlight_radius,
            )

        # Set the camera view
        if cpos is None or clook is None:
            ps.reset_camera_to_home_view()
        else:
            ps.look_at(cpos, clook)

        # Show the Polyscope viewer if in debug mode
        if debug == True:
            ps.show()

        # Save the screenshot for the current rotation
        ps.screenshot(f"{frame_path}/{name}_{i}.png")
        ps.remove_all_structures()

    # Save the frames as a GIF if specified
    if save_video == True:
        import glob
        from PIL import Image

        fp_in = f"{frame_path}/{name}_*.png"
        fp_out = f"{save_path}/{name}.gif"
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(
            fp=fp_out,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=200,
            loop=0,
        )


# ================== POSITIONAL ENCODERS =============================
class FourierFeatureTransform(torch.nn.Module):
    """An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height], returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10, exclude=0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self.exclude = exclude

        # Generate a random matrix B for Fourier feature mapping, scaled by the specified factor
        B = torch.randn((num_input_channels, mapping_size)) * scale

        # Sort B by the L2 norm of its rows
        B_sort = sorted(B, key=lambda x: torch.norm(x, p=2))

        # Register B as a non-trainable parameter (fixed after initialization)
        self._B = nn.Parameter(torch.stack(B_sort), requires_grad=False)

    def forward(self, x):
        """Forward pass for the Fourier feature transform.

        Args:
            x (torch.Tensor): Input tensor of size [batches, num_input_channels].

        Returns:
            torch.Tensor: Transformed tensor with Fourier features.
        """
        # Ensure input tensor has the expected number of dimensions
        # assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        # Extract batch size and number of channels from input tensor
        batches, channels = x.shape

        # Ensure input tensor has the expected number of channels
        assert (
            channels == self._num_input_channels
        ), "Expected input to have {} channels (got {} channels)".format(
            self._num_input_channels, channels
        )

        # Make shape compatible for matmul with _B (if necessary)
        # From [B, C, W, H] to [(B*W*H), C].
        # x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        # Apply Fourier feature mapping
        res = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        # x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        # x = x.permute(0, 3, 1, 2)

        # Scale the result by 2π for the sine and cosine functions
        res = 2 * np.pi * res

        # Concatenate the original input with its sine and cosine transformed versions
        return torch.cat([x, torch.sin(res), torch.cos(res)], dim=1)
