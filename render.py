from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

import kaolin as kal

from mesh import Mesh
from utils import get_camera_from_view2, device


class Renderer:
    def __init__(
        self,
        mesh: Mesh,
        lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
        dim=(224, 224),
    ):
        """Initialize the Renderer.

        Args:
            mesh (Mesh): The mesh to render.
            lights (torch.Tensor, optional): Lighting parameters. Defaults to a tensor with predefined values.
            camera (torch.Tensor, optional): Camera projection matrix. Defaults to a perspective projection.
            dim (tuple, optional): Dimensions of the rendered output (width, height). Defaults to (224, 224).
        """

        self.mesh = mesh
        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(
                device
            )

        self.lights = lights.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device
        self.camera_projection = camera  # Camera projection matrix
        self.dim = dim  # Dimensions of the rendered output

    def render_y_views(
        self,
        mesh: Mesh,
        num_views=8,
        show=False,
        lighting=True,
        background: Optional[torch.Tensor] = None,
        mask=False,
    ):
        """Render the mesh from multiple viewpoints along the y-axis.

        Args:
            mesh (Mesh): The mesh to render.
            num_views (int, optional): Number of viewpoints. Defaults to 8.
            show (bool, optional): Whether to display the rendered images. Defaults to False.
            lighting (bool, optional): Whether to apply lighting. Defaults to True.
            background (torch.Tensor, optional): Background color to apply. Defaults to None.
            mask (bool, optional): Whether to generate a mask. Defaults to False.

        Returns:
            torch.Tensor: Rendered images.
        """
        faces = mesh.faces
        n_faces = faces.shape[0]

        # Generate evenly spaced azimuth angles from 0 to 2*pi
        # Don't include the last element since 0 = 2*pi
        azim = torch.linspace(0, 2 * np.pi, num_views + 1)[:-1]

        # elev = torch.cat((torch.linspace(0, np.pi/2, int((num_views+1)/2)), torch.linspace(0, -np.pi/2, int((num_views)/2))))
        elev = torch.zeros(len(azim))  # Set elevation angles to 0
        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            # Combine face attributes with a background attribute
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device),
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            # Get the camera transformation matrix for the current view
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)

            # Prepare the mesh vertices for rendering
            face_vertices_camera, face_vertices_image, face_normals = (
                kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device),
                    mesh.faces.to(device),
                    self.camera_projection,
                    camera_transform=camera_transform,
                )
            )

            # Perform rasterization to get image features and mask
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            # Clamp image features to [0, 1] range
            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                # Apply spherical harmonic lighting
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                    image_normals, self.lights
                ).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(
                    0, 2, 3, 1
                ).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                # Apply background color
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(
                    image[torch.where(mask == 0)] == torch.zeros(3).to(device)
                )
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0.0, 1.0)

            images.append(image)

        # Concatenate images and masks along the batch dimension
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(
                    1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4)
                )
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(images[i].permute(1,2,0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_single_view(
        self,
        mesh: Mesh,
        elev=0,
        azim=0,
        show=False,
        lighting=True,
        background=None,
        radius=2,
        return_mask=False,
    ):
        """Render a single view of the mesh.

        Args:
            mesh (Mesh): The mesh to render.
            elev (float, optional): Elevation angle in radians. Defaults to 0.
            azim (float, optional): Azimuth angle in radians. Defaults to 0.
            show (bool, optional): Whether to display the rendered image. Defaults to False.
            lighting (bool, optional): Whether to apply lighting. Defaults to True.
            background (Optional[torch.Tensor], optional): Background color to apply. Defaults to None.
            radius (float, optional): Distance from the camera to the object. Defaults to 2.
            return_mask (bool, optional): Whether to return the mask. Defaults to False.

        Returns:
            torch.Tensor: Rendered image (and mask if return_mask is True).
        """
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if background is not None:
            # Combine face attributes with a background attribute
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device),
            ]
        else:
            face_attributes = mesh.face_attributes

        # Get the camera transformation matrix
        camera_transform = get_camera_from_view2(
            torch.tensor(elev), torch.tensor(azim), r=radius
        ).to(device)

        # Prepare the mesh vertices for rendering
        face_vertices_camera, face_vertices_image, face_normals = (
            kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device),
                mesh.faces.to(device),
                self.camera_projection,
                camera_transform=camera_transform,
            )
        )

        # Perform rasterization to get image features and mask
        image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            self.dim[1],
            self.dim[0],
            face_vertices_camera[:, :, :, -1],
            face_vertices_image,
            face_attributes,
            face_normals[:, :, -1],
        )

        # Debugging: color where soft mask is 1
        # tmp_rgb = torch.ones((224,224,3))
        # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
        # rgb_mask.append(tmp_rgb)

        if background is not None:
            image_features, mask = image_features

        # Clamp image features to [0, 1] range
        image = torch.clamp(image_features, 0.0, 1.0)

        if lighting:
            # Apply spherical harmonic lighting
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                image_normals, self.lights
            ).unsqueeze(0)
            image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(
                device
            )
            image = torch.clamp(image, 0.0, 1.0)

        if background is not None:
            # Apply background color
            background_mask = torch.zeros(image.shape).to(device)
            mask = mask.squeeze(-1)
            assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
            background_mask[torch.where(mask == 0)] = background
            image = torch.clamp(image + background_mask, 0.0, 1.0)

        if show:
            # Display the rendered image
            with torch.no_grad():
                fig, axs = plt.subplots(figsize=(89.6, 22.4))
                axs.imshow(image[0].cpu().numpy())
                # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if return_mask == True:
            # Return the rendered image and mask
            return image.permute(0, 3, 1, 2), mask

        # Return the rendered image
        return image.permute(0, 3, 1, 2)

    def render_uniform_views(
        self,
        mesh: Mesh,
        num_views=8,
        show=False,
        lighting=True,
        background=None,
        mask=False,
        center=[0, 0],
        radius=2.0,
    ):
        """Render the mesh from multiple uniformly distributed viewpoints.

        Args:
            mesh (Mesh): The mesh to render.
            num_views (int, optional): Number of viewpoints. Defaults to 8.
            show (bool, optional): Whether to display the rendered images. Defaults to False.
            lighting (bool, optional): Whether to apply lighting. Defaults to True.
            background (Optional[torch.Tensor], optional): Background color to apply. Defaults to None.
            mask (bool, optional): Whether to generate a mask. Defaults to False.
            center (list, optional): Center of the view angles [azimuth, elevation]. Defaults to [0, 0].
            radius (float, optional): Distance from the camera to the object. Defaults to 2.0.

        Returns:
            torch.Tensor: Rendered images (and masks if specified).
        """
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        # Generate evenly spaced azimuth angles from 0 to 2*pi centered around `center[0]`
        # Don't include the last element since 0 = 2*pi
        azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[:-1]

        # Generate elevation angles centered around `center[1]`
        elev = torch.cat(
            (
                torch.linspace(
                    center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)
                ),
                torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2)),
            )
        )

        images = []
        masks = []
        background_masks = []

        if background is not None:
            # Combine face attributes with a background attribute
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device),
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            # Get the camera transformation matrix for the current view
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(
                device
            )

            # Prepare the mesh vertices for rendering
            face_vertices_camera, face_vertices_image, face_normals = (
                kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device),
                    mesh.faces.to(device),
                    self.camera_projection,
                    camera_transform=camera_transform,
                )
            )

            # Perform rasterization to get image features and mask
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((224,224,3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1,0,0]).float()
            # rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            # Clamp image features to [0, 1] range
            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                # Apply spherical harmonic lighting
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                    image_normals, self.lights
                ).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(
                    0, 2, 3, 1
                ).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                # Apply background color
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(
                    image[torch.where(mask == 0)] == torch.zeros(3).to(device)
                )
                background_mask[torch.where(mask == 0)] = background
                background_masks.append(background_mask)
                image = torch.clamp(image + background_mask, 0.0, 1.0)

            images.append(image)

        # Concatenate images and masks along the batch dimension
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        if background is not None:
            background_masks = torch.cat(background_masks, dim=0).permute(0, 3, 1, 2)

        if show:
            # Display the rendered images
            with torch.no_grad():
                fig, axs = plt.subplots(
                    1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4)
                )
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    # ax.imshow(background_masks[i].permute(1,2,0).cpu().numpy())
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        return images

    def render_front_views(
        self,
        mesh: Mesh,
        num_views=8,
        std=8,
        center_elev=0,
        center_azim=0,
        show=False,
        lighting=True,
        background=None,
        mask=False,
        return_views=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Render the mesh from multiple front viewpoints with small perturbations.

        Args:
            mesh (Mesh): The mesh to render.
            num_views (int, optional): Number of viewpoints. Defaults to 8.
            std (int, optional): Standard deviation for perturbations in viewing angle. Defaults to 8.
            center_elev (float, optional): Center elevation angle in radians. Defaults to 0.
            center_azim (float, optional): Center azimuth angle in radians. Defaults to 0.
            show (bool, optional): Whether to display the rendered images. Defaults to False.
            lighting (bool, optional): Whether to apply lighting. Defaults to True.
            background (Optional[torch.Tensor], optional): Background color to apply. Defaults to None.
            mask (bool, optional): Whether to generate a mask. Defaults to False.
            return_views (bool, optional): Whether to return the viewing angles. Defaults to False.

        Returns:
            torch.Tensor: Rendered images (and masks if specified).
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Elevation and azimuth angles if return_views is True.
        """
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        # Generate elevation and azimuth angles with small perturbations around the center
        elev = torch.cat(
            (
                torch.tensor([center_elev]),
                torch.randn(num_views - 1) * np.pi / std + center_elev,
            )
        )
        azim = torch.cat(
            (
                torch.tensor([center_azim]),
                torch.randn(num_views - 1) * 2 * np.pi / std + center_azim,
            )
        )

        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            # Combine face attributes with a background attribute
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device),
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            # Get the camera transformation matrix for the current view
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=2).to(device)

            # Prepare the mesh vertices for rendering
            face_vertices_camera, face_vertices_image, face_normals = (
                kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device),
                    mesh.faces.to(device),
                    self.camera_projection,
                    camera_transform=camera_transform,
                )
            )

            # Perform rasterization to get image features and mask
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            masks.append(soft_mask)

            # Debugging: color where soft mask is 1
            # tmp_rgb = torch.ones((224, 224, 3))
            # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
            # rgb_mask.append(tmp_rgb)

            if background is not None:
                image_features, mask = image_features

            # Clamp image features to [0, 1] range
            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                # Apply spherical harmonic lighting
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                    image_normals, self.lights
                ).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(
                    0, 2, 3, 1
                ).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                # Apply background color
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(
                    image[torch.where(mask == 0)] == torch.zeros(3).to(device)
                )
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0.0, 1.0)
            images.append(image)

        # Concatenate images and masks along the batch dimension
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        # rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            # Display the rendered images
            with torch.no_grad():
                fig, axs = plt.subplots(
                    1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4)
                )
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images

    def render_prompt_views(
        self,
        mesh: Mesh,
        prompt_views: List[str],
        center=[0, 0],
        background=None,
        show=False,
        lighting=True,
        mask=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Render the mesh from specified viewpoints.

        Args:
            mesh (Mesh): The mesh to render.
            prompt_views (list): List of viewpoints to render from (e.g., ["front", "right"]).
            center (list, optional): Center of the view angles [azimuth, elevation]. Defaults to [0, 0].
            background (Optional[torch.Tensor], optional): Background color to apply. Defaults to None.
            show (bool, optional): Whether to display the rendered images. Defaults to False.
            lighting (bool, optional): Whether to apply lighting. Defaults to True.
            mask (bool, optional): Whether to return the mask. Defaults to False.

        Returns:
            torch.Tensor: Rendered images.
            torch.Tensor (optional): Masks if mask is True.
        """
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]
        num_views = len(prompt_views)

        images = []
        masks = []
        rgb_mask = []
        face_attributes = mesh.face_attributes

        for i in range(num_views):
            view = prompt_views[i]

            # Determine elevation and azimuth angles based on the view prompt
            if view == "front":
                elev = 0 + center[1]
                azim = 0 + center[0]
            if view == "right":
                elev = 0 + center[1]
                azim = np.pi / 2 + center[0]
            if view == "back":
                elev = 0 + center[1]
                azim = np.pi + center[0]
            if view == "left":
                elev = 0 + center[1]
                azim = 3 * np.pi / 2 + center[0]
            if view == "top":
                elev = np.pi / 2 + center[1]
                azim = 0 + center[0]
            if view == "bottom":
                elev = -np.pi / 2 + center[1]
                azim = 0 + center[0]

            if background is not None:
                # Combine face attributes with a background attribute
                face_attributes = [
                    mesh.face_attributes,
                    torch.ones((1, n_faces, 3, 1), device=device),
                ]
            else:
                face_attributes = mesh.face_attributes

            # Get the camera transformation matrix for the current view
            camera_transform = get_camera_from_view2(
                torch.tensor(elev), torch.tensor(azim), r=2
            ).to(device)

            # Prepare the mesh vertices for rendering
            face_vertices_camera, face_vertices_image, face_normals = (
                kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device),
                    mesh.faces.to(device),
                    self.camera_projection,
                    camera_transform=camera_transform,
                )
            )

            # Perform rasterization to get image features and mask
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1],
                self.dim[0],
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                face_normals[:, :, -1],
            )
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            # Clamp image features to [0, 1] range
            image = torch.clamp(image_features, 0.0, 1.0)

            if lighting:
                # Apply spherical harmonic lighting
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(
                    image_normals, self.lights
                ).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(
                    0, 2, 3, 1
                ).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            if background is not None:
                # Apply background color
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(
                    image[torch.where(mask == 0)] == torch.zeros(3).to(device)
                )
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0.0, 1.0)
            images.append(image)

        # Concatenate images and masks along the batch dimension
        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        if show:
            # Display the rendered images
            with torch.no_grad():
                fig, axs = plt.subplots(
                    1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4)
                )
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                    # ax.imshow(rgb_mask[i].cpu().numpy())
                plt.show()

        if not mask:
            return images
        else:
            return images, masks


if __name__ == "__main__":
    # Initialize the mesh with a sample .obj file
    mesh = Mesh("sample.obj")

    # Set the texture for the mesh using a sample texture file
    mesh.set_image_texture("sample_texture.png")

    # Initialize the renderer
    renderer = Renderer(mesh)

    # Uncomment the line below to render the uniform views without dividing the mesh
    # renderer.render_uniform_views(mesh, show=True, texture=True)

    # Divide the mesh to add more vertices and faces
    mesh = mesh.divide()

    # Render the uniform views of the divided mesh and display the images
    renderer.render_uniform_views(mesh, show=True, texture=True)
