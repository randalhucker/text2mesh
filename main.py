import argparse
import copy
import os
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import kaolin as kal
import kaolin.ops.mesh
import clip

from mesh import Mesh
from Normalization import MeshNormalizer
from neural_style_field import NeuralStyleField, load_model, save_model
from render import Renderer
from utils import device


def run_branched(args: argparse.Namespace):
    """Run the main training loop for the Neural Style Field model.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Create the output directory if it does not exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Constrain all sources of randomness for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load the specified CLIP model and preprocessing function
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)

    # Adjust output resolution depending on model type
    res = 224  # Default resolution
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448

    # Extract the base name and extension of the input OBJ file
    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))

    # Check if output files already exist and whether to overwrite them
    if (
        (not args.overwrite)
        and os.path.exists(os.path.join(args.output_dir, "loss.png"))
        and os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj"))
    ):
        print(f"Already done with {args.output_dir}")
        exit()
    elif (
        args.overwrite
        and os.path.exists(os.path.join(args.output_dir, "loss.png"))
        and os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj"))
    ):
        import shutil

        # Delete existing files if overwrite is enabled
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(
                        file_path
                    )  # Remove the directory and all its contents
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))

    # Load the mesh from the OBJ file and initialize the renderer
    mesh = Mesh(args.obj_path)
    render = Renderer(mesh, dim=(res, res))

    # Normalize the mesh vertices
    MeshNormalizer(mesh)()

    # Initialize prior color to a neutral gray
    prior_color = torch.full(
        size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device
    )

    # Set background if specified
    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)

    losses = []  # List to store loss values

    n_augs = args.n_augs  # Number of augmentations for the rendered images
    dir = args.output_dir  # Output directory

    # Normalize transform for image data
    # Mean and standard deviation values for the images in the dataset
    clip_normalizer = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    # CLIP Transform
    clip_transform = transforms.Compose(
        [transforms.Resize((res, res)), clip_normalizer]
    )

    # Augmentation settings
    # The augmentations are applied to the rendered images
    # in the order they're listed to increase the diversity of the training data
    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(res, scale=(1, 1)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer,
        ]
    )

    # Crop settings for normal augmentations
    if args.cropforward:
        curcrop = args.normmincrop
    else:
        curcrop = args.normmaxcrop

    # Normal augment transform
    normaugment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer,
        ]
    )

    cropiter = 0  # Crop iteration
    cropupdate = 0  # Crop update

    if args.normmincrop < args.normmaxcrop and args.cropsteps > 0:
        # Calculate crop iteration and update
        cropiter = round(args.n_iter / (args.cropsteps + 1))
        cropupdate = (args.maxcrop - args.mincrop) / cropiter

        # Set crop direction
        if not args.cropforward:
            cropupdate *= -1

    # Displacement-only augmentations
    displaugment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                res, scale=(args.normmincrop, args.normmincrop)
            ),
            transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
            clip_normalizer,
        ]
    )

    normweight = 1.0  # Normal for the normal augmentations

    # MLP Settings
    # Input dimensions for the MLP network (3 for XYZ, 6 for XYZ+Normals)
    input_dim = 6 if args.input_normals else 3

    # Only Z dimension
    if args.only_z:
        input_dim = 1

    # Initialize the MLP network
    mlp = NeuralStyleField(
        args.sigma,
        args.depth,
        args.width,
        "gaussian",
        args.colordepth,
        args.normdepth,
        args.normratio,
        args.clamp,
        args.normclamp,
        niter=args.n_iter,
        progressive_encoding=args.pe,
        input_dim=input_dim,
        exclude=args.exclude,
    ).to(device)

    # Initialize the normals network
    mlp.reset_weights()

    # Optimizer settings
    # Use Adam optimizer
    optim = torch.optim.Adam(
        mlp.parameters(), args.learning_rate, weight_decay=args.decay
    )

    # Learning rate scheduler
    activate_scheduler = (
        args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    )

    # Use learning rate plateau scheduler
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=args.decay_step, gamma=args.lr_decay
        )

    if args.checkpoint_path is not None:
        mlp, optim, lr_scheduler, last_loss = load_model(
            mlp, optim, lr_scheduler, args.checkpoint_path
        )
        losses.append(last_loss)

    # Handle text prompt
    if not args.no_prompt:
        if args.prompt:
            prompt = " ".join(args.prompt)
            prompt_token = clip.tokenize([prompt]).to(device)
            encoded_text: torch.Tensor = clip_model.encode_text(prompt_token)

            # Save prompt to file
            with open(os.path.join(dir, prompt), "w") as f:
                f.write("")

            # Same with normprompt
            norm_encoded = encoded_text

    # Handle normalization prompt
    if args.normprompt is not None:
        prompt = " ".join(args.normprompt)
        prompt_token = clip.tokenize([prompt]).to(device)
        norm_encoded = clip_model.encode_text(prompt_token)

        # Save prompt to file
        with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
            f.write("")

    # Handle image prompt
    if args.image:
        img = Image.open(args.image)
        img = preprocess(img).to(device)
        encoded_image = clip_model.encode_image(img.unsqueeze(0))

        if args.no_prompt:
            norm_encoded = encoded_image

    # Initialize variables for the training loop
    loss_check = None  # Loss check value
    vertices = copy.deepcopy(mesh.vertices)  # Copy of the mesh vertices
    network_input = copy.deepcopy(vertices)  # Copy of vertices for the network input

    if args.symmetry == True:
        # Symmetry along the X-axis
        network_input[:, 2] = torch.abs(network_input[:, 2])

    if args.standardize == True:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0)) / torch.std(
            network_input, dim=0
        )

    # Choose the appropriate encoded input for updating the mesh
    if args.no_prompt and args.image:
        encoded_input = encoded_image
    elif args.normprompt is not None:
        encoded_input = norm_encoded
    else:
        encoded_input = encoded_text

    # Main training loop
    for epoch in tqdm(range(args.n_iter)):
        optim.zero_grad()  # Zero the gradients

        sampled_mesh = mesh  # Sampled mesh

        # Update the mesh with the network output
        update_mesh(
            mlp, network_input, encoded_input, prior_color, sampled_mesh, vertices
        )

        # Render the front views of the mesh
        rendered_images, elev, azim = render.render_front_views(
            sampled_mesh,
            num_views=args.n_views,
            show=args.show,
            center_azim=args.frontview_center[0],
            center_elev=args.frontview_center[1],
            std=args.frontview_std,
            return_views=True,
            background=background,
        )

        # rendered_images = torch.stack([preprocess(transforms.ToPILImage()(image)) for image in rendered_images])

        if n_augs == 0:
            clip_image = clip_transform(
                rendered_images
            )  # CLIP transform for the rendered images
            encoded_renders = clip_model.encode_image(
                clip_image
            )  # Encode the rendered images
            if not args.no_prompt:
                # Calculate the loss based on the text prompt
                loss = torch.mean(
                    torch.cosine_similarity(encoded_renders, encoded_text)
                )

        # Check augmentation steps
        if (
            args.cropsteps != 0
            and cropupdate != 0
            and epoch != 0
            and epoch % args.cropsteps == 0
        ):
            # Update crop value
            curcrop += cropupdate
            # print(curcrop)

            # Update normal augment transform
            normaugment_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
                    transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                    clip_normalizer,
                ]
            )

        # Apply augmentations if specified
        if n_augs > 0:
            loss = 0.0  # Initialize loss to zero
            for _ in range(n_augs):
                # Normal augment transform the rendered images
                augmented_image = augment_transform(rendered_images)

                # Encode the augmented images using the CLIP model
                encoded_renders = clip_model.encode_image(augmented_image)
                if not args.no_prompt:
                    if args.prompt:
                        if args.clipavg == "view":  # Average view
                            if encoded_text.shape[0] > 1:  # Multiple text prompts
                                # Calculate the loss based on the avg of the text prompts and the rendered images
                                loss -= torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0),
                                    torch.mean(encoded_text, dim=0),
                                    dim=0,
                                )
                            else:  # Single text prompt
                                # Calculate the loss based on the text prompt and the rendered images
                                loss -= torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0, keepdim=True),
                                    encoded_text,
                                )

                        else:  # Not average view
                            # Calculate the loss based on the cosine similarity of the text prompts and the rendered images
                            loss -= torch.mean(
                                torch.cosine_similarity(encoded_renders, encoded_text)
                            )

                # If image is specified
                if args.image:
                    if encoded_image.shape[0] > 1:  # If multiple images
                        # Calculate the loss based on the cosine similarity of the rendered images and the input image
                        loss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0),
                            torch.mean(encoded_image, dim=0),
                            dim=0,
                        )
                    else:  # Single image
                        # Calculate the loss based on the cosine similarity of the rendered images and the input image
                        loss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0, keepdim=True),
                            encoded_image,
                        )

        # Handle seperate loss on the uncolored displacements
        if args.splitnormloss:
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = False

        loss.backward(retain_graph=True)

        # optim.step()

        # with torch.no_grad():
        #     losses.append(loss.item())

        # Normal augment transform
        # loss = 0.0

        if args.n_normaugs > 0:
            normloss = 0.0  # Initialize normal loss to zero
            for _ in range(args.n_normaugs):
                # Apply normal augment transform
                augmented_image = normaugment_transform(rendered_images)

                # Encode the augmented images
                encoded_renders = clip_model.encode_image(augmented_image)

                if not args.no_prompt:
                    if args.prompt:  # If text prompt is specified
                        if args.clipavg == "view":  # Average view
                            if norm_encoded.shape[0] > 1:  # Multiple text prompts
                                # Calculate the loss based on the avg of the text prompts and the rendered images
                                normloss -= normweight * torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0),
                                    torch.mean(norm_encoded, dim=0),
                                    dim=0,
                                )
                            else:  # Single text prompt
                                # Calculate the loss based on the text prompt and the rendered images
                                normloss -= normweight * torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0, keepdim=True),
                                    norm_encoded,
                                )
                        else:  # Not average view
                            # Calculate the loss based on the cosine similarity of the text prompts and the rendered images
                            normloss -= normweight * torch.mean(
                                torch.cosine_similarity(encoded_renders, norm_encoded)
                            )

                # If image is specified
                if args.image:
                    if encoded_image.shape[0] > 1:  # If multiple images
                        # Calculate the loss based on the cosine similarity of the rendered images and the input image
                        loss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0),
                            torch.mean(encoded_image, dim=0),
                            dim=0,
                        )
                    else:  # Single image
                        # Calculate the loss based on the cosine similarity of the rendered images and the input image
                        loss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0, keepdim=True),
                            encoded_image,
                        )

            if args.splitnormloss:  # Split normal loss
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = True

            if args.splitcolorloss:  # Split color loss
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = False

            if not args.no_prompt:  # If text prompt is specified
                normloss.backward(retain_graph=True)

        # Handle geometric loss being used in the training
        if args.geoloss:
            # Initialize the mesh with default color
            default_color = torch.zeros(len(mesh.vertices), 3).to(device)
            # Index vertices by faces
            default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
            # Index vertices by faces
            sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
                default_color.unsqueeze(0), sampled_mesh.faces
            )

            # Render the front views of the mesh
            geo_renders, elev, azim = render.render_front_views(
                sampled_mesh,
                num_views=args.n_views,
                show=args.show,
                center_azim=args.frontview_center[0],
                center_elev=args.frontview_center[1],
                std=args.frontview_std,
                return_views=True,
                background=background,
            )

            if args.n_normaugs > 0:
                normloss = 0.0  # Initialize normal loss to zero
                ### avgview != aug
                for _ in range(args.n_normaugs):
                    # Apply normal augment transform
                    augmented_image = displaugment_transform(geo_renders)

                    # Encode the augmented images
                    encoded_renders = clip_model.encode_image(augmented_image)

                    if norm_encoded.shape[0] > 1:  # If multiple text prompts
                        # Calculate the loss based on the cosine similarity of the text prompts and the rendered images
                        normloss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0),
                            torch.mean(norm_encoded, dim=0),
                            dim=0,
                        )
                    else:  # Single text prompt
                        # Calculate the loss based on the cosine similarity of the text prompts and the rendered images
                        normloss -= torch.cosine_similarity(
                            torch.mean(encoded_renders, dim=0, keepdim=True),
                            norm_encoded,
                        )

                    if args.image:  # If image is specified
                        if encoded_image.shape[0] > 1:  # If multiple images
                            # Calculate the loss based on the cosine similarity of the rendered images and the input image
                            loss -= torch.cosine_similarity(
                                torch.mean(encoded_renders, dim=0),
                                torch.mean(encoded_image, dim=0),
                                dim=0,
                            )
                        else:  # Single image
                            # Calculate the loss based on the cosine similarity of the rendered images and the input image
                            loss -= torch.cosine_similarity(
                                torch.mean(encoded_renders, dim=0, keepdim=True),
                                encoded_image,
                            )
                # Backward pass for the normal loss
                normloss.backward(retain_graph=True)

        # Update the model parameters based on the gradients
        optim.step()

        for (
            param
        ) in mlp.mlp_normal.parameters():  # Normal parameters are enabled for training
            param.requires_grad = True

        for (
            param
        ) in mlp.mlp_rgb.parameters():  # RGB parameters are enabled for training
            param.requires_grad = True

        if activate_scheduler:  # If learning rate scheduler is active
            lr_scheduler.step()

        with torch.no_grad():  # No gradient calculation
            losses.append(loss.item())

        # Adjust normweight if set
        if args.decayfreq is not None:
            # How often the normweight is adjusted
            if epoch % args.decayfreq == 0:
                # Multiply normweight by decay factor
                # Gradually reduce the contribution of the normals network
                normweight *= args.cropdecay

        if epoch % 100 == 0:  # If iteration is a multiple of 100
            report_process(args, dir, epoch, loss, loss_check, losses, rendered_images)

    export_final_results(args, dir, losses, mesh, mlp, network_input, vertices)
    save_model(mlp, optim, lr_scheduler, losses.pop(), "models/")


def report_process(
    args: argparse.Namespace,
    dir: str,
    epoch: int,
    loss: torch.Tensor,
    loss_check: float,
    losses: List[float],
    rendered_images: torch.Tensor,
):
    """Report the progress of the training loop.

    Args:
        args: Command-line arguments.
        dir (str): Output directory path.
        epoch (int): Current iteration number (0-indexed count).
        loss (torch.Tensor): Current loss value.
        loss_check (float): Loss check value.
        losses (List[float]): List of loss values.
        rendered_images (torch.Tensor): Rendered images.
    """
    # Print the current iteration and loss value
    print("iter: {} loss: {}".format(epoch, loss.item()))

    # Save the rendered images for the current iteration
    torchvision.utils.save_image(
        rendered_images, os.path.join(dir, "iter_{}.jpg".format(epoch))
    )

    # Learning rate adjustment for plateau detection
    if args.lr_plateau and loss_check is not None:
        # Calculate the average loss over the last 100 iterations
        new_loss_check = np.mean(losses[-100:])
        # If the average loss increased or plateaued, reduce the learning rate
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g["lr"] *= 0.5
        # Update the loss check value
        loss_check = new_loss_check

    # Initialize the loss check value if not already set and there are enough losses
    elif args.lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])


def export_final_results(
    args: argparse.Namespace,
    dir: str,
    losses: List[float],
    mesh: Mesh,
    mlp: NeuralStyleField,
    network_input: torch.Tensor,
    vertices: torch.Tensor,
):
    """Export the final results of the training process.

    Args:
        args (argparse.Namespace or dict): Command-line arguments.
        dir (str): Output directory.
        losses (List[float]): List of loss values over the training period.
        mesh (Mesh): The mesh object being trained on.
        mlp (NeuralStyleField): The trained Multi-Layer Perceptron (MLP) model.
        network_input (torch.Tensor): Input to the MLP for generating predictions.
        vertices (torch.Tensor): Original vertices of the mesh.
    """
    # Ensure no gradients are calculated during the export process
    with torch.no_grad():
        # Get the RGB and normal predictions from the MLP
        pred_rgb, pred_normal = mlp(network_input)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        # Save the predicted RGB and normal values
        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        # Create a base color tensor with a default value of 0.5
        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        # Clamp the final color values between 0 and 1
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        # Update mesh vertices using the predicted normals
        mesh.vertices = (
            vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal
        )

        # Extract the base name and extension of the object file path
        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        # Export the final mesh with the updated vertices and colors
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)

        # If save_render flag is set, render and save the results
        if args.save_render:
            save_rendered_results(args, dir, final_color, mesh)

        # Save the list of loss values
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))


def save_rendered_results(
    args: Union[argparse.Namespace, dict],
    dir: str,
    final_color: torch.Tensor,
    mesh: Mesh,
):
    """Save rendered results of the mesh before and after applying final colors.

    Args:
        args (argparse.Namespace or dict): Command-line arguments.
        dir (str): Output directory.
        final_color (torch.Tensor): Final color predictions for the mesh.
        mesh (Mesh): The mesh object being rendered.
    """
    # Create a default color tensor with a fill value of 0.5
    default_color = torch.full(
        size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device
    )

    # Assign default colors to the mesh faces
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        default_color.unsqueeze(0), mesh.faces.to(device)
    )

    # Initialize the renderer
    kal_render = Renderer(
        mesh=mesh,
        camera=kal.render.camera.generate_perspective_projection(
            np.pi / 4, 1280 / 720
        ).to(device),
        dim=(1280, 720),
    )

    # Normalize the mesh vertices
    MeshNormalizer(mesh)()

    # Render the mesh with default colors
    img, mask = kal_render.render_single_view(
        mesh,
        args.frontview_center[1],
        args.frontview_center[0],
        radius=2.5,
        background=torch.tensor([1, 1, 1]).to(device).float(),
        return_mask=True,
    )
    img = img[0].cpu()
    mask = mask[0].cpu()

    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))

    # Normalize the mesh vertices again
    MeshNormalizer(mesh)()

    # Assign final colors to the mesh faces
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
        final_color.unsqueeze(0).to(device), mesh.faces.to(device)
    )

    # Render the mesh with final colors
    img, mask = kal_render.render_single_view(
        mesh,
        args.frontview_center[1],
        args.frontview_center[0],
        radius=2.5,
        background=torch.tensor([1, 1, 1]).to(device).float(),
        return_mask=True,
    )
    img = img[0].cpu()
    mask = mask[0].cpu()

    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))


def update_mesh(
    mlp: NeuralStyleField,
    network_input: torch.Tensor,
    text_encoding: torch.Tensor,
    prior_color: torch.Tensor,
    sampled_mesh: Mesh,
    vertices: torch.Tensor,
):
    """Update the mesh with new predictions from the MLP.

    Args:
        mlp (NeuralStyleField): The trained Multi-Layer Perceptron (MLP) model.
        network_input (torch.Tensor): Input to the MLP for generating predictions.
        text_encoding (torch.Tensor): Encoded text prompt.
        prior_color (torch.Tensor): Prior color values for the mesh.
        sampled_mesh (Mesh): The mesh object being updated.
        vertices (torch.Tensor): Original vertices of the mesh.
    """

    # Get the RGB and normal predictions from the MLP
    # pred_rgb is the predicted RGB values
    # pred_normal is used to update the vertex positions of the mesh.
    # - If pred_normal is zero, the vertices remain unchanged.
    # - If pred_normal has positive or negative values,
    #       the vertices move along the direction of their normals,
    #       either outward or inward, respectively.
    pred_rgb, pred_normal = mlp(network_input, text_encoding)

    # Update the mesh face attributes with the predicted RGB values
    sampled_mesh.face_attributes = (
        prior_color
        + kaolin.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0), sampled_mesh.faces
        )
    )

    # Update the mesh vertices with the predicted normals
    sampled_mesh.vertices = vertices + sampled_mesh.vertex_normals * pred_normal

    # Normalize the mesh vertices
    MeshNormalizer(sampled_mesh)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint file (optional)",
    )

    parser.add_argument(
        "--obj_path",
        type=str,
        default="meshes/mesh1.obj",
        help="Path to the input OBJ file",
    )

    parser.add_argument(
        "--prompt",
        nargs="+",
        default="a pig with pants",
        help="Text prompt describing the desired output",
    )

    parser.add_argument(
        "--normprompt",
        nargs="+",
        default=None,
        help="Normal prompt (if any) for the normal network",
    )

    parser.add_argument(
        "--promptlist", nargs="+", default=None, help="List of text prompts"
    )

    parser.add_argument(
        "--normpromptlist", nargs="+", default=None, help="List of normal prompts"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to the input image file (optional)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="round2/alpha5",
        help="Directory to save the output files",
    )

    parser.add_argument(
        "--traintype",
        type=str,
        default="shared",
        help="Type of training (shared or separate)",
    )

    parser.add_argument(
        "--sigma", type=float, default=10.0, help="Sigma value for the color network"
    )

    parser.add_argument(
        "--normsigma",
        type=float,
        default=10.0,
        help="Sigma value for the normals network",
    )

    parser.add_argument("--depth", type=int, default=4, help="Depth of the MLP network")

    parser.add_argument(
        "--width", type=int, default=256, help="Width of the MLP network"
    )

    parser.add_argument(
        "--colordepth", type=int, default=2, help="Depth of the color network"
    )

    parser.add_argument(
        "--normdepth", type=int, default=2, help="Depth of the normals network"
    )

    parser.add_argument(
        "--normwidth", type=int, default=256, help="Width of the normals network"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate for the optimizer",
    )

    parser.add_argument(
        "--normal_learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate for the normals network",
    )

    parser.add_argument(
        "--decay", type=float, default=0, help="Weight decay for the optimizer"
    )

    parser.add_argument(
        "--lr_decay", type=float, default=1, help="Learning rate decay factor"
    )

    parser.add_argument(
        "--lr_plateau", action="store_true", help="Use learning rate plateau scheduler"
    )

    parser.add_argument(
        "--no_pe",
        dest="pe",
        default=True,
        action="store_false",
        help="Do not use positional encoding",
    )

    parser.add_argument(
        "--decay_step", type=int, default=100, help="Learning rate decay step"
    )

    parser.add_argument(
        "--n_views", type=int, default=5, help="Number of views to render"
    )

    parser.add_argument("--n_augs", type=int, default=0, help="Number of augmentations")

    parser.add_argument(
        "--n_normaugs", type=int, default=0, help="Number of normal augmentations"
    )

    parser.add_argument("--n_iter", type=int, default=6000, help="Number of iterations")

    parser.add_argument(
        "--encoding",
        type=str,
        default="gaussian",
        help="Encoding type for the MLP network (gaussian or positional)",
    )

    parser.add_argument(
        "--normencoding",
        type=str,
        default="xyz",
        help="Encoding type for normals network (gaussian or positional)",
    )

    parser.add_argument(
        "--layernorm", action="store_true", help="Use layer normalization"
    )

    parser.add_argument(
        "--run", type=str, default=None, help="Run identifier (optional)"
    )

    parser.add_argument("--gen", action="store_true", help="Generate output")

    parser.add_argument(
        "--clamp",
        type=str,
        default="tanh",
        help="Clamping function for the MLP network",
    )

    parser.add_argument(
        "--normclamp",
        type=str,
        default="tanh",
        help="Clamping function for the normals network",
    )

    parser.add_argument(
        "--normratio",
        type=float,
        default=0.1,
        help="Ratio of normal network contribution",
    )

    parser.add_argument("--frontview", action="store_true", help="Use front view")

    parser.add_argument(
        "--no_prompt", default=False, action="store_true", help="Do not use text prompt"
    )

    parser.add_argument(
        "--exclude", type=int, default=0, help="Number of layers to exclude"
    )

    # Training settings
    parser.add_argument(
        "--frontview_std",
        type=float,
        default=8,
        help="Standard deviation for the front view",
    )

    parser.add_argument(
        "--frontview_center",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        help="Center for the front view",
    )
    parser.add_argument(
        "--clipavg", type=str, default=None, help="Average type for CLIP"
    )
    parser.add_argument("--geoloss", action="store_true", help="Use geometry loss")
    parser.add_argument(
        "--samplebary", action="store_true", help="Sample barycentric coordinates"
    )
    parser.add_argument(
        "--promptviews", nargs="+", default=None, help="List of views for the prompt"
    )
    parser.add_argument("--mincrop", type=float, default=1, help="Minimum crop value")
    parser.add_argument("--maxcrop", type=float, default=1, help="Maximum crop value")
    parser.add_argument(
        "--normmincrop",
        type=float,
        default=0.1,
        help="Minimum crop value for the normals network",
    )
    parser.add_argument(
        "--normmaxcrop",
        type=float,
        default=0.1,
        help="Maximum crop value for the normals network",
    )
    parser.add_argument(
        "--splitnormloss", action="store_true", help="Split normal loss"
    )
    parser.add_argument(
        "--splitcolorloss", action="store_true", help="Split color loss"
    )
    parser.add_argument("--nonorm", action="store_true", help="Do not use normals")

    parser.add_argument("--cropsteps", type=int, default=0, help="Crop steps")

    parser.add_argument("--cropforward", action="store_true", help="Crop forward")

    parser.add_argument("--cropdecay", type=float, default=1.0, help="Crop decay")

    parser.add_argument("--decayfreq", type=int, default=None, help="Decay frequency")

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    parser.add_argument("--show", action="store_true", help="Show render")

    parser.add_argument(
        "--background", nargs=3, type=float, default=None, help="Background color"
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )

    parser.add_argument("--save_render", action="store_true", help="Save render")

    parser.add_argument(
        "--input_normals", default=False, action="store_true", help="Use input normals"
    )

    parser.add_argument(
        "--symmetry", default=False, action="store_true", help="Use symmetry"
    )

    parser.add_argument(
        "--only_z", default=False, action="store_true", help="Use only Z"
    )

    parser.add_argument(
        "--standardize", default=False, action="store_true", help="Standardize input"
    )

    # CLIP model settings
    parser.add_argument(
        "--clipmodel", type=str, default="ViT-B/32", help="CLIP model type"
    )

    parser.add_argument("--jit", action="store_true", help="Use JIT compilation")

    args = parser.parse_args()

    run_branched(args)
