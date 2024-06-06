import argparse
from pathlib import Path

import clip
import copy
import torch

from main import export_final_results, update_mesh
from mesh import Mesh
from neural_style_field import NeuralStyleField, load_model
from utils import device


def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Mesh Transformation with CLIP and NeuralStyleField"
    )

    # Input/output settings
    parser.add_argument(
        "--prompt", nargs="+", type=str, default="", help="Text prompt for the model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--obj_path", type=str, required=True, help="Path to the input OBJ file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Directory to save the output"
    )
    parser.add_argument("--save_render", action="store_true", help="Save render")

    # CLIP model settings
    parser.add_argument(
        "--clipmodel", type=str, default="ViT-B/32", help="CLIP model type"
    )
    parser.add_argument(
        "--jit", action="store_true", help="Use JIT compilation for CLIP model"
    )

    # Model hyperparameters
    parser.add_argument(
        "--sigma",
        type=float,
        required=True,
        help="Standard deviation for Gaussian Fourier features",
    )
    parser.add_argument("--depth", type=int, default=4, help="Depth of the base MLP")
    parser.add_argument(
        "--width", type=int, default=256, help="Width of the MLP layers"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="gaussian",
        help="Encoding type ('gaussian' or other)",
    )
    parser.add_argument(
        "--colordepth", type=int, default=2, help="Depth of the color branch"
    )
    parser.add_argument(
        "--normdepth", type=int, default=2, help="Depth of the normal branch"
    )
    parser.add_argument(
        "--normratio", type=float, default=0.1, help="Scaling factor for normals"
    )
    parser.add_argument(
        "--clamp",
        type=str,
        default="tanh",
        help="Clamping method for colors ('tanh' or 'clamp')",
    )
    parser.add_argument(
        "--normclamp",
        type=str,
        default="tanh",
        help="Clamping method for normals ('tanh' or 'clamp')",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=6000,
        help="Number of iterations for progressive encoding",
    )
    parser.add_argument(
        "--input_dim", type=int, default=3, help="Dimensionality of the input"
    )
    parser.add_argument(
        "--no_pe",
        dest="pe",
        default=True,
        action="store_false",
        help="Do not use positional encoding",
    )
    parser.add_argument(
        "--exclude",
        type=int,
        default=0,
        help="Exclusion parameter for Fourier features",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        required=False,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="Weight decay for the optimizer"
    )

    return parser.parse_args()


def load_clip_model(clip_model_name, device, jit=False):
    """Load the CLIP model."""
    clip_model, preprocess = clip.load(clip_model_name, device, jit=jit)
    return clip_model, preprocess


def prepare_text_prompt(prompt, clip_model):
    """Encode the text prompt using the CLIP model."""
    prompt = " ".join(prompt)
    prompt_token = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        encoded_text = clip_model.encode_text(prompt_token)
    return encoded_text


def execute():
    args = parse_args()

    # Create the output directory if it does not exist
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Load the CLIP model
    clip_model, preprocess = load_clip_model(args.clipmodel, device, jit=args.jit)

    # Prepare the text prompt
    encoded_text = prepare_text_prompt(args.prompt, clip_model)

    # Initialize the model
    model = NeuralStyleField(
        clamp=args.clamp,
        normclamp=args.normclamp,
        sigma=args.sigma,
        depth=args.depth,
        width=args.width,
        encoding=args.encoding,
        colordepth=args.colordepth,
        normdepth=args.normdepth,
        normratio=args.normratio,
        niter=args.niter,
        input_dim=args.input_dim,
        progressive_encoding=args.pe,
        exclude=args.exclude,
        text_encoding_dim=512,  # Assuming text_encoding_dim from CLIP is 512
    ).to(device)

    # Initialize the optimizer
    optim = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.decay
    )

    # Load the mesh
    mesh = Mesh(args.obj_path)

    # Prepare inputs for the update function
    vertices = copy.deepcopy(mesh.vertices)  # Copy of the mesh vertices
    network_input = copy.deepcopy(vertices)  # Copy of vertices for the network input
    prior_color = torch.full(
        size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device
    )

    # Load the model from checkpoint
    _, _, _, _ = load_model(model, optim, args.model_path)

    # Update the mesh with the new input
    update_mesh(model, network_input, encoded_text, prior_color, mesh, vertices)

    # Export the final results
    export_final_results(
        args, args.output_path, mesh, model, network_input, encoded_text, vertices
    )


if __name__ == "__main__":
    execute()
