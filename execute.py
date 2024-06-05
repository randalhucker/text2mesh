import torch
import torch.nn as nn
import clip
from PIL import Image
import copy
import kaolin as kal

from main import update_mesh
from neural_style_field import NeuralStyleField

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    return checkpoint


# Load the checkpoint
checkpoint_path = "path/to/your/checkpoint.tar"
checkpoint = load_checkpoint(checkpoint_path)

# Reinitialize the model
model = NeuralStyleField(
    sigma=checkpoint["sigma"],
    depth=checkpoint["depth"],
    width=checkpoint["width"],
    encoding=checkpoint["encoding"],
    colordepth=checkpoint["colordepth"],
    normdepth=checkpoint["normdepth"],
    normratio=checkpoint["normratio"],
    clamp=checkpoint["clamp"],
    normclamp=checkpoint["normclamp"],
    niter=checkpoint["niter"],
    input_dim=checkpoint["input_dim"],
    progressive_encoding=checkpoint["progressive_encoding"],
    exclude=checkpoint["exclude"],
    text_encoding_dim=512,  # Assuming text_encoding_dim from CLIP is 512
).to(device)

# Load model state
model.load_state_dict(checkpoint["model_state_dict"])

# Reinitialize the optimizer and scheduler
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01
)  # Adjust as per your optimizer configuration
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=100, gamma=0.1
)  # Adjust as per your scheduler configuration
if checkpoint["scheduler_state_dict"] is not None:
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

# Load the loss (if needed)
loss = checkpoint["loss"]

# Load the CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# New text prompt
new_prompt = "A spiky, rough star"
new_prompt_token = clip.tokenize([new_prompt]).to(device)
with torch.no_grad():
    encoded_new_text = clip_model.encode_text(new_prompt_token)

# Prepare the new mesh
original_mesh = ...  # Load or define your original mesh here
new_mesh = original_mesh.clone().to(device)
network_input = copy.deepcopy(new_mesh.vertices).to(device)
prior_color = torch.full((new_mesh.faces.shape[0], 3, 3), fill_value=0.5).to(device)
sampled_mesh = new_mesh.clone()
vertices = copy.deepcopy(new_mesh.vertices).to(device)

# Update the mesh with the new input
update_mesh(model, network_input, encoded_new_text, prior_color, sampled_mesh, vertices)


def render_mesh(mesh, output_path):
    rendered_images = kal.render(
        mesh, num_views=1, show=True
    )  # Adjust parameters as needed
    # Save the rendered images
    for i, img in enumerate(rendered_images):
        img.save(f"{output_path}/rendered_view_{i}.png")


# Specify the output path
output_path = "path/to/output"

# Render and save the updated mesh
render_mesh(sampled_mesh, output_path)
