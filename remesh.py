import argparse
import pymeshlab

# Initialize argument parser
parser = argparse.ArgumentParser()

# Add argument for the input OBJ file path
parser.add_argument("--obj_path", type=str, help="Path to the input OBJ file")

# Add argument for the output OBJ file path with a default value
parser.add_argument(
    "--output_path",
    type=str,
    default="./remeshed_obj.obj",
    help="Path to save the remeshed OBJ file",
)

# Parse command-line arguments
args = parser.parse_args()

# Initialize a MeshSet object
ms = pymeshlab.MeshSet()

# Load the input mesh from the specified OBJ file
ms.load_new_mesh(args.obj_path)

# Apply isotropic explicit remeshing to the loaded mesh
ms.meshing_isotropic_explicit_remeshing()

# Save the remeshed mesh to the specified output path
ms.save_current_mesh(args.output_path)
