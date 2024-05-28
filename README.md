# Text2Mesh [[Project Page](https://threedle.github.io/text2mesh/)]
[![arXiv](https://img.shields.io/badge/arXiv-Text2Mesh-b31b1b.svg)](https://arxiv.org/abs/2112.03221)
![Pytorch](https://img.shields.io/badge/PyTorch->=1.9.0-Red?logo=pytorch)
![crochet candle](images/vases.gif)
**Text2Mesh** is a method for text-driven stylization of a 3D mesh, as described in "Text2Mesh: Text-Driven Neural Stylization for Meshes" CVPR 2022.

## Getting Started
### Installation

1. **Installing Ubuntu:**
    - Text2Mesh requires Ubuntu: Install [FocalFossa 20.04](https://releases.ubuntu.com/focal/ubuntu-20.04.6-desktop-amd64.iso).
    - Use some iso software to mount that iso to a USB.
        - I recommend using [Rufus](https://rufus.ie/en).
        - Requires a USB with ≥4GB, but use a ≥16GB drive for easiest installation.
    - Create a partition on your device to hold Ubuntu, you shouldn't need more than 64GB for the entire project.
    - Make sure to enable proprietary drivers on installation.
    - [Here](https://www.youtube.com/watch?v=GXxTxBPKecQ&themeRefresh=1) is a link to a good side-load installation video.

2. **Setting Up Environment:**
    - After installation:
        ```bash
        sudo apt update
        sudo apt upgrade -y
        sudo reboot
        ```
    - After reboot:
        ```bash
        sudo apt install -y git python3 python3-pip gcc
        git clone https://github.com/randalhucker/text2mesh
        ```

3. **Installing CUDA-11.3:**
    - Start by following all the steps starting [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#prepare-ubuntu).

    - For step 3.9.3: The desired distro/arch is *ubuntu2004/x86_64*.

    - For step 3.9.4: 
        ```bash
        sudo apt-get install cuda-toolkit-11.3
        ```
    
    - For Post-install Actions:
        - Make sure to run 13.1.1.
        - Also run the *Ubuntu* section of 13.3.1.

3.5. **If you have any version of CUDA besides 11.3:**

    - Removing existing CUDA:

        ```bash
        sudo apt --purge remove "cublas*" "cuda*"
        sudo apt --purge remove "nvidia*"
        rm -rf /usr/local/cuda*
        sudo apt-get autoremove && sudo apt-get autoclean
        sudo reboot
        ```

    - Downloading specific CUDA-11.3 version:

        ```bash
        sudo apt-get install g++freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libomp-dev

        sudo reboot
        ```

4. **Installing the rest of CUDA 11.3:** 

        ```bash
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub ## This ________.pub file may be incorrect, the next line will tell you if the wrong pub key is being used. Just replace this 8 keyed pubkey with the last 8 of the suggested pub key in the log. Ex. Mine was 3bf863cc
        sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
        sudo apt-get update
        sudo apt-get -y install cuda-11.3

        sudo reboot
        ```

    - Set Env Vars
        ```bash
        echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        export CUDA_HOME=/usr/local/cuda-11.3
        export PATH=$CUDA_HOME/bin:$PATH
        source ~/.bashrc
        ```

    - Install CuDNN:
        - Download [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive).
        - You want v8.2.1 (June 7th, 2021) for Cuda 11.x
        - Download *cuDNN Library for Linux (x86_64)*
        - In the folder where you downloaded that tarball...
            ```bash
            tar -xzvf ((filename)).tgz ## Just press tab after tar -xzvf 

            sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.3/include
            sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64/
            sudo chmod a+r /usr/local/cuda-11.3/lib64/libcudnn*

            nvcc -V ## Check to make install worked.
            ```

4. **Installing Miniconda:**
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

    ## Follow prompts to complete installation, make sure to enter yes for both prompts
    source ~/.bashrc
    ```

5. **Note:** The below installation will fail if run on something other than a CUDA GPU machine.
    - In a new terminal:
        ```bash
        git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
        cd kaolin
        pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
        python setup.py develop

        ## Test your install
        python -c "import kaolin; print(kaolin.__version__)"

        sudo reboot
        ```

    - Finally:
    
    ```bash
    cd text2mesh
    conda env create --file text2mesh.yml
    conda activate text2mesh
    ```

If you experience an error installing kaolin saying something like `nvcc not found`, you may need to set your `CUDA_HOME` environment variable to the 11.3 folder i.e. `export CUDA_HOME=/usr/local/cuda-11.3`, then rerunning the installation. 

### System Requirements
- Python 3.7
- CUDA 11
- GPU w/ minimum 8 GB ram

### Run examples
Call the below shell scripts to generate example styles. 
```bash
# cobblestone alien
./demo/run_alien_cobble.sh
# shoe made of cactus 
./demo/run_shoe.sh
# lamp made of brick
./demo/run_lamp.sh
# ...
```
The outputs will be saved to `results/demo`, with the stylized .obj files, colored and uncolored render views, and screenshots during training.

#### Outputs
<p float="center">
<img alt="alien" height="135" src="images/alien.png" width="240"/>
<img alt="alien geometry" height="135" src="images/alien_cobble_init.png" width="240"/>
<img alt="alien style" height="135" src="images/alien_cobble_final.png" width="240"/>
</p>

<p float="center">
<img alt="alien" height="135" src="images/alien.png" width="240"/>
<img alt="alien geometry" height="135" src="images/alien_wood_init.png" width="240"/>
<img alt="alien style" height="135" src="images/alien_wood_final.png" width="240"/>
</p>

<p float="center">
<img alt="candle" height="135" src="images/candle.png" width="240"/>
<img alt="candle geometry" height="135" src="images/candle_init.png" width="240"/>
<img alt="candle style" height="135" src="images/candle_final.png" width="240"/>
</p>

<p float="center">
<img alt="person" height="135" src="images/person.png" width="240"/>
<img alt="ninja geometry" height="135" src="images/ninja_init.png" width="240"/>
<img alt="ninja style" height="135" src="images/ninja_final.png" width="240"/>
</p>

<p float="center">
<img alt="shoe" height="135" src="images/shoe.png" width="240"/>
<img alt="shoe geometry" height="135" src="images/shoe_init.png" width="240"/>
<img alt="shoe style" height="135" src="images/shoe_final.png" width="240"/>
</p>

<p float="center">
<img alt="vase" height="135" src="images/vase.png" width="240"/>
<img alt="vase geometry" height="135" src="images/vase_init.png" width="240"/>
<img alt="vase style" height="135" src="images/vase_final.png" width="240"/>
</p>

<p float="center">
<img alt="lamp" height="135" src="images/lamp.png" width="240"/>
<img alt="lamp geometry" height="135" src="images/lamp_init.png" width="240"/>
<img alt="lamp style" height="135" src="images/lamp_final.png" width="240"/>
</p>

<p float="center">
<img alt="horse" height="135" src="images/horse.png" width="240"/>
<img alt="horse geometry" height="135" src="images/horse_init.png" width="240"/>
<img alt="horse style" height="135" src="images/horse_final.png" width="240"/>
</p>

## Important tips for running on your own meshes
Text2Mesh learns to produce color and displacements over the input mesh vertices. The mesh triangulation effectively defines the resolution for the stylization. Therefore, it is important that the mesh triangles are small enough such that they can accurately potray the color and displacement. If a mesh contains large triangles, the stylization will not contain sufficent resolution (and leads to low quality results). For example, the triangles on the seat of the chair below are too large.

<p align="center">
<img alt="large-triangles" src="images/large-triangles.png" height="25%" width="25%" />
</p>

You should remesh such shapes as a pre-process in to create smaller triangles which are uniformly dispersed over the surface. Our example remeshing script can be used with the following command (and then use the remeshed shape with Text2Mesh):

```
python3 remesh.py --obj_path [the mesh's path] --output_path [the full output path]
```

For example, to remesh a file name called `chair.obj`, the following command should be run:  

```
python3 remesh.py --obj_path chair.obj --output_path chair-remesh.obj
```


## Other implementations
[Kaggle Notebook](https://www.kaggle.com/neverix/text2mesh/) (by [neverix](https://www.kaggle.com/neverix))

## External projects using Text2Mesh
- [Endava 3D Asset Tool](https://www.endava.com/en/blog/Engineering/2022/An-R-D-Project-on-AI-in-3D-Asset-Creation-for-Games) integrates Text2Mesh into their modeling software to create 3D assets for games.

- [Psychedelic Trips Art Gallery](https://www.flickr.com/photos/mcanet/sets/72177720299890759/) uses Text2Mesh to generate AI Art and fabricate (3D print) the results.

## Citation
```
@InProceedings{Michel_2022_CVPR,
    author    = {Michel, Oscar and Bar-On, Roi and Liu, Richard and Benaim, Sagie and Hanocka, Rana},
    title     = {Text2Mesh: Text-Driven Neural Stylization for Meshes},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {13492-13502}
}
```
