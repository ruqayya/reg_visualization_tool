<h1 align="center">Visualization Tool for Registered WSIs.</h1>
<p align="center">
  <img src="https://github.com/ruqayya/reg_visualization_tool/blob/main/doc/interface_snapshot.png">
</p>

This is a web-based visualization tool which enables the user to visualise registered images while being able to zoom in and out and pan across a pair of WSIs. It was developed as a part of my PhD thesis and is made publicly available with the acceptance of our paper titled "Deep Feature based Cross-slide Registration" in Computerized Medical Imaging and Graphics. The implementation is carried out in Python and JavaScript. OpenSeadragon, an open-source viewer, is used for this tool.

### Features

This tool comprises a split screen as shown in the above figure. The left and right panels display the reference and registered moving images. On each split screen, a dot pointer with a different colour is shown. This changes its position with the mouse movement. The regions indicated by these points on the two screens are extremely helpful in visually estimating the performance of the registration method. 

This visualization tool applies the transformation to tiles on the fly as they are viewed. Therefore, there is no need to generate a transformed WSI in a pyramidal format. One can use the 'Fix Offset' button to deal with local translative misalignment/offset. The phase correlation method is used for this purpose. Once the offset is computed, it is applied to every FOV as the user zooms or pans through the slide.

### Inputs

The input to this tool comprises three directory paths: to reference and moving images and the pre-computed affine transformation parameters (3x3 matrix). One can use [TIAToolbox]() for estimating affine transformation which uses Deep Feature Based Registration (DFBR) [1] method for registering an image pair. [Here](https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/10-wsi-registration.html) is an example demonstrating the usage of TIAToolbox for registration. 

[1] Awan, Ruqayya, et al. "Deep Feature based Cross-slide Registration." arXiv preprint arXiv:2202.09971 (2022).


### Usage

In [data folder](https://github.com/ruqayya/reg_visualization_tool/tree/main/data), a pair of unregistered images and their corresponding transformation parameters are provided for the user to experience this visualization tool. Please use the following commands to use this tool in your local system.

1. Open a terminal window<br/>

```sh
    cd <future-home-of-tool-directory>
```

2. Download a complete copy of this tool.

```sh
    git clone https://github.com/ruqayya/reg_visualization_tool.git
```

3. Change directory to `reg_visualization_tool`

```sh
    cd reg_visualization_tool
```

4. Create a virtual environment for this tool using

```sh
    python -m venv reg_vis_tool_env
    reg_vis_tool_env\Scripts\activate
    pip install -r requirements.txt
```
5. To use the packages installed in the environment, run the command:

```sh
    reg_vis_tool_env\Scripts\activate 
```
6. 
```sh
    python deepzoom_server_COMET.py "data\06-18270_5_A1MLH1_1.tif" "data\06-18270_5_A1MSH2_1.tif" "data\transform_matrix.npy"
```
