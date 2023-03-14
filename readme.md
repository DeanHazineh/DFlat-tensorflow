![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/dflat-master/badge/?version=latest)](https://dflat-master.readthedocs.io/en/latest/?badge=latest)

<img src=/docs/imgs/DFlat_Long.png alt="Dflat" width="500"/>
<img src=/docs/imgs/autoGDS_metalens.png alt="Dflat" width="500"/>

# An End-to-End Design Framework for Metasurface-Based Vision Sensors V2.3.2
`D-Flat` is a forward and inverse design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing. This package is reviewed in paper https://arxiv.org/abs/2207.14780. D-Flat provides users with: 
- A validated, auto-differentiable (AD) framework for optical field propagation built on TF Keras
- Pre-trained, efficient neural models to describe the optical response of metasurface cells
- An AD field solver (RCWA) packaged as a TF-Keras layer that is automatically set to optimize pre-defined cell types
- A community driven, maintained framework for rendering and end-to-end design

By treating optical layers in the same fashion as standard, TF neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends. 

## Usage and Documentation: 
For usage and documentation, a readthedocs page is in active development. Examples for inverse design are provided in `DFlat/examples/`. Additional examples will be provided in the future (we welcome community made examples). 

For developers and researchers, 
- a script to train neural models can be found in `DFlat/dflat/neural_optical_layer/core/trainer_models.py`
- a script to build a cell library using RCWA_TF can be found in `DFlat/dflat/cell_library_generation/generate_cell_library.py`.

### (a) install and run DFlat in Google collab:
DFlat can be easily installed in used in the cloud on Google Collab. This is ideal for mac os users as tensorflow_gpu is not supported by mac. 
Google collab versions of current examples can be found in the examples folder and online at the links:


 - <a href="https://colab.research.google.com/drive/1MknLVB6cQ1GQ2xRfHhlAfCQPiWXLHUM3?usp=sharing" target="_blank">Tutorial_I_Dflat_Field_Propagation</a>
 - <a href="https://colab.research.google.com/drive/162Fg0P_QGiddUUeXUrJhikAdy2qpNXpc?usp=sharing" target="_blank">Tutorial_I_RCWA_and_Physical_Layers</a>
 - <a href="https://colab.research.google.com/drive/1a27zLKMXfObyjQDF5nWZ9ug-7jWzmQer?usp=sharing" target="_blank">Tutorial_I_Dflat_Library_and_Neural_Models </a>
 - <a href="https://colab.research.google.com/drive/1uGNU0PsCUunibnkyLZUnGa4Y54vj6XZ3?usp=sharing" target="_blank">Demo_optimize_broadband_rcwa_metalens</a> 
 - <a href="https://colab.research.google.com/drive/1F2KR87CHTBnMHkAHDb04F3FBHE0iUueV?usp=sharing" target="_blank">Demo_optimize_monochromatic_neural_metalens</a>
 - <a href="https://colab.research.google.com/drive/1an1HWkMf0ynw0F1YZx5s82pKVwxw2E0X?usp=sharing" target="_blank">Demo_optimize_dual_polarization_hologram</a>
 
### (b) install and run locally:
To use DFlat on your own machine, first download the repository:
```
git clone https://github.com/DeanHazineh/DFlat
```
Note that git LFS should be installed if not already via `git lfs install` at the terminal. Next install the local repository to your venv by entering the following in terminal:
```
python setup.py develop
```
You can then install additional dependencies via
```
pip install -r requirements.txt
```
Note that you should not just download the zip file from above because this will not download the files hosted on githubs LFS database. 

## Contact:
This repository is intended to be accessible and community driven. It may not be fully error-proof and will be continually updated. 
If you have improvements, fixes, or contributions, please branch and initiate a merge request to master (or email me)!

For any questions, functionality requests, or other concerns, don't hesitate to contact me at dhazineh@g.harvard.edu. 

## Acknowledgements and Support:
This work pulls inspiration from, builds on, or otherwise ports/modifies previous open-source contributions from the following individuals:
 * Shane Colburn - RCWA_TF
 * Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega - Pyhank
 
It also involves some contribution or inspiration from the following: Petra Vidnerova - rbf_for_tf2; 
 
This work was supported by the National Science Foundation (IIS) for End-to-End Computational Sensing.
We thank Professors Todd Zickler, Qi Guo, and Federico Capasso for their role in the conception of this software. For further interest or discussion on research, please direct contact and questions there. We also thank Dr. Zhujun Shi for contributions to the conception and early development and Soon Wei Daniel Lim for help with validations.

## Credits
If you utilize DFlat or included data sets for your own work, please cite it by clicking the github citation link on the right or by copying:
```
  @misc{https://doi.org/10.48550/arxiv.2207.14780,
    doi = {10.48550/ARXIV.2207.14780},
    url = {https://arxiv.org/abs/2207.14780},
    author = {Hazineh, Dean S. and Lim, Soon Wei Daniel and Shi, Zhujun and Capasso, Federico and Zickler, Todd and Guo, Qi},
    keywords = {Optics (physics.optics), Applied Physics (physics.app-ph), FOS: Physical sciences, FOS: Physical sciences},
    title = {D-Flat: A Differentiable Flat-Optics Framework for End-to-End Metasurface Visual Sensor Design},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
  }
```

...
