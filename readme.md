<img src=/docs/imgs/DFlat_Long.png alt="Dflat" width="500"/>
<img src=/docs/imgs/autoGDS_metalens.png alt="Dflat" width="500"/>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
<!-- [![Documentation Status](https://readthedocs.org/projects/dflat-master/badge/?version=latest)](https://dflat-master.readthedocs.io/en/latest/?badge=latest) --> 

# An End-to-End Design Framework for Diffractive Optics and Metasurface-Based Vision Systems
`D-Flat` is an auto-differentiable design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing tasks. This package was officially released alongside our paper,  <a href="https://deanhazineh.github.io/publications/Multi_Image_Synthesis/combined_paper.pdf" target="_blank"> Polarization Multi-Image Synthesis with Birefringent Metasurfaces</a>, published in the proceedings of the 2023 IEEE International Conference of Computational Photography (ICCP). The package is further documented and discussed in the manuscript available on <a href="https://arxiv.org/abs/2207.14780" target="_blank">arxiv</a>. If you use this package, please cite the ICCP paper (See below for details). 

D-Flat provides users with:
- A validated, auto-differentiable framework for optical field propagation and rendering built on tensorflow Keras
- Pre-trained, efficient neural models to describe the optical response of metasurface cells
- An auto-differentiable field solver (RCWA) packaged as a TF-Keras layer that is easy to use

By treating optical layers in the same fashion as standard, differentiable neural layers, deep learning pipelines can be built to simultaneously optimize optical hardware and ML computational back-ends for the next generation of computational imaging devices.

## Usage and Documentation:

For usage and documentation, a readthedocs page is in active development. You may also visit the code pages for projects that use DFlat to learn more (example: https://github.com/DeanHazineh/Multi-Image-Synthesis). A few tutorial notebooks (google collab) as a demo are also provided in `DFlat/examples/`. Additional examples will be provided in the future (we welcome community made examples sent by email).

For developers and researchers,

- a script to train neural models can be found in `DFlat/dflat/neural_optical_layer/core/trainer_models.py`
- a script to build a cell library using RCWA_TF can be found in `DFlat/dflat/cell_library_generation/generate_cell_library.py`.

### (a) install and run DFlat in Google collab:

Note that DFlat can be easily installed and used in the cloud on Google Collab if desired. This is a potential solution for mac users who wish to use the code without access to a windows/linux compute cluster,  as tensorflow_gpu is not supported by mac.
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

## Future Notes:

- Please note that Tensorflow v2.10 is the last official tensorflow release with native GPU compatibility for Windows OS. DFlat is currently kept on v2.10. In the future, DFlat may be ugraded to the newest Tensorflow version.
- Lastly, we are also considering a port of the package to Pytorch which has now become the dominant ML framework. At that time, DFlat on tensorflow will still be available but will no longer be updated.

## Credits and Acknowledgements:
If you utilize DFlat or included data sets for your own work, please cite it by clicking the github citation link on the right or by copying:

```
@INPROCEEDINGS{Hazineh2023,
  Author = {Dean Hazineh and Soon Wei Daniel Lim and Qi Guo and Federico Capasso and Todd Zickler},
  booktitle = {2023 IEEE International Conference on Computational Photography (ICCP)}, 
  Title = {Polarization Multi-Image Synthesis with Birefringent Metasurfaces},
  Year = {2023},
}
```

This work pulls inspiration from, builds on, or otherwise ports/modifies previous open-source contributions from the following individuals:
 * Shane Colburn - RCWA_TF
 * Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega - Pyhank
 * Petra Vidnerova - rbf_for_tf2; 
 
This work was supported by the National Science Foundation (IIS). We thank Professors Todd Zickler, Qi Guo, and Federico Capasso for their role in the conception of this software. We also thank Dr. Zhujun Shi for contributions to the conception and early development and Soon Wei Daniel Lim for help with validations.

## Contact:
This repository is intended to be accessible and community driven. It may not be fully error-proof.
If you have improvements, fixes, or contributions, please branch and initiate a merge request to master (or email me)!

For any questions, functionality requests, or other concerns, don't hesitate to contact me at dhazineh@g.harvard.edu. 

