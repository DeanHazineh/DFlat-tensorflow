# DFlat - V1.1.3
<img src=/docs/imgs/DFlat_Long.png alt="Dflat" width="500"/>

# An End-to-End Design Framework for Metasurface-Based Visual Sensors (Python Tensorflow)
`D-Flat` is a forward and inverse design framework for flat optics, specially geared to the design of dielectric metasurfaces for imaging and sensing. This package is reviewed in paper https://arxiv.org/abs/2207.14780. D-Flat provides users with: 
- A validated, autodifferentiable framework for optical field propagation
- Pre-trained, efficient neural models to describe the optical response of metasurface cells
- A community driven, maintained framework for rendering and end-to-end design

## Usage and Documentation: 
For usage and documentation, a readthedocs page is in active development. Examples for inverse design are provided in `DFlat/examples/`. Additional examples will be provided in the future (we welcome community made examples).

For developers and researchers, 
- a script to train neural models can be found in `DFlat/dflat/neural_optical_layer/core/runtraining_neural_models.py`
- a script to build a cell library using RCWA_TF can be found in `DFlat/dflat/cell_library_generation/generate_cell_library.py`

### (a) install and run DFlat in Google collab:
DFlat can be easily installed in used in the cloud on Google Collab. This is ideal for mac os users as tensorflow_gpu is not supported by mac. 
Google collab notebooks for the two current examples can be accessed at the links:
 - <a href="https://colab.research.google.com/drive/1CVZnfwPmyd6V2qdYSXI5vShGgJecMENX?usp=sharing" target="_blank">achromatic_metalens_rcwa.py</a>
 - <a href="https://colab.research.google.com/drive/1cOeSNBQ4vS6xNZlOPBQhMdcViHQHclyi?usp=sharing" target="_blank">metalens_neural.py</a>

### (b) install and run locally:
To use DFlat on your own machine, first download the repository:
```
git clone https://github.com/DeanHazineh/DFlat
```
Note that git LFS should be installed if not already via `git lfs install` at the terminal. Next install the local repository to your venv by entering the following in terminal:
```
python setup.py develop
```
or (above the root)
```
pip install -e DFlat
```
Install additional dependencies via
```
pip install -r requirements.txt
```


## Contact:
This repository is intended to be accessible and community driven. It may not be fully error-proof and will be continually updated.
If you have improvements, fixes, or contributions, please branch and initiate a merge request to master (or email me)!

For any questions, functionality requests, or other concerns, don't hesitate to contact me at dhazineh@g.harvard.edu. 

## Acknowledgements and Support:
This work pulls inspiration from, builds on, or otherwise ports/modifies previous open-source contributions from the following individuals:
 * Shane Colburn - RCWA_TF
 * Manuel Guizar-Sicairos and Julio C. Guitierrez-Vega - Pyhank
 * Petra Vidnerova - rbf_for_tf2
 
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
