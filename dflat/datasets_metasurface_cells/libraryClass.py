import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import dflat.plot_utilities.graphFunc as graphFunc
from .core.util_library_lookup import *

listLibraryNames = [
    "Nanofins_U350nm_H600nm",
    "Nanocylinders_U180nm_H600nm",
    "Nanoellipse_U350nm_H600nm",
]


def get_path_to_data(file_name: str):
    ## This is not the accepted solution but it should work for bootstrapping research with few users
    resource_path = Path(__file__).parent / "raw_meta_libraries"
    return resource_path.joinpath(file_name)


class Nanofins_U350nm_H600nm:
    def __init__(self):
        __rawPath = get_path_to_data("data_Nanofins_Unit350nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenx, leny, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, and param3 all have units of m
        param1 = data["lenx"]
        param2 = data["leny"]
        param3 = data["wavelength_m"].flatten()
        param1, param2, param3 = np.meshgrid(param1, param2, param3)
        self.params = [param1, param2, param3]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]

    def plotLibrary(self, savepath=[]):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test.
        # It is also a test that we have flattened and packaged things in the way that is expected when later loading
        # the training data

        # we keep it here with a plt.show() command so that at any point in another script, we can call and check the library we are using
        # without digging up the saved png
        lx = self.params[0][0, :, 0] * 1e9
        ly = self.params[1][:, 0, 0] * 1e9
        wavelength = self.params[2][0, 0, :] * 1e9

        num_plt = 5
        wl_idx_set = np.linspace(0, len(wavelength) - 1, num_plt).astype(int)
        fig = plt.figure(figsize=(40*.75, 30*.75))
        axisList = graphFunc.addAxis(fig, 4, num_plt)

        for iter, idx in enumerate(wl_idx_set):
            tx = axisList[iter].imshow(self.transmission[0, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1)
            phix = axisList[num_plt + iter].imshow(
                self.phase[0, :, :, idx],
                extent=(min(lx), max(lx), max(ly), min(ly)),
                vmin=-np.pi,
                vmax=np.pi,
                cmap="hsv",
            )

            ty = axisList[num_plt * 2 + iter].imshow(
                self.transmission[1, :, :, idx], extent=(min(lx), max(lx), max(ly), min(ly)), vmin=0, vmax=1
            )
            phiy = axisList[num_plt * 3 + iter].imshow(
                self.phase[1, :, :, idx],
                extent=(min(lx), max(lx), max(ly), min(ly)),
                vmin=-np.pi,
                vmax=np.pi,
                cmap="hsv",
            )

            graphFunc.formatPlots(
                fig,
                axisList[iter],
                tx,
                title=f"lambda (nm): {wavelength[idx]:3.0f}",
                ylabel="x-Trans \n Fin width y" if iter == 0 else "",
                rmvxLabel=True,
                rmvyLabel=False if iter == 0 else True,
                addcolorbar=True if iter == 4 else False,
            )
            graphFunc.formatPlots(
                fig,
                axisList[num_plt + iter],
                phix,
                ylabel="x-phase \n Fin width y" if iter == 0 else "",
                rmvxLabel=True,
                rmvyLabel=False if iter == 0 else True,
                addcolorbar=True if iter == 4 else False,
            )
            graphFunc.formatPlots(
                fig,
                axisList[num_plt * 2 + iter],
                ty,
                ylabel="y-Trans \n Fin width y" if iter == 0 else "",
                rmvxLabel=True,
                rmvyLabel=False if iter == 0 else True,
                addcolorbar=True if iter == 4 else False,
            )
            graphFunc.formatPlots(
                fig,
                axisList[num_plt * 3 + iter],
                phiy,
                ylabel="y-Phase \n Fin width y" if iter == 0 else "",
                xlabel="Fin width x",
                rmvyLabel=False if iter == 0 else True,
                addcolorbar=True if iter == 4 else False,
            )

        if savepath:
            plt.savefig(savepath + ".png")
            plt.savefig(savepath + ".pdf")
        else:
            plt.show()

        return

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape):
        ### Given list of target transmission and phase profiles (and list of wavelengths they are used for),
        ### assemble the best shape vector that most minimizes the complex error between target profiles and true profiles
        # Note this is not the single lens that best realizes the trans and phase across all wavelengths at once.
        # That type of design requires optimization rather than naive forward assembly

        ### Run input assertions
        length = len(wavelength_asList)
        # List assertion
        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")
        # List length assertion
        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")
        # Assert polarization basis dimension is two
        if not all([trans.shape[0] == 2 for trans in trans_asList]) or not all([phase.shape[0] == 2 for phase in phase_asList]):
            raise ValueError(
                "optical_response_to_param: All transmission/phase profiles in the list must be a stack of two profiles, (2, Ny, Nx)"
            )

        ### Get the library data
        phaseTable = self.phase
        transTable = np.clip(self.transmission, 0.0, 1.0)
        lx = self.params[0][:, :, 0].flatten()
        ly = self.params[1][:, :, 0].flatten()
        wavelength = self.params[2][0, 0, :]

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = ms_trans[0:1].shape

            design_lx, design_ly = lookup_D2_pol2(phaseTable, transTable, lx, ly, wavelength, use_wavelength, ms_trans, ms_phase)

            # Define a normalized shape vector for convenience
            norm_design_lx = np.clip((design_lx - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0]), 0, 1)
            norm_design_ly = np.clip((design_ly - self.__param2Limits[0]) / (self.__param2Limits[1] - self.__param2Limits[0]), 0, 1)

            if reshape:
                shape_Vector.append(
                    np.vstack(
                        (
                            np.reshape(design_lx, initial_shape),
                            np.reshape(design_ly, initial_shape),
                        )
                    )
                )

                shape_Vector_norm.append(
                    np.vstack(
                        (
                            np.reshape(norm_design_lx, initial_shape),
                            np.reshape(norm_design_ly, initial_shape),
                        )
                    )
                )
            else:
                shape_Vector.append(np.hstack((np.expand_dims(design_lx, -1), np.expand_dims(design_ly, -1))))
                shape_Vector_norm.append(np.hstack((np.expand_dims(norm_design_lx, -1), np.expand_dims(norm_design_ly, -1))))

        return shape_Vector, shape_Vector_norm


class Nanocylinders_U180nm_H600nm:
    def __init__(self):

        # __rawPath = "dflat/datasets_metasurface_cells/raw_meta_libraries/data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat"
        __rawPath = get_path_to_data("data_Nanocylinders_Unit180nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [wavelength=441, lenr=191]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenr, wavelength) are loaded in here
        # all parameters must be converted to meshgrid format for compatibility with model class
        # param1, param2, all have units of m
        param1, param2 = np.meshgrid(data["radius_m"], data["wavelength_m"])
        self.params = [param1, param2]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [30e-9, 150e-9]
        self.__param2Limits = [310e-9, 750e-9]

    def plotLibrary(self, savepath=[]):
        # This function is added so the user may get familiar with the data themselves
        # and poke around the visualization as a test
        lr = self.params[0][0, :] * 1e9
        wl = self.params[1][:, 0] * 1e9

        fig = plt.figure(figsize=(25, 10))
        axisList = graphFunc.addAxis(fig, 1, 2)
        tt = axisList[0].imshow(self.transmission[:, :], extent=(min(lr), max(lr), max(wl), min(wl)), vmin=0, vmax=1)
        phi = axisList[1].imshow(self.phase[:, :], extent=(min(lr), max(lr), max(wl), min(wl)), cmap="hsv")
        graphFunc.formatPlots(fig, axisList[0], tt, "len r (nm)", "wavelength (nm)", "phase", addcolorbar=True)
        graphFunc.formatPlots(fig, axisList[1], phi, "len r (nm)", "wavelength (nm)", "transmission", addcolorbar=True)

        if savepath:
            plt.savefig(savepath + ".png")
            plt.savefig(savepath + ".pdf")
        else:
            plt.show()

        return

    def optical_response_to_param(self, trans_asList, phase_asList, wavelength_asList, reshape):
        ### Given list of target transmission and phase profiles (and list of wavelengths they are used for),
        ### assemble the best shape vector that most minimizes the complex error between target profiles and true profiles
        # Note this is not the single lens that best realizes the trans and phase across all wavelengths at once.
        # That type of design requires optimization rather than naive forward assembly

        ### Run input assertions
        length = len(wavelength_asList)
        # list assertion
        if not all(type(input) is list for input in [trans_asList, phase_asList, wavelength_asList]):
            raise TypeError("optical_response_to_param: trans, phase, and wavelength must all be passed in as lists")
        # list length assertion
        if not all(len(lst) == length for lst in [trans_asList, phase_asList]):
            raise ValueError("optical_response_to_param: All lists must be the same length")
        # polarization dimensionality check
        if not all([trans.shape[0] == 1 for trans in trans_asList]) or not all([phase.shape[0] == 1 for phase in phase_asList]):
            raise ValueError(
                "optical_response_to_param: All transmission/phase profiles in the list must be a single transmission profile, (1, Ny, Nx)"
            )

        ### Get the library data
        phaseTable = self.phase
        # transTable = np.clip(self.transmission, 0.0, 1.0)
        transTable = self.transmission
        radius = self.params[0][0, :].flatten()
        wavelength = self.params[1][:, 0]

        ### Assemble metasurfaces
        shape_Vector = []
        shape_Vector_norm = []
        for i in range(length):
            use_wavelength = wavelength_asList[i]
            ms_trans = trans_asList[i]
            ms_phase = phase_asList[i]
            initial_shape = ms_trans[0:1].shape

            design_radius = lookup_D1_pol1(phaseTable, transTable, radius, wavelength, use_wavelength, ms_trans, ms_phase)
            norm_design_radius = np.clip((design_radius - self.__param1Limits[0]) / (self.__param1Limits[1] - self.__param1Limits[0]), 0, 1)

            if reshape:
                shape_Vector.append(np.reshape(design_radius, initial_shape))
                shape_Vector_norm.append(np.reshape(norm_design_radius, initial_shape))
            else:
                shape_Vector.append(np.expand_dims(design_radius, -1))
                shape_Vector_norm.append(np.expand_dims(norm_design_radius, -1))

        return shape_Vector, shape_Vector_norm


class Nanoellipse_U350nm_H600nm(Nanofins_U350nm_H600nm):
    def __init__(self):
        super(Nanoellipse_U350nm_H600nm, self).__init__()

        # __rawPath = (
        #    "dflat/datasets_metasurface_cells/raw_meta_libraries/data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat"
        # )
        __rawPath = get_path_to_data("data_NanoEllipse_Unit350nm_Height600nm_EngineFDTD.mat")
        data = scipy.io.loadmat(__rawPath)

        # Phase and transmission has shape [Npol=2, leny=49, lenx=49, wavelength=441]
        self.phase = data["phase"]
        self.transmission = data["transmission"]

        # the input parameters (lenx, leny, wavelength) are loaded here
        # all parameters must be converted to meshgrid format for compatibility with model class
        param1 = data["lenx"]
        param2 = data["leny"]
        param3 = data["wavelength_m"].flatten()
        param1, param2, param3 = np.meshgrid(param1, param2, param3)
        self.params = [param1, param2, param3]

        # These are the min-max for the FDTD gridsweep
        self.__param1Limits = [60e-9, 300e-9]
        self.__param2Limits = [60e-9, 300e-9]
        self.__param3Limits = [310e-9, 750e-9]

