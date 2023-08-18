import os
import zipfile
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
import shutil


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        shutil.rmtree("build", ignore_errors=True)
        print("Removed the build directory")


class CustomInstallCommand(install):
    """Customized setuptools install command - unzips the data files."""

    def run(self):
        print("Running custom install command")
        install.run(self)
        self.unzip_data_files()

    def unzip_data_files(self):
        import dflat  # Import your package in order to get the exact location following

        package_dir = os.path.dirname(dflat.__file__)  # Get the directory where your package is installed
        list_zip = [
            "fourier_layer/validation_scripts/heart_singularity_hologram.zip",
            "metasurface_library/core/raw_meta_libraries.zip",
            "neural_optical_layer/core/trained_MLP_models.zip",
            "metasurface_library/core/pregen_lookup_tables.zip",
            "physical_optical_layer/core/material_index.zip",
        ]
        for zippedfold in list_zip:
            data_file = os.path.join(package_dir, zippedfold)
            extract_dir = os.path.dirname(data_file)

            zip_ref = zipfile.ZipFile(data_file, "r")
            zip_ref.extractall(extract_dir)
            zip_ref.close()


setup(
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        "install": CustomInstallCommand,
        "clean": CleanCommand,
    },
)
