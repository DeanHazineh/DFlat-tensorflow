import os
import zipfile
from setuptools import setup
from setuptools.command.install import install


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
        ]
        for zippedfold in list_zip:
            data_dir = os.path.join(package_dir, zippedfold)  # Create the path to the data directory
            if not os.path.exists(data_dir[:-4]):
                os.makedirs(data_dir[:-4])

            zip_ref = zipfile.ZipFile(data_dir, "r")
            zip_ref.extractall(data_dir[:-4])
            zip_ref.close()


setup(
    cmdclass={
        "install": CustomInstallCommand,
    }
)
