import tensorflow as tf
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import dflat.plot_utilities.graphFunc as gF


#  BASE CLASS FOR PIPELINES TO USE TRAIN HELPERS
class Pipeline_Object(tf.keras.Model):
    """Baseclass for DFlat custom pipelines, inherits tf.keras.Model structure.

    Attributes:
        `loss_vector` (list): List storing the loss at each epoch after training
        `test_loss_vector` (list): List storing the test loss at each epoch
        `savepath` (str): Pipeline savepath to store model checkpoints, data, and figures.
        `saveAtEpochs` (int): Number of training epochs between intermediate saves.
    """

    def __init__(self, savepath, saveAtEpochs=None):
        """Initialization for base class

        Args:
            `savepath` (str): Pipeline savepath to store model checkpoints, data, and figures.
            `saveAtEpochs` (int): Number of training epochs between intermediate saves.
        """
        super(Pipeline_Object, self).__init__()

        # Define class variables
        self.loss_vector = []
        self.test_loss_vector = []
        self.savepath = savepath
        self.saveAtEpochs = saveAtEpochs

        # create the savepath folder if it does not exist
        self.__checkModelPath()

    def __checkModelPath(self):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Make folders for images too
        if not os.path.exists(self.savepath + "/png_images/"):
            os.makedirs(self.savepath + "/png_images/")
        if not os.path.exists(self.savepath + "/pdf_images/"):
            os.makedirs(self.savepath + "/pdf_images/")

        return

    def customSaveCheckpoint(self, loss_vector=[], test_loss_vector=[]):
        # Save Weights
        self.save_weights(self.savepath)
        print("\n Model Saved Succesfully \n")

        if loss_vector:
            self.loss_vector = np.concatenate((self.loss_vector, loss_vector))
        if test_loss_vector:
            self.test_loss_vector = np.concatenate((self.test_loss_vector, test_loss_vector))
        data = {"trainingLoss": self.loss_vector, "testLoss": self.test_loss_vector}
        pickle.dump(data, open(self.savepath + "trainingHistory.pickle", "wb"))

        # Make and save a plot of the training history
        fig = plt.figure(figsize=(15, 15))
        ax = gF.addAxis(fig, 1, 2)
        ax[0].plot(self.loss_vector, "k-")
        ax[0].plot(self.test_loss_vector, "b-")

        ax[1].plot(np.log10(self.loss_vector), "k-")
        ax[1].plot(np.log10(self.test_loss_vector), "b-")
        gF.formatPlots(fig, ax[0], None, "epoch", "Loss", "Traning Loss")
        gF.formatPlots(fig, ax[1], None, "epoch", "Log10(Loss)", "Log Loss")

        plt.savefig(self.savepath + "/png_images/trainingHistory.png")
        plt.savefig(self.savepath + "/pdf_images/trainingHistory.pdf")
        plt.close()

    def customLoad(self, verbose=False):
        # If a checkpoint file exists then load the checkpoint weights to architecture
        if verbose:
            print("Checking for model checkpoint at: " + self.savepath)

        if os.path.exists(self.savepath + "checkpoint"):
            self.load_weights(self.savepath).expect_partial()
        else:
            print("\n No Model Checkpoint Found")

        if verbose:
            print("\n Model Checkpoint Loaded \n")

        # Load the previous training loss vector if it exists
        if os.path.exists(self.savepath + "trainingHistory.pickle"):
            with open(self.savepath + "trainingHistory.pickle", "rb") as handle:
                trackHistory = pickle.load(handle)
                self.loss_vector = trackHistory["trainingLoss"]
                self.test_loss_vector = trackHistory["testLoss"]

    def visualizeTrainingCheckpoint(self, epoch_str):
        # It is expected that this function is overloaded by child class
        return

    def get_trainable_variables(self):
        return self.trainable_variables

    def get_variable_names(self):
        trianable_variables = self.get_trainable_variables()
        variable_names = [v.name for v in trianable_variables]
        return variable_names

    def get_variable_by_name(self, name):
        # Note that variables have str like ':0' at the end depending on the operation number
        if not ":" in name:
            name = name + ":0"

        return [v for v in self.trainable_variables if v.name == name]
