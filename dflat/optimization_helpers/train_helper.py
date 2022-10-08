import tensorflow as tf
import time
import numpy as np


def run_pipeline_optimization(pipeline, optimizer, num_epochs, loss_fn, allow_gpu=True):
    """Runs the training for DFlat's custom pipelines.

    Args:
        `pipeline` (pipeline_Object): Computational pipeline to optimize. This helper function is compatible and designed
            for the custom pipeline_Object in common_pipelines.pipelines, included in DFlat.
        `optimizer` (tf.keras.optimizers): Keras optimizer used during training.
        `num_epochs` (int): Number of training epochs.
        `loss_fn` (tf.function or function handle): Loss function to use. The input to the function must be the
            pipeline's __call__ function output only.
        `allow_gpu` (bool, optional): Boolean flag indicating if training should be done on the discovered gpu device.
            Defaults to True. If memory of gpu is found to be insufficient, the trainer will automatically catch and
            switch to cpu.
    """
    if not allow_gpu:
        with tf.device("/cpu:0"):
            train_loop(pipeline, optimizer, tf.function(loss_fn), num_epochs)
    else:
        try:
            with tf.device("/gpu:0"):
                train_loop(pipeline, optimizer, tf.function(loss_fn), num_epochs)
        except:
            with tf.device("/cpu:0"):
                train_loop(pipeline, optimizer, tf.function(loss_fn), num_epochs)

    return


def train_loop(pipeline, optimizer, loss_fn, num_epochs):
    lossVec = []
    mini_ckpt = pipeline.saveAtEpochs  # Checkpoint epoch number

    # Call the pipeline once before training
    pipeline()

    # Run All Training Steps
    for epoch in range(num_epochs):
        start = time.time()
        current_loss = train(pipeline, loss_fn, optimizer)
        end = time.time()

        print("Training Log | (Step, time, loss): ", epoch, end - start, current_loss)
        lossVec.append(current_loss)

        # After every N steps, save a figure with useful information
        if mini_ckpt:
            if np.mod(epoch, mini_ckpt) == 0 and epoch > 0:
                print("Log Training at step: " + str(epoch))
                pipeline.visualizeTrainingCheckpoint(str(epoch))
                pipeline.customSaveCheckpoint(lossVec)
                lossVec = []

    # Save the model and loss post-training
    pipeline.customSaveCheckpoint(lossVec)

    return


def train(pipeline, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(pipeline())

    gradients = tape.gradient(current_loss, pipeline.trainable_variables)
    optimizer.apply_gradients(zip(gradients, pipeline.trainable_variables))

    return current_loss.numpy()
