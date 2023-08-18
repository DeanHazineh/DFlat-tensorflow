import tensorflow as tf
import time
import numpy as np
import math


class LinearRampCosineDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, ramp_up_epochs, total_epochs, min_learning_rate=0, warmup=False):
        super(LinearRampCosineDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.ramp_up_epochs = ramp_up_epochs
        self.total_epochs = total_epochs
        self.min_learning_rate = min_learning_rate
        self.warmup = warmup

    def __call__(self, step):
        if self.warmup:
            # Linear warmup
            warmup_lr = self.initial_learning_rate * tf.math.minimum(1.0, (step + 1) / self.ramp_up_epochs)
            completed_fraction = (step - self.ramp_up_epochs) / (self.total_epochs - self.ramp_up_epochs)
        else:
            completed_fraction = step / self.total_epochs

        # Cosine decay
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * completed_fraction))
        decayed_learning_rate = self.initial_learning_rate * cosine_decay
        decayed_learning_rate = tf.maximum(decayed_learning_rate, self.min_learning_rate)

        if self.warmup:
            return tf.cond(step < self.ramp_up_epochs, lambda: warmup_lr, lambda: decayed_learning_rate)
        else:
            return decayed_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "ramp_up_epochs": self.ramp_up_epochs,
            "total_epochs": self.total_epochs,
            "min_learning_rate": self.min_learning_rate,
            "warmup": self.warmup,
        }


def run_pipeline_optimization(pipeline, optimizer, num_epochs, loss_fn=None):
    """Runs the training for DFlat's custom pipelines.

    Args:
        `pipeline` (pipeline_Object): Computational pipeline to optimize. This helper function is compatible and designed
            for the custom pipeline_Object in common_pipelines.pipelines, included in DFlat.
        `optimizer` (tf.keras.optimizers): Keras optimizer used during training.
        `num_epochs` (int): Number of training epochs.
        `loss_fn` (tf.function or function handle): Loss function to use. The input to the function must be the
            pipeline's __call__ function output only. If none, the pipeline output value will be the loss by default.
        `allow_gpu` (bool, optional): Boolean flag indicating if training should be done on the discovered gpu device.
            Defaults to True. If memory of gpu is found to be insufficient, the trainer will automatically catch and
            switch to cpu.
    """
    if loss_fn is None:

        def loss_fn(pipeline_output):
            return pipeline_output

    train_loop(pipeline, optimizer, loss_fn, num_epochs)

    return


def train_loop(pipeline, optimizer, loss_fn, num_epochs):
    # Create default loss function if none was provided
    lossVec = []
    mini_ckpt = pipeline.saveAtEpochs  # Checkpoint epoch number
    pipeline_loss = pipeline.loss_vector
    start_iter = len(pipeline_loss) if len(pipeline_loss) else 0

    # Call once before starting training and save initial state visualization
    pipeline()
    pipeline.visualizeTrainingCheckpoint(str(start_iter))

    # Run All Training Steps
    for epoch in range(num_epochs):
        start = time.time()
        current_loss = train(pipeline, loss_fn, optimizer)
        end = time.time()

        # Depending on the tensorflow version the following lines might need editing
        try:
            print("Training Log | (Step, time, loss, lr): ", start_iter + epoch, end - start, current_loss, optimizer.learning_rate(epoch).numpy())
            # print("Training Log | (Step, time, loss, lr): ", start_iter + epoch, end - start, current_loss, optimizer.lr.numpy())
        except:
            print("Training Log | (Step, time, loss): ", start_iter + epoch, end - start, current_loss)
        lossVec.append(current_loss)

        # After every N steps, save a figure with useful information
        if mini_ckpt:
            if np.mod(start_iter + epoch + 1, mini_ckpt) == 0:
                print("Log Training at step: " + str(start_iter + epoch + 1))
                pipeline.visualizeTrainingCheckpoint(str(start_iter + epoch + 1))

                print("Save Checkpoint Model:")
                pipeline.customSaveCheckpoint(lossVec)
                lossVec = []

    # Save the model and loss post-training
    pipeline.customSaveCheckpoint(lossVec)

    return


def train(pipeline, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(pipeline())

    gradients = tape.gradient(current_loss, pipeline.trainable_variables)
    gradients = [tf.where(tf.math.is_nan(gradient), tf.zeros_like(gradient), gradient) for gradient in gradients]
    optimizer.apply_gradients(zip(gradients, pipeline.trainable_variables))

    return current_loss.numpy()
