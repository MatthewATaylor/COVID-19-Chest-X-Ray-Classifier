import tensorflow as tf


class TrainingNotifier(tf.keras.callbacks.Callback):
    """
    Callback class for sending Discord notifications at the end
    of each epoch
    """
    def __init__(self, notifier):
        """
        Initializes TrainingNotifier with Notifier object
        """
        super().__init__()
        self.notifier = notifier

    def on_epoch_end(self, epoch, logs=None):
        self.notifier.send(
            f"Finished training epoch {epoch} with "
            f"training accuracy: {logs['accuracy']} and "
            f"validation accuracy: {logs['val_accuracy']}",
            print_message=False
        )
