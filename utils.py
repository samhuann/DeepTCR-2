import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle


class KerasCrossValidation:
    def __init__(self, X, y, keras_model, save_filepath):
        self.X = X # list of input predictors
        self.y = y # integer inputs
        self.keras_model = keras_model
        self.initial_state = self.keras_model.get_weights()
        self.save_filepath = save_filepath

    def __call__(self, n_splits=10, batch_size=200, min_epochs=20, patience=30, max_epochs=100, training_repeat=1):
        # instantiate stratified k fold
        self.kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

        # set vars to recieve y_pred and which were in test folds
        y_pred = np.empty((len(self.y), self.keras_model.output_shape[1]))
        y_test = np.zeros((len(self.y), n_splits), dtype=bool)
        history = []
        
        # perform cross validation
        for i, (train_idx, test_idx) in enumerate(self.kfold.split(self.y, self.y)):
            # set indicator for which records are in test fold
            y_test[test_idx, i] = True

            # partition out a validation set for model stopping criteria
            train_idx, valid_idx = [train_idx[idx] for idx in next(StratifiedShuffleSplit(n_splits=1, test_size=len(test_idx)).split(self.y[train_idx], self.y[train_idx]))]

            # set keras model to initial state it was recieved in
            self.keras_model.set_weights(self.initial_state)

            # set training data
            train_tfds = tf.data.Dataset.from_tensor_slices((
                tuple(x[train_idx] for x in self.X),
                np.eye(self.keras_model.output_shape[-1], dtype=np.float32)[self.y[train_idx]]
            ))
            train_tfds = train_tfds.shuffle(len(train_idx), reshuffle_each_iteration=True)
            train_tfds = train_tfds.batch(batch_size=batch_size, drop_remainder=True)
            train_tfds = train_tfds.repeat(training_repeat).prefetch(tf.data.AUTOTUNE)

            # set validation data
            valid_tfds = tf.data.Dataset.from_tensor_slices((
                tuple(x[valid_idx] for x in self.X),
                np.eye(self.keras_model.output_shape[-1])[self.y[valid_idx]]
            ))
            valid_tfds = valid_tfds.batch(batch_size=batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

            # set callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='min', restore_best_weights=True, start_from_epoch=min_epochs),
            ]

            # running training
            self.keras_model.fit(
                train_tfds, # training data - used to update calculate model parameter updates
                validation_data=valid_tfds, # used to assess stopping criteria for training
                class_weight=dict(zip(np.arange(self.keras_model.output_shape[-1]), np.ones(self.keras_model.output_shape[-1]))), # balanced training
                callbacks=callbacks, # callback such are early stopping based on validation data performance
                epochs=max_epochs
            )

            # save model
            history.append(self.keras_model.history.history)
            self.keras_model.save_weights('%s/_%d_cell_model.weights.h5' % (self.save_filepath, i))

            # set test data
            test_tfds = tf.data.Dataset.from_tensor_slices((
                tuple(x[test_idx] for x in self.X),
                np.eye(self.keras_model.output_shape[-1])[self.y[test_idx]]
            ))
            test_tfds = test_tfds.batch(batch_size=batch_size, drop_remainder=False)
            y_pred[test_idx] = self.keras_model.predict(test_tfds)

        # dump to pickle
        pickle.dump((y_pred, y_test, history), open('%s/a.pkl' % self.save_filepath, 'wb'))

        return y_pred, y_test, history


def plot_classification_performance(axs, y_true, y_pred, training_history=None, normalize=None, include_values=True, display_labels=None, fontsize=8):
    # compute the confusion matrix
    cm = confusion_matrix(y_true, np.argmax(y_pred, axis=-1), normalize=normalize)
    # plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Blues, ax=axs[0], colorbar=False, include_values=include_values)

    # plot the ROCs and display the AUCs
    for i in range(y_pred.shape[1]):
        fpr, tpr, _ = roc_curve(y_true == i, y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        axs[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f})')
    axs[1].legend(fontsize=fontsize)

    # plot the training losses if provided
    if training_history is not None:
        if 'loss' in training_history: 
            axs[2].plot(training_history['loss'], label='train')
        if 'val_loss' in training_history: 
            axs[2].plot(training_history['val_loss'], label='valid')
        axs[2].legend()
