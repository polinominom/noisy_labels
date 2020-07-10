from __future__ import print_function, division
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from loss import *


# losses that need sigmoid on top of last layer
yes_softmax = ['crossentropy', 'forward', 'est_forward', 'backward',
               'est_backward', 'boot_soft', 'savage']
# unhinged needs bounded models or it diverges
yes_bound = ['unhinged', 'ramp', 'sigmoid']


class KerasModel():
    # custom losses for the CNN
    def make_loss(self, loss, P=None, binary=True):
        if loss == 'crossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if binary:
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            return loss
        elif loss in ['forward', 'backward']:
            return robust(loss, P)
        elif loss == 'unhinged':
            return unhinged
        elif loss == 'sigmoid':
            return sigmoid
        elif loss == 'ramp':
            return ramp
        elif loss == 'savage':
            return savage
        elif loss == 'boot_soft':
            return boot_soft
        else:
            ValueError("Loss unknown.")

    def compile(self, loss, P=None, binary=True):
        if self.optimizer is None or self.model is None:
            ValueError()
        
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        if binary:
            accuracy = tf.keras.metrics.binary_accuracy

        self.model.compile(loss=self.make_loss(loss, P, binary), optimizer=self.optimizer, metrics=metrics)
        self.model.summary()

    def load_model(self, file):
        self.model.load_weights(file)
        print('Loaded model from %s' % file)

    def fit_model(self, model_file, additional_callbacks = []):
        if self.train_loader is None or self.val_loader is None or self.epoch is None or self.batch_size is None:
            print('Parameters are not initialized correctly. ABORTING NETWORK TRANING.')
            return

        callbacks = []
        monitor = 'val_loss'
        mc_callback = ModelCheckpoint(model_file, monitor=monitor,
                                      verbose=1, save_best_only=True)
        callbacks.append(mc_callback)
        for c in additional_callbacks:
            callbacks.append(c)


        if hasattr(self, 'scheduler'):
            callbacks.append(self.scheduler)

        # use data augmentation
        if hasattr(self, 'train_generator'):
            history = \
                self.model.fit(self.train_generator, validation_data = self.validation_generator, 
                                epochs = self.epochs, batch_size=self.batch_size, verbose=1, 
                                callbacks=callbacks)

        else:        
            history = model.fit(self.train_loader, validation_data=self.val_loader, 
                            epochs=self.epochs, batch_size=self.batch_size, 
                            verbose=1, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file)
        return history.history

    def evaluate_model(self, loader):
        score = self.model.evaluate(loader, batch_size=self.num_batch, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score[1]

    def predict_proba(self, loader):
        pred = self.model.predict(loader, batch_size=self.num_batch, verbose=1)
        return pred


class ChexpertModel(KerasModel):

    def __init__(self, model, train_loader, val_loader, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.val_laoder = val_loader
        self.num_batch = train_loader.batch_size
        self.classes = 2
        self.augmentation = True
        self.optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0) # TODO: ADAM ?
        # ASSIGN THE PARAMATER 'self.scheduler'
        self.lr_scheduler()
        self.decay = 0.0001

    def prepare_data_augmentation(self, folder):
        if self.train_loader is None or self.val_loader is None:
            print("Loader/generator does not exist. Aborting load data")
            return

        if self.augmentation:
            print('Data Augmentation')
            # data augmentation
            image_generator = ImageDataGenerator(
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True)


            self.train_generator = image_generator.flow_from_directory(
                '%s/train'%folder,
                target_size=(256, 256, 3),
                batch_size=32,
                class_mode='binary')
            self.validation_generator = image_generator.flow_from_directory(
                '%s/val',
                target_size=(256, 256, 3),
                batch_size=32,
                class_mode='binary')

    def lr_scheduler(self):
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1

        print('LR scheduler')
        self.scheduler = LearningRateScheduler(scheduler)

    def build_model(self, model, loss, P=None, binary=True):
        self.compile(model, loss, P, binary=binary)

class NoiseEstimator():
    def __init__(self, classifier, row_normalize=True, alpha=0.0,
                 filter_outlier=False, cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to label noise."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, loader):
        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))
        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(loader)
        # find a 'perfect example' for each class
        for i in np.arange(c):
            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)
            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):
        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T
