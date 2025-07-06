import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import matplotlib.pyplot as plt
import json
import os

class LeafDiseaseDetectionResNet:
    def __init__(self, dataset_path, img_height=224, img_width=224, batch_size=32, epochs=50):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = 0
        self.train_generator = None
        self.val_generator = None
        self.model = None

    def prepare_data(self):
        """Prepares training and validation datasets with preprocessing."""
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        self.train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        self.val_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True
        )

        self.num_classes = len(self.train_generator.class_indices)

        # Save class indices
        with open("class_indices.json", "w") as f:
            json.dump(self.train_generator.class_indices, f)

    def build_model(self):
        """Builds the model using ResNet50."""
        base_model = ResNet50(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # Freeze base

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.summary()

    def train(self):
        """Trains the model."""
        model_checkpoint = ModelCheckpoint('best_leaf_disease_model_resnet.keras', save_best_only=True, monitor='val_loss')
        csv_logger = CSVLogger('training_log.csv', append=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // self.batch_size,
            callbacks=[model_checkpoint, csv_logger, early_stop]
        )

        self.plot_history(history)

    def plot_history(self, history):
        """Plots training and validation accuracy and loss."""
        plt.figure(figsize=(12, 6))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate_best_model(self):
        """Loads and evaluates the best saved model."""
        best_model = tf.keras.models.load_model('best_leaf_disease_model_resnet.keras')
        val_loss, val_acc = best_model.evaluate(self.val_generator)
        print(f"Best Model Validation Accuracy: {val_acc * 100:.2f}%")

# ======= Run the Program =======
if __name__ == "__main__":
    dataset_path = "C:/Users/Gayatri C B/Desktop/LF_disease/train_val"  # <-- Your dataset path
    detector = LeafDiseaseDetectionResNet(dataset_path)
    detector.prepare_data()
    detector.build_model()
    detector.train()
    detector.evaluate_best_model()
