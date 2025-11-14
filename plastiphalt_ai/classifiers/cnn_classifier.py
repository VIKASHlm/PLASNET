import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import json

class CNNPlasticClassifier:
    def __init__(self, model_path=None, input_shape=(128, 128, 3), num_classes=7):
        self.model_path = model_path
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def train(self, train_dir, model_out, epochs=10, batch_size=32):
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(128,128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        val_gen = datagen.flow_from_directory(
            train_dir,
            target_size=(128,128),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        self.build_model()
        history = self.model.fit(train_gen, validation_data=val_gen, epochs=epochs)

        # Save model
        self.model.save(model_out)

        # Evaluate
        val_preds = self.model.predict(val_gen)
        y_true = val_gen.classes
        y_pred = np.argmax(val_preds, axis=1)

        report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))

        results = {
            "accuracy": float(history.history['val_accuracy'][-1]),
            "report": report
        }

        print(json.dumps(results, indent=2))
        return results

    def load(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        return self

    def predict(self, image_path):
        from tensorflow.keras.preprocessing import image
        img = image.load_img(image_path, target_size=(128,128))
        x = image.img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        preds = self.model.predict(x)
        return np.argmax(preds, axis=1)[0], preds.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or use CNN Plastic Classifier")
    parser.add_argument("--train", type=str, help="Path to dataset directory")
    parser.add_argument("--model", type=str, required=True, help="Path to save/load model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()

    clf = CNNPlasticClassifier(model_path=args.model)

    if args.train:
        clf.train(args.train, args.model, epochs=args.epochs)
    else:
        clf.load()
        print(f"Model loaded from {args.model}")
