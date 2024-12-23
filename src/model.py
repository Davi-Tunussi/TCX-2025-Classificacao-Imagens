from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

def create_model(input_shape=(100, 100, 3), num_classes=3):
    """
    Função que cria e compila o modelo utilizando Transfer Learning com VGG16.
    :param input_shape: Dimensão das imagens de entrada.
    :param num_classes: Número de classes de saída.
    :return: Modelo compilado.
    """
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Congelar camadas base

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
