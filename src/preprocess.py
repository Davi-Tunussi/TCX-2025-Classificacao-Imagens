import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_images(data_dir, img_size=(100, 100), classes=["apple", "banana", "orange"]):
    """
    Função que carrega as imagens de um diretório, realiza o pré-processamento
    e divide os dados em conjuntos de treino e teste.
    :param data_dir: Caminho para o diretório com as subpastas de cada classe.
    :param img_size: Dimensão das imagens a serem redimensionadas.
    :param classes: Lista com os nomes das classes.
    :return: Dados divididos em treino e teste.
    """
    X, y = [], []
    for idx, fruit in enumerate(classes):
        fruit_dir = os.path.join(data_dir, fruit)
        for img_name in os.listdir(fruit_dir):
            img_path = os.path.join(fruit_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(idx)
    X = np.array(X, dtype="float32") / 255.0  # Normalização
    y = to_categorical(y, num_classes=len(classes))
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_images(data_dir='data_balanced/train')
    print(f"Conjunto de treino: {X_train.shape}, Conjunto de teste: {X_test.shape}")
