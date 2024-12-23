from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

def predict_image(img_path, model_path="src/fruit_classifier.keras", img_size=(100, 100)):
    """
    Realiza a predição para uma imagem fornecida.
    :param img_path: Caminho para a imagem.
    :param model_path: Caminho para o modelo salvo.
    :param img_size: Dimensão para redimensionar a imagem.
    :return: Classe prevista e probabilidades.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Arquivo do modelo {model_path} não encontrado.")
    if not os.path.exists(img_path):
        raise ValueError(f"Arquivo de imagem {img_path} não encontrado.")
    
    # Carregar o modelo
    model = load_model(model_path)
    
    # Carregar e pré-processar a imagem
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img / 255.0, axis=0)  # Normalização
    
    # Realizar a predição
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return predicted_class, predictions

if __name__ == "__main__":
    # Solicitar ao usuário o caminho da imagem
    img_path = input("Digite o caminho da imagem que deseja analisar: ").strip()
    
    # Realizar a predição
    try:
        predicted_class, predictions = predict_image(img_path)
        
        # Mapeamento das classes
        class_names = ["maçã", "banana", "laranja"]
        print(f"Classe prevista: {class_names[predicted_class]}")
        print("Probabilidades:")
        for idx, prob in enumerate(predictions[0]):
            print(f"{class_names[idx]}: {prob*100:.2f}%")
    except ValueError as e:
        print(e)
