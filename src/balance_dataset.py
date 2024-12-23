import os
import shutil
import random

def balance_dataset(source_dir, target_dir, num_images):
    """
    Copia um número fixo de imagens por classe para um diretório balanceado.
    :param source_dir: Diretório com as imagens originais.
    :param target_dir: Diretório onde as imagens balanceadas serão salvas.
    :param num_images: Número de imagens por classe.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path):
            target_class_path = os.path.join(target_dir, class_name)
            os.makedirs(target_class_path, exist_ok=True)
            
            images = os.listdir(class_path)
            selected_images = random.sample(images, min(num_images, len(images)))
            
            for img in selected_images:
                shutil.copy(os.path.join(class_path, img), target_class_path)
            print(f"Copiado {len(selected_images)} imagens para {target_class_path}.")

if __name__ == "__main__":
    balance_dataset("data/train", "data_balanced/train", num_images=479)
    balance_dataset("data/test", "data_balanced/test", num_images=100)
