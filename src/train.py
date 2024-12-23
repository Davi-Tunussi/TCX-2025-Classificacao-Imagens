from preprocess import load_images
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Carregar os dados
X_train, X_test, y_train, y_test = load_images('data_balanced/train')

# Criar o modelo
model = create_model()

# Configurar Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Configurar Early Stopping
early_stop = EarlyStopping(monitor="val_loss", patience=5)

# Treinar o modelo
model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop]
)

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {accuracy * 100:.2f}%")

# Salvar o modelo
model.save("fruit_classifier.keras")
