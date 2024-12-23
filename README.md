# Projeto 2: Rede Neural para Classificação de Frutas

## Descrição do Projeto
Este projeto implementa uma rede neural para classificar imagens de frutas (maçã, banana e laranja) usando o **Fruits 360 Dataset**. Ele aborda os seguintes aspectos principais:
- **Pré-processamento das imagens**: Redimensionamento, normalização e divisão em conjuntos de treino e teste.
- **Balanceamento do conjunto de dados**: Garantia de que cada classe tem um número uniforme de imagens.
- **Treinamento do modelo**: Utilizando uma arquitetura de Transfer Learning com a rede pré-treinada **VGG16**.
- **Predição**: Capacidade de prever a classe de uma imagem fornecida pelo usuário.

## Estrutura do Projeto

A estrutura de diretórios do projeto está organizada da seguinte maneira:

```plaintext
PROJETO_2_TCX_Rede_Neural/
├── src/                  # Scripts do projeto
│   ├── __init__.py       # Define src como um pacote Python
│   ├── balance_dataset.py # Script para balancear o dataset
│   ├── model.py          # Script para criar o modelo
│   ├── predict.py        # Script para realizar a predição
│   ├── preprocess.py     # Script para carregar e pré-processar as imagens
│   ├── train.py          # Script para treinar o modelo
│   ├── fruit_classifier.h5
│   └── fruit_classifier.keras
├── data/                 # Dataset original
│   ├── train/            # Conjunto de treino
│   └── test/             # Conjunto de teste
├── data_balanced/        # Dataset balanceado
│   ├── train/            # Conjunto de treino balanceado
│   └── test/             # Conjunto de teste balanceado
├── README.md             # Explicação do projeto
├── requirements.txt      # Dependências do projeto
├── .gitignore            # Arquivos ignorados pelo Git
└── venv/                 # Ambiente virtual Python

## Como Executar

### 1. Preparação do Ambiente
1. Certifique-se de que o **Python 3.8 ou superior** esteja instalado.
2. Crie um ambiente virtual:
   python -m venv venv

*Ative o ambiente virtual:
Windows: venv\Scripts\activate
Linux/Mac: source venv/bin/activate

*Instale as dependências:
pip install -r requirements.txt


### 2. Balancear o Dataset
Execute o script para balancear o dataset:
python src/balance_dataset.py

### 3. Treinamento do Modelo
Treine o modelo usando o seguinte comando:
python src/train.py

### 4. Realizar Predições
Para realizar a predição, execute:
python src/predict.py
O programa solicitará o caminho de uma imagem para análise.

### Exemplo de Saída:
Classe prevista: maçã
Probabilidades:
  maçã: 99.82%
  banana: 0.10%
  laranja: 0.08%

### Requisitos Técnicos
Acurácia mínima esperada no conjunto de teste: 90%.
Ferramentas utilizadas:
Transfer Learning com VGG16.
Data Augmentation para melhorar a generalização.

### Desafios e Melhorias Futuras
Implementar suporte para mais classes do dataset.
Otimizar o desempenho em máquinas sem GPU.
Expandir para outros tipos de modelos de Transfer Learning (ex.: ResNet)

### Dependências
As dependências estão listadas no arquivo requirements.txt. Certifique-se de instalá-las no ambiente virtual.

### Referências
Fruits 360 Dataset no Kaggle
Documentação do TensorFlow: tensorflow.org
