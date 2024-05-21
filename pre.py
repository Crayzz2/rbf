import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Carregar dados JSON
with open('./bases/berlin/berlin_airBnbHotels.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extrair entradas (preço) e saídas (avaliação)
inputs = []
outputs = []

for item in data:
    price = item['price']['value']
    rating = item['rating']
    inputs.append(price)
    outputs.append(rating)

# Converter listas para arrays numpy
X = np.array(inputs).reshape(-1, 1)  # Entradas (preço)
y = np.array(outputs)  # Saídas (avaliação)

class RBFRegression:
    def __init__(self, num_centers=10, lr=0.1, epochs=100, sigma=None):
        self.num_centers = num_centers  # Número de centros
        self.lr = lr  # Taxa de aprendizado
        self.epochs = epochs  # Número de épocas
        self.sigma = sigma  # Sigma
        self.centers = None  # Centros
        self.weights = None  # Pesos

    def _rbf(self, X):
        distances = cdist(X, self.centers)
        return np.exp(- (distances ** 2) / (2 * self.sigma ** 2))

    def fit(self, X, y):
        # Inicializar centros usando KMeans
        kmeans = KMeans(n_clusters=self.num_centers)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        
        # Calcular sigma como a média da distância entre os centros
        if self.sigma is None:
            self.sigma = np.mean(cdist(self.centers, self.centers)) / np.sqrt(2 * self.num_centers)
        
        # Calcular as ativações RBF
        phi = self._rbf(X)
        
        # Adicionar coluna de bias
        phi = np.hstack([np.ones((X.shape[0], 1)), phi])
        
        # Inicializar pesos
        self.weights = np.linalg.lstsq(phi, y, rcond=None)[0]

    def predict(self, X):
        phi = self._rbf(X)
        phi = np.hstack([np.ones((X.shape[0], 1)), phi])
        return np.dot(phi, self.weights)

np.random.seed(0)
# Criar e treinar a rede
rbf_regression = RBFRegression(num_centers=90, lr=0.1, epochs=1000)
rbf_regression.fit(X, y)


# Fazer uma previsão
novo_valor_preco = np.array([[111]])  # Exemplo de novo valor para prever
print("Previsão:", rbf_regression.predict(novo_valor_preco))
