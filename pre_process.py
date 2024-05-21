import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

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
X = np.array(inputs).reshape(-1, 1)  # Entradas
y = np.array(outputs)  # Saídas

class RBFNet:
    def __init__(self, k=2, lr=0.01, iterations=100, sigma=None):
        self.k = k  # Número de centros
        self.lr = lr  # Taxa de aprendizado
        self.iterations = iterations  # Iterações de treinamento
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, x, center, sigma):
        epsilon = 1e-6  # Valor mínimo para sigma
        sigma = max(sigma, epsilon)  # Garante que sigma não seja zero
        
        return np.exp(-euclidean_distances(x, center)**2 / (2 * sigma**2)).reshape(-1)

    def fit(self, X, y):
        # K-means para encontrar os centros
        kmeans = KMeans(n_clusters=self.k).fit(X)
        self.centers = kmeans.cluster_centers_
        if self.sigma is None:
            self.sigma = np.mean([np.std(x) for x in X])
        
        # Inicializar pesos
        self.weights = np.random.randn(self.k)
        
        # Treinamento
        for iteration in range(self.iterations):
            for i, x in enumerate(X):
                # Saída da RBF para cada centro
                rbf_outputs = self._rbf(np.array([x]), self.centers, self.sigma)
                F = rbf_outputs.T.dot(self.weights)
                
                # Atualização dos pesos - Gradiente Descendente
                error = y[i] - F
                self.weights += self.lr * error * rbf_outputs.ravel()

    def predict(self, X):
        rbf_outputs = self._rbf(X, self.centers, self.sigma)
        return rbf_outputs.dot(self.weights)

# Criar e treinar a rede
y = np.array(outputs).astype(float)  # Converter saídas para tipo float

rbf_net = RBFNet(k=10, lr=0.1, iterations=1000)
rbf_net.fit(X, y)

# Fazer uma previsão
novos_valores = np.array([[100]])  # Exemplo de novo valor para prever
print("Previsão:", rbf_net.predict(novos_valores))
