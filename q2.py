import numpy as np
X = np.array([
[5.1, 1.6], # não venenosa
[4.0, 1.3], # não venenosa
[4.8, 1.8], # não venenosa
[1.4, 0.3], # venenosa
[1.9, 0.4], # venenosa
[1.5, 0.2], # venenosa
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0=não venenosa, 1=venenosa
eta = 0.1
epocas = 15
pesos = np.zeros(X.shape[1] + 1)
def step_function(u):
	return 1 if u >= 0 else 0
for epoca in range(epocas):
	print(f"Época {epoca+1}")
	for i in range(len(X)):
		u = np.dot(X[i], pesos[1:]) + pesos[0]
		y_pred = step_function(u)
		erro = y[i] - y_pred
		pesos[1:] += eta * erro * X[i]
		pesos[0] += eta * erro
		print(f" it {i+1}: pesos = {pesos}, erro = {erro}")
print("\nPesos finais:", pesos)
novos = np.array([
[5.0, 1.5],
[1.6, 0.3],
	[4.5, 1.2],
	[1.7, 0.4],
	[3.0, 0.8]
])
print("\nClassificação de novos casos:")
for p in novos:
	u = np.dot(p, pesos[1:]) + pesos[0]
	classe = "Venenosa" if step_function(u) == 1 else "Não venenosa"
	print(f"{p} -> {classe}")