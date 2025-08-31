import numpy as np

var = np.array([
[1, 1], # maca
[2, 1], # maca
[1, 2], # maca
[2, 3], # laranja
[3, 3], # laranja
[3, 4]  # laranja
])
v = np.array([0, 0, 0, 1, 1, 1])
eta = 0.1
epochs = 10
weight = np.zeros(var.shape[1] + 1) # +1 para bias

def step_function(u):
	return 1 if u >= 0 else 0

for epoch in range(epochs):
	print(f"Época {epoch + 1 }")
	for i in range(len(var)):
		u = np.dot(var[i], weight[1:]) + weight[0]
		y_pred = step_function(u)
		erro = v[i] - y_pred
		weight[1:] += eta * erro * var[i]
		weight[0] += eta * erro
		print(f"Iteração {i+1}, pesos: {weight}")

print("\nPesos finais:", weight)

newData = np.array([
	[1, 3],
	[4, 2],
	[2, 2],
	[3, 1],
	[4, 4]
])
for ponto in newData:
	u = np.dot(ponto, weight[1:]) + weight[0]
	print(ponto, "->", "Laranja" if step_function(u) == 1 else "Maçã")
