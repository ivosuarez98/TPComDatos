import numpy as np

import pandas as pd
from itertools import product

# Datos del código
n = 24
k = 12

# Matriz generadora G de tamaño 12x24
G = np.array([
    [1,1,1,1,1,0,0,0,1,0,1,0, 1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,0,1,0,0,0,1,0,1, 0,1,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,0,1,0,1,0,0,0,1,1, 0,0,1,0,0,0,0,0,0,0,0,0],
    [1,1,0,1,0,0,0,1,1,1,0,1, 0,0,0,1,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,0,1,1,1,0,1,1, 0,0,0,0,1,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,1,1,0,1,1,1, 0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,1,0,1,1,1,0,1,1,1,0, 0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,1,1,1,0,1,1,1,0,1, 0,0,0,0,0,0,0,1,0,0,0,0],
    [1,0,0,1,1,0,1,1,1,0,0,1, 0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,1,0,1,1,1,0,1,0,1, 0,0,0,0,0,0,0,0,0,1,0,0],
    [1,0,1,0,1,1,1,0,0,0,1,1, 0,0,0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,0,1,1,0,1,0, 0,0,0,0,0,0,0,0,0,0,0,1]
], dtype=int)

# Verificación de dimensiones
print("Dimensión de G:", G.shape)

# Variables para guardar resultados
d_min = n + 1
mensaje_min = None
codigo_min = None

# Generar todos los mensajes posibles de k bits
for bits in product([0, 1], repeat=k):
    u = np.array(bits, dtype=int)

    # Codificación: c = uG mod 2
    c = (u @ G) % 2

    # Peso de Hamming
    peso = np.sum(c)

    # Ignorar la palabra código nula
    if peso != 0 and peso < d_min:
        d_min = peso
        mensaje_min = u
        codigo_min = c

# Cálculo de capacidad de detección y corrección
e = d_min - 1
t = (d_min - 1) // 2

print("\nResultados:")
print("d_min =", d_min)
print("e =", e)
print("t =", t)

print("\nMensaje que genera una palabra de peso mínimo:")
print(mensaje_min)

print("\nPalabra de código de peso mínimo:")
print(codigo_min)

print("\nPeso de Hamming de esa palabra:")
print(np.sum(codigo_min))


k, n = G.shape
r = n - k

P = G[:, :r]
I = np.eye(r, dtype=int)

H = np.concatenate((I, P.T), axis=1) % 2

print("H:")
print(H)

print("\nVerificación G @ H.T mod 2:")
print((G @ H.T) % 2)

print("=== Ejemplo de detección de error con síndrome ===\n")

# Mensaje original de k = 12 bits
w = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
], dtype=int)

print("Mensaje original w:")
print(w)

# Codificación: u = wG mod 2
u = (w @ G) % 2
print("\nu = (w @ G) % 2")
print("\nPalabra codificada u:")
print(u)

# Patrón de error: error en el bit 6
e = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
], dtype=int)

print("\nPatrón de error e:")
print(e)

# Palabra recibida con error: u_tilde = u + e mod 2
u_tilde = (u + e) % 2

print("\nPalabra recibida con error u_tilde:")
print(u_tilde)

# Cálculo del síndrome: s = u_tilde H^T mod 2
print("\ns = (u_tilde @ H.T) % 2")
s = (u_tilde @ H.T) % 2

print("\nSíndrome s:")
print(s)