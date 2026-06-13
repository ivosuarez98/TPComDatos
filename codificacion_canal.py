from __future__ import annotations

from itertools import combinations, product
from typing import Dict, Tuple

import numpy as np


def validar_binario(vector, nombre="vector") -> np.ndarray:
    vector = np.array(vector, dtype=int)
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError(f"{nombre} solo puede contener valores binarios 0 o 1")
    return vector


def codificar_mensaje(mensaje, k: int, n: int, G) -> np.ndarray:
    mensaje = validar_binario(mensaje, "El mensaje")
    G = validar_binario(G, "La matriz G")

    if mensaje.shape[0] != k:
        raise ValueError(f"El mensaje debe tener {k} bits")
    if G.shape != (k, n):
        raise ValueError(f"La matriz G debe tener tamaño {k}x{n}")

    return np.mod(np.dot(mensaje, G), 2).astype(int)


def codificar_vector_bits(bits, k: int, n: int, G) -> Tuple[np.ndarray, int]:
    bits = validar_binario(bits, "El vector de bits")
    G = validar_binario(G, "La matriz G")

    if G.shape != (k, n):
        raise ValueError(f"La matriz G debe tener tamaño {k}x{n}")

    resto = len(bits) % k
    relleno = 0 if resto == 0 else k - resto
    if relleno:
        bits = np.concatenate((bits, np.zeros(relleno, dtype=int)))

    codigo_total = []
    for i in range(0, len(bits), k):
        codigo_total.extend(codificar_mensaje(bits[i:i + k], k, n, G))

    return np.array(codigo_total, dtype=int), relleno


def obtener_matriz_paridad(k: int, n: int, G) -> np.ndarray:
    G = validar_binario(G, "La matriz G")
    if G.shape != (k, n):
        raise ValueError(f"La matriz G debe tener tamaño {k}x{n}")

    I_k = G[:, :k]
    P = G[:, k:]
    if not np.array_equal(I_k, np.eye(k, dtype=int)):
        raise ValueError("La matriz G no es sistemática de la forma [I_k | P]")

    H = np.concatenate((P.T, np.eye(n - k, dtype=int)), axis=1)
    verificacion = np.mod(np.dot(G, H.T), 2)
    if not np.all(verificacion == 0):
        raise ValueError("La matriz H construida no cumple G . H^T = 0 mod 2")

    return H.astype(int)


def calcular_tabla_sindromes(H) -> Dict[tuple, np.ndarray]:
    H = validar_binario(H, "La matriz H")
    filas_H, n = H.shape
    cantidad_sindromes = 2 ** filas_H
    tabla_sindromes = {}

    for peso in range(n + 1):
        for posiciones in combinations(range(n), peso):
            error = np.zeros(n, dtype=int)
            for pos in posiciones:
                error[pos] = 1

            sindrome = np.mod(np.dot(H, error.T), 2)
            sindrome_clave = tuple(sindrome)
            if sindrome_clave not in tabla_sindromes:
                tabla_sindromes[sindrome_clave] = error.copy()

            if len(tabla_sindromes) == cantidad_sindromes:
                return tabla_sindromes

    return tabla_sindromes


def corregir_palabra(H, S, palabra_recibida):
    H = validar_binario(H, "La matriz H")
    palabra_recibida = validar_binario(palabra_recibida, "La palabra recibida")
    _, n = H.shape

    if palabra_recibida.shape[0] != n:
        raise ValueError(f"La palabra recibida debe tener longitud {n}")

    sindrome = np.mod(np.dot(H, palabra_recibida.T), 2)
    sindrome_clave = tuple(sindrome)

    if np.all(sindrome == 0):
        patron_error = np.zeros(n, dtype=int)
        palabra_corregida = palabra_recibida.copy()
        error_detectado = False
        error_corregido = False
    else:
        error_detectado = True
        if sindrome_clave in S:
            patron_error = np.array(S[sindrome_clave], dtype=int)
            palabra_corregida = np.mod(palabra_recibida + patron_error, 2)
            error_corregido = True
        else:
            patron_error = np.zeros(n, dtype=int)
            palabra_corregida = palabra_recibida.copy()
            error_corregido = False

    return {
        "palabra_recibida": palabra_recibida,
        "sindrome": sindrome,
        "patron_error": patron_error,
        "palabra_corregida": palabra_corregida.astype(int),
        "error_detectado": error_detectado,
        "error_corregido": error_corregido,
    }


def decodificar_vector_bits(H, S, bits_recibidos, relleno: int = 0):
    H = validar_binario(H, "La matriz H")
    bits_recibidos = validar_binario(bits_recibidos, "El vector recibido")
    filas_H, n = H.shape
    k = n - filas_H

    if len(bits_recibidos) % n != 0:
        raise ValueError(f"La longitud del vector recibido debe ser múltiplo de n = {n}")

    bits_decodificados = []
    palabras_corregidas = []
    info_errores = []

    for i in range(0, len(bits_recibidos), n):
        resultado = corregir_palabra(H, S, bits_recibidos[i:i + n])
        palabra_corregida = resultado["palabra_corregida"]
        bits_decodificados.extend(palabra_corregida[:k])
        palabras_corregidas.append(palabra_corregida)
        info_errores.append(resultado)

    bits_decodificados = np.array(bits_decodificados, dtype=int)
    if relleno > 0:
        bits_decodificados = bits_decodificados[:-relleno]

    return bits_decodificados, np.array(palabras_corregidas, dtype=int), info_errores


def calcular_parametros_codigo(k: int, n: int, G):
    G = validar_binario(G, "La matriz G")
    if G.shape != (k, n):
        raise ValueError(f"La matriz G debe tener tamaño {k}x{n}")

    pesos = []
    for mensaje in product([0, 1], repeat=k):
        mensaje = np.array(mensaje, dtype=int)
        if np.all(mensaje == 0):
            continue
        palabra_codigo = np.mod(np.dot(mensaje, G), 2)
        pesos.append(int(np.sum(palabra_codigo)))

    dmin = min(pesos)
    return dmin, dmin - 1, (dmin - 1) // 2
