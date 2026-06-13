from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def normalizar_modulacion(modulacion: str) -> str:
    valor = modulacion.strip().upper().replace("_", "-")
    alias = {
        "QAM": "M-QAM",
        "MQAM": "M-QAM",
        "M-QAM": "M-QAM",
        "FSK": "M-FSK",
        "MFSK": "M-FSK",
        "M-FSK": "M-FSK",
    }
    if valor not in alias:
        raise ValueError("La modulación debe ser M-QAM o M-FSK")
    return alias[valor]


def normalizar_etiquetado(tipo_etiquetado: str) -> str:
    valor = tipo_etiquetado.strip().lower()
    if valor in ("gray", "gris"):
        return "gray"
    if valor in ("binario", "binary"):
        return "binario"
    raise ValueError("El etiquetado debe ser 'gray' o 'binario'")


def validar_configuracion_modulacion(modulacion: str, M: int, tipo_etiquetado: str) -> Tuple[str, int, str]:
    modulacion = normalizar_modulacion(modulacion)
    tipo_etiquetado = normalizar_etiquetado(tipo_etiquetado)

    if M not in (2, 4, 8, 16):
        raise ValueError("M debe ser 2, 4, 8 o 16")
    if modulacion == "M-QAM":
        raiz = int(math.sqrt(M))
        if raiz * raiz != M:
            raise ValueError("La QAM implementada requiere M cuadrado perfecto: 4 o 16")

    return modulacion, int(M), tipo_etiquetado


def entero_a_bits(numero: int, cantidad_bits: int) -> str:
    return format(numero, f"0{cantidad_bits}b")


def entero_a_gray(numero: int) -> int:
    return numero ^ (numero >> 1)


def etiqueta_indice(indice: int, bits_por_simbolo: int, tipo_etiquetado: str) -> str:
    if tipo_etiquetado == "gray":
        return entero_a_bits(entero_a_gray(indice), bits_por_simbolo)
    return entero_a_bits(indice, bits_por_simbolo)


def generar_constelacion_fsk(M: int, tipo_etiquetado: str) -> Dict[str, List[float]]:
    bits_por_simbolo = int(math.log2(M))
    constelacion = {}
    for indice in range(M):
        etiqueta = etiqueta_indice(indice, bits_por_simbolo, tipo_etiquetado)
        vector = [0.0] * M
        vector[indice] = 1.0
        constelacion[etiqueta] = vector
    return constelacion


def generar_constelacion_qam(M: int, tipo_etiquetado: str) -> Dict[str, List[float]]:
    raiz = int(math.sqrt(M))
    if raiz * raiz != M:
        raise ValueError("La QAM implementada requiere M cuadrado perfecto")

    bits_por_eje = int(math.log2(raiz))
    xs = [i - (raiz - 1) / 2 for i in range(raiz)]
    ys = [(raiz - 1) / 2 - j for j in range(raiz)]

    constelacion = {}
    for fila, y in enumerate(ys):
        for columna, x in enumerate(xs):
            if tipo_etiquetado == "gray":
                bits_fila = entero_a_bits(entero_a_gray(fila), bits_por_eje)
                bits_columna = entero_a_bits(entero_a_gray(columna), bits_por_eje)
            else:
                bits_fila = entero_a_bits(fila, bits_por_eje)
                bits_columna = entero_a_bits(columna, bits_por_eje)
            constelacion[bits_fila + bits_columna] = [float(x), float(y)]
    return constelacion


def generar_constelacion(modulacion: str, M: int, tipo_etiquetado: str) -> Dict[str, List[float]]:
    modulacion, M, tipo_etiquetado = validar_configuracion_modulacion(modulacion, M, tipo_etiquetado)
    if modulacion == "M-FSK":
        return generar_constelacion_fsk(M, tipo_etiquetado)
    return generar_constelacion_qam(M, tipo_etiquetado)


def agrupar_bits(bits: List[int], bits_por_simbolo: int) -> Tuple[List[str], int]:
    bits_str = [str(int(bit)) for bit in bits]
    resto = len(bits_str) % bits_por_simbolo
    relleno = 0 if resto == 0 else bits_por_simbolo - resto
    if relleno:
        bits_str.extend("0" for _ in range(relleno))

    grupos = ["".join(bits_str[i:i + bits_por_simbolo]) for i in range(0, len(bits_str), bits_por_simbolo)]
    return grupos, relleno


def modular_bits(bits: List[int], modulacion: str, M: int, tipo_etiquetado: str):
    modulacion, M, tipo_etiquetado = validar_configuracion_modulacion(modulacion, M, tipo_etiquetado)
    bits_por_simbolo = int(math.log2(M))
    constelacion = generar_constelacion(modulacion, M, tipo_etiquetado)
    grupos, relleno = agrupar_bits(bits, bits_por_simbolo)
    simbolos = [constelacion[grupo] for grupo in grupos]
    return simbolos, constelacion, grupos, relleno, bits_por_simbolo


def distancia_cuadratica(simbolo_1: List[float], simbolo_2: List[float]) -> float:
    if len(simbolo_1) != len(simbolo_2):
        raise ValueError("Los símbolos deben tener la misma dimensión")
    return sum((a - b) ** 2 for a, b in zip(simbolo_1, simbolo_2))


def demodular_simbolos(simbolos_rx: List[List[float]], modulacion: str, M: int, tipo_etiquetado: str, bits_relleno: int = 0):
    constelacion = generar_constelacion(modulacion, M, tipo_etiquetado)
    bits_rx = []
    simbolos_demodulados = []

    for simbolo_rx in simbolos_rx:
        etiqueta_detectada, simbolo_detectado = min(
            constelacion.items(),
            key=lambda item: distancia_cuadratica(simbolo_rx, item[1]),
        )
        simbolos_demodulados.append(simbolo_detectado)
        bits_rx.extend(int(bit) for bit in etiqueta_detectada)

    if bits_relleno > 0:
        bits_rx = bits_rx[:-bits_relleno]

    return bits_rx, simbolos_demodulados, constelacion


def calcular_energias(simbolos: List[List[float]], bits_por_simbolo: int) -> Tuple[float, float]:
    if not simbolos:
        return 0.0, 0.0
    energias = [sum(coordenada ** 2 for coordenada in simbolo) for simbolo in simbolos]
    Es = sum(energias) / len(energias)
    return Es, Es / bits_por_simbolo


def calcular_energias_constelacion(constelacion: Dict[str, List[float]], bits_por_simbolo: int) -> Tuple[float, float]:
    return calcular_energias(list(constelacion.values()), bits_por_simbolo)


def graficar_constelacion(constelacion: Dict[str, List[float]], modulacion: str, M: int, tipo_etiquetado: str, archivo_figura: str) -> None:
    Path(archivo_figura).parent.mkdir(parents=True, exist_ok=True)
    modulacion = normalizar_modulacion(modulacion)

    fig, ax = plt.subplots()
    if modulacion == "M-QAM":
        xs, ys = [], []
        for etiqueta, punto in constelacion.items():
            x, y = punto
            xs.append(x)
            ys.append(y)
            ax.scatter(x, y)
            ax.text(x + 0.05, y + 0.05, etiqueta)

        for valores, dibujar in ((sorted(set(xs)), ax.axvline), (sorted(set(ys)), ax.axhline)):
            for i in range(len(valores) - 1):
                dibujar((valores[i] + valores[i + 1]) / 2, linestyle="--")

        ax.set_xlabel("Componente en fase")
        ax.set_ylabel("Componente en cuadratura")
        ax.set_aspect("equal", adjustable="box")
    else:
        for etiqueta, vector in constelacion.items():
            indice = vector.index(1.0) + 1
            ax.scatter(indice, 1)
            ax.text(indice + 0.05, 1.03, etiqueta)
            ax.vlines(indice, 0, 1, linestyle="--")
        ax.set_xlabel("Índice de función ortonormal / frecuencia")
        ax.set_ylabel("Coordenada activa")
        ax.set_xticks(range(1, M + 1))
        ax.set_ylim(0, 1.3)

    ax.set_title(f"Constelación transmitida {M}-{modulacion.split('-')[-1]} - {tipo_etiquetado}")
    ax.grid(True)
    plt.savefig(archivo_figura, bbox_inches="tight")
    plt.close(fig)


def graficar_constelacion_recibida(simbolos_rx: List[List[float]], constelacion: Dict[str, List[float]], modulacion: str, M: int, tipo_etiquetado: str, archivo_figura: str, atenuacion: float, N0: float) -> None:
    Path(archivo_figura).parent.mkdir(parents=True, exist_ok=True)
    modulacion = normalizar_modulacion(modulacion)

    fig, ax = plt.subplots()
    if modulacion == "M-QAM":
        xs_ideales, ys_ideales = [], []
        for etiqueta, punto in constelacion.items():
            x, y = punto
            xs_ideales.append(x)
            ys_ideales.append(y)
            ax.scatter(x, y, marker="o", s=80, edgecolors="black")
            ax.text(x + 0.05, y + 0.05, etiqueta)

        for valores, dibujar in ((sorted(set(xs_ideales)), ax.axvline), (sorted(set(ys_ideales)), ax.axhline)):
            for i in range(len(valores) - 1):
                dibujar((valores[i] + valores[i + 1]) / 2, linestyle="--")

        max_puntos = 8000
        simbolos_a_graficar = simbolos_rx[:max_puntos]
        if simbolos_a_graficar:
            ax.scatter([s[0] for s in simbolos_a_graficar], [s[1] for s in simbolos_a_graficar], marker="x", s=15)
        ax.set_xlabel("Componente en fase")
        ax.set_ylabel("Componente en cuadratura")
        ax.set_aspect("equal", adjustable="box")
    else:
        for etiqueta, vector in constelacion.items():
            indice = vector.index(1.0) + 1
            ax.scatter(indice, 1, marker="o")
            ax.text(indice + 0.05, 1.03, etiqueta)
            ax.vlines(indice, 0, 1, linestyle="--")
        for simbolo_rx in simbolos_rx[:8000]:
            indice_max = max(range(len(simbolo_rx)), key=lambda i: simbolo_rx[i]) + 1
            valor_max = max(simbolo_rx)
            ax.scatter(indice_max, valor_max, marker="x", s=15)
        ax.set_xlabel("Índice de función ortonormal / frecuencia")
        ax.set_ylabel("Salida del correlador")
        ax.set_xticks(range(1, M + 1))

    ax.set_title(f"Constelación recibida {M}-{modulacion.split('-')[-1]} - atenuación={atenuacion:.4f} - N0={N0}")
    ax.grid(True)
    plt.savefig(archivo_figura, bbox_inches="tight")
    plt.close(fig)
