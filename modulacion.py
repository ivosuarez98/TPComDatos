from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Energía de bit objetivo
# ============================================================

EB_OBJETIVO = 1.0


# ============================================================
# Normalización de nombres
# ============================================================

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


def validar_configuracion_modulacion(
    modulacion: str,
    M: int,
    tipo_etiquetado: str
) -> Tuple[str, int, str]:

    modulacion = normalizar_modulacion(modulacion)
    tipo_etiquetado = normalizar_etiquetado(tipo_etiquetado)

    if M not in (2, 4, 8, 16):
        raise ValueError("M debe ser 2, 4, 8 o 16")

    if modulacion == "M-QAM":
        raiz = int(math.sqrt(M))
        if raiz * raiz != M:
            raise ValueError("La QAM implementada requiere M cuadrado perfecto: 4 o 16")

    return modulacion, int(M), tipo_etiquetado


# ============================================================
# Utilidades de bits y etiquetado
# ============================================================

def entero_a_bits(numero: int, cantidad_bits: int) -> str:
    return format(numero, f"0{cantidad_bits}b")


def entero_a_gray(numero: int) -> int:
    return numero ^ (numero >> 1)


def etiqueta_indice(
    indice: int,
    bits_por_simbolo: int,
    tipo_etiquetado: str
) -> str:

    if tipo_etiquetado == "gray":
        return entero_a_bits(entero_a_gray(indice), bits_por_simbolo)

    return entero_a_bits(indice, bits_por_simbolo)


# ============================================================
# Cálculo de energías
# ============================================================

def calcular_energias(
    simbolos: List[List[float]],
    bits_por_simbolo: int
) -> Tuple[float, float]:

    if not simbolos:
        return 0.0, 0.0

    if bits_por_simbolo <= 0:
        raise ValueError("bits_por_simbolo debe ser mayor que cero")

    energias = [
        sum(coordenada ** 2 for coordenada in simbolo)
        for simbolo in simbolos
    ]

    Es = sum(energias) / len(energias)
    Eb = Es / bits_por_simbolo

    return Es, Eb


def calcular_energias_constelacion(
    constelacion: Dict[str, List[float]],
    bits_por_simbolo: int
) -> Tuple[float, float]:

    return calcular_energias(
        list(constelacion.values()),
        bits_por_simbolo
    )


def normalizar_constelacion_a_eb(
    constelacion: Dict[str, List[float]],
    bits_por_simbolo: int,
    Eb_objetivo: float = EB_OBJETIVO
) -> Dict[str, List[float]]:
    """
    Escala la constelación para que la energía media de bit teórica
    sea constante.

    Eb = Es / bits_por_simbolo

    Si Eb_objetivo = 1:
        M = 2   -> Es = 1
        M = 4   -> Es = 2
        M = 8   -> Es = 3
        M = 16  -> Es = 4
    """

    Es_actual, Eb_actual = calcular_energias_constelacion(
        constelacion,
        bits_por_simbolo
    )

    if Eb_actual <= 0:
        raise ValueError("No se puede normalizar la constelación: Eb_actual es cero.")

    factor = math.sqrt(Eb_objetivo / Eb_actual)

    constelacion_normalizada = {}

    for etiqueta, punto in constelacion.items():
        constelacion_normalizada[etiqueta] = [
            factor * coordenada
            for coordenada in punto
        ]

    return constelacion_normalizada


# ============================================================
# Generación de constelaciones
# ============================================================

def generar_constelacion_fsk(
    M: int,
    tipo_etiquetado: str
) -> Dict[str, List[float]]:

    bits_por_simbolo = int(math.log2(M))
    constelacion = {}

    for indice in range(M):
        etiqueta = etiqueta_indice(
            indice,
            bits_por_simbolo,
            tipo_etiquetado
        )

        vector = [0.0] * M
        vector[indice] = 1.0

        constelacion[etiqueta] = vector

    return constelacion


def generar_constelacion_qam(
    M: int,
    tipo_etiquetado: str
) -> Dict[str, List[float]]:

    raiz = int(math.sqrt(M))

    if raiz * raiz != M:
        raise ValueError("La QAM implementada requiere M cuadrado perfecto")

    bits_por_eje = int(math.log2(raiz))

    xs = [
        i - (raiz - 1) / 2
        for i in range(raiz)
    ]

    ys = [
        (raiz - 1) / 2 - j
        for j in range(raiz)
    ]

    constelacion = {}

    for fila, y in enumerate(ys):
        for columna, x in enumerate(xs):

            if tipo_etiquetado == "gray":
                bits_fila = entero_a_bits(
                    entero_a_gray(fila),
                    bits_por_eje
                )

                bits_columna = entero_a_bits(
                    entero_a_gray(columna),
                    bits_por_eje
                )

            else:
                bits_fila = entero_a_bits(
                    fila,
                    bits_por_eje
                )

                bits_columna = entero_a_bits(
                    columna,
                    bits_por_eje
                )

            etiqueta = bits_fila + bits_columna

            constelacion[etiqueta] = [
                float(x),
                float(y)
            ]

    return constelacion


def generar_constelacion(
    modulacion: str,
    M: int,
    tipo_etiquetado: str
) -> Dict[str, List[float]]:
    """
    Genera la constelación y la normaliza para que Eb teórica
    sea constante e igual a EB_OBJETIVO.
    """

    modulacion, M, tipo_etiquetado = validar_configuracion_modulacion(
        modulacion,
        M,
        tipo_etiquetado
    )

    bits_por_simbolo = int(math.log2(M))

    if modulacion == "M-FSK":
        constelacion = generar_constelacion_fsk(
            M,
            tipo_etiquetado
        )
    else:
        constelacion = generar_constelacion_qam(
            M,
            tipo_etiquetado
        )

    constelacion = normalizar_constelacion_a_eb(
        constelacion,
        bits_por_simbolo,
        EB_OBJETIVO
    )

    return constelacion


# ============================================================
# Modulación
# ============================================================

def agrupar_bits(
    bits: List[int],
    bits_por_simbolo: int
) -> Tuple[List[str], int]:

    bits_str = [
        str(int(bit))
        for bit in bits
    ]

    resto = len(bits_str) % bits_por_simbolo

    relleno = 0 if resto == 0 else bits_por_simbolo - resto

    if relleno:
        bits_str.extend(
            "0"
            for _ in range(relleno)
        )

    grupos = [
        "".join(bits_str[i:i + bits_por_simbolo])
        for i in range(0, len(bits_str), bits_por_simbolo)
    ]

    return grupos, relleno


def modular_bits(
    bits: List[int],
    modulacion: str,
    M: int,
    tipo_etiquetado: str
):

    modulacion, M, tipo_etiquetado = validar_configuracion_modulacion(
        modulacion,
        M,
        tipo_etiquetado
    )

    bits_por_simbolo = int(math.log2(M))

    constelacion = generar_constelacion(
        modulacion,
        M,
        tipo_etiquetado
    )

    grupos, relleno = agrupar_bits(
        bits,
        bits_por_simbolo
    )

    simbolos = [
        list(constelacion[grupo])
        for grupo in grupos
    ]

    return simbolos, constelacion, grupos, relleno, bits_por_simbolo


# ============================================================
# Demodulación
# ============================================================

def distancia_cuadratica(
    simbolo_1: List[float],
    simbolo_2: List[float]
) -> float:

    if len(simbolo_1) != len(simbolo_2):
        raise ValueError("Los símbolos deben tener la misma dimensión")

    return sum(
        (a - b) ** 2
        for a, b in zip(simbolo_1, simbolo_2)
    )


def demodular_simbolos(
    simbolos_rx: List[List[float]],
    modulacion: str,
    M: int,
    tipo_etiquetado: str,
    bits_relleno: int = 0
):

    constelacion = generar_constelacion(
        modulacion,
        M,
        tipo_etiquetado
    )

    bits_rx = []
    simbolos_demodulados = []

    for simbolo_rx in simbolos_rx:

        etiqueta_detectada, simbolo_detectado = min(
            constelacion.items(),
            key=lambda item: distancia_cuadratica(
                simbolo_rx,
                item[1]
            )
        )

        simbolos_demodulados.append(
            list(simbolo_detectado)
        )

        bits_rx.extend(
            int(bit)
            for bit in etiqueta_detectada
        )

    if bits_relleno > 0:
        bits_rx = bits_rx[:-bits_relleno]

    return bits_rx, simbolos_demodulados, constelacion


# ============================================================
# Gráficos
# ============================================================

def graficar_constelacion(
    constelacion: Dict[str, List[float]],
    modulacion: str,
    M: int,
    tipo_etiquetado: str,
    archivo_figura: str
) -> None:

    Path(archivo_figura).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    modulacion = normalizar_modulacion(modulacion)

    fig, ax = plt.subplots()

    if modulacion == "M-QAM":

        xs = []
        ys = []

        for etiqueta, punto in constelacion.items():
            x, y = punto

            xs.append(x)
            ys.append(y)

            ax.scatter(x, y)
            ax.text(
                x + 0.05,
                y + 0.05,
                etiqueta
            )

        for valores, dibujar in (
            (sorted(set(xs)), ax.axvline),
            (sorted(set(ys)), ax.axhline)
        ):
            for i in range(len(valores) - 1):
                dibujar(
                    (valores[i] + valores[i + 1]) / 2,
                    linestyle="--"
                )

        ax.set_xlabel("Componente en fase")
        ax.set_ylabel("Componente en cuadratura")
        ax.set_aspect("equal", adjustable="box")

    else:

        valor_maximo = 0.0

        for etiqueta, vector in constelacion.items():
            indice = max(
                range(len(vector)),
                key=lambda i: vector[i]
            ) + 1

            valor_activo = max(vector)
            valor_maximo = max(valor_maximo, valor_activo)

            ax.scatter(indice, valor_activo)
            ax.text(
                indice + 0.05,
                valor_activo + 0.03,
                etiqueta
            )
            ax.vlines(
                indice,
                0,
                valor_activo,
                linestyle="--"
            )

        ax.set_xlabel("Índice de función ortonormal / frecuencia")
        ax.set_ylabel("Coordenada activa")
        ax.set_xticks(range(1, M + 1))
        ax.set_ylim(0, valor_maximo * 1.3)

    ax.set_title(
        f"Constelación transmitida {M}-{modulacion.split('-')[-1]} - {tipo_etiquetado}"
    )

    ax.grid(True)

    plt.savefig(
        archivo_figura,
        bbox_inches="tight"
    )

    plt.close(fig)


def graficar_constelacion_recibida(
    simbolos_rx: List[List[float]],
    constelacion: Dict[str, List[float]],
    modulacion: str,
    M: int,
    tipo_etiquetado: str,
    archivo_figura: str,
    atenuacion: float,
    N0: float
) -> None:

    Path(archivo_figura).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    modulacion = normalizar_modulacion(modulacion)

    fig, ax = plt.subplots()

    if modulacion == "M-QAM":

        # ====================================================
        # 1) Preparar colores por símbolo ideal
        # ====================================================

        etiquetas = list(constelacion.keys())
        colores = plt.cm.tab20(range(len(etiquetas)))

        color_por_etiqueta = {}
        for i, etiqueta in enumerate(etiquetas):
            color_por_etiqueta[etiqueta] = colores[i]

        # ====================================================
        # 2) Graficar constelación ideal
        # ====================================================

        xs_ideales = []
        ys_ideales = []

        for etiqueta, punto in constelacion.items():
            x, y = punto

            xs_ideales.append(x)
            ys_ideales.append(y)

            ax.scatter(
                x,
                y,
                marker="o",
                s=60,
                color=color_por_etiqueta[etiqueta],
                edgecolors="black",
                linewidths=0.8,
                zorder=3
            )

            ax.text(
                x + 0.05,
                y + 0.05,
                etiqueta,
                fontsize=11,
                color="black",
                zorder=4
            )

        # ====================================================
        # 3) Dibujar líneas de decisión
        # ====================================================

        for valores, dibujar in (
            (sorted(set(xs_ideales)), ax.axvline),
            (sorted(set(ys_ideales)), ax.axhline)
        ):
            for i in range(len(valores) - 1):
                dibujar(
                    (valores[i] + valores[i + 1]) / 2,
                    linestyle="--",
                    color="tab:blue",
                    alpha=0.9,
                    zorder=1
                )

        # ====================================================
        # 4) Clasificar cada símbolo recibido según
        #    el símbolo ideal más cercano
        # ====================================================

        max_puntos = 8000
        simbolos_a_graficar = simbolos_rx[:max_puntos]

        puntos_por_etiqueta = {etiqueta: {"x": [], "y": []} for etiqueta in constelacion}

        for simbolo_rx in simbolos_a_graficar:
            etiqueta_detectada, simbolo_detectado = min(
                constelacion.items(),
                key=lambda item: distancia_cuadratica(simbolo_rx, item[1]),
            )

            puntos_por_etiqueta[etiqueta_detectada]["x"].append(simbolo_rx[0])
            puntos_por_etiqueta[etiqueta_detectada]["y"].append(simbolo_rx[1])

        # ====================================================
        # 5) Graficar símbolos recibidos con el color
        #    del símbolo ideal detectado
        # ====================================================

        for etiqueta, datos in puntos_por_etiqueta.items():
            if datos["x"]:
                ax.scatter(
                    datos["x"],
                    datos["y"],
                    marker="x",
                    s=18,
                    color=color_por_etiqueta[etiqueta],
                    alpha=0.55,
                    zorder=2
                )

        ax.set_xlabel("Componente en fase")
        ax.set_ylabel("Componente en cuadratura")
        ax.set_aspect("equal", adjustable="box")

    else:

        valor_maximo = 0.0

        for etiqueta, vector in constelacion.items():
            indice = max(
                range(len(vector)),
                key=lambda i: vector[i]
            ) + 1

            valor_activo = max(vector)
            valor_maximo = max(valor_maximo, valor_activo)

            ax.scatter(
                indice,
                valor_activo,
                marker="o"
            )

            ax.text(
                indice + 0.05,
                valor_activo + 0.03,
                etiqueta
            )

            ax.vlines(
                indice,
                0,
                valor_activo,
                linestyle="--"
            )

        for simbolo_rx in simbolos_rx[:8000]:
            indice_max = max(
                range(len(simbolo_rx)),
                key=lambda i: simbolo_rx[i]
            ) + 1

            valor_max = max(simbolo_rx)

            ax.scatter(
                indice_max,
                valor_max,
                marker="x",
                s=15
            )

        ax.set_xlabel("Índice de función ortonormal / frecuencia")
        ax.set_ylabel("Salida del correlador")
        ax.set_xticks(range(1, M + 1))
        ax.set_ylim(0, valor_maximo * 1.5)

    ax.set_title(
        f"Constelación recibida {M}-{modulacion.split('-')[-1]} - "
        f"atenuación={atenuacion:.4f} - N0={N0}"
    )

    ax.grid(True)

    plt.savefig(
        archivo_figura,
        bbox_inches="tight"
    )

    plt.close(fig)