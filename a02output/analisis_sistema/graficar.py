from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Si querés usar directamente el archivo que subiste:
ARCHIVO_CSV = "a02output/analisis_sistema/resumen_probabilidades_error.csv"

# Si después lo movés a tu proyecto, podés cambiarlo por algo así:
# ARCHIVO_CSV = "a02output/analisis_sistema/resumen_probabilidades_error.csv"

CARPETA_SALIDA = Path("a02output/analisis_sistema/graficos_desde_csv")

# Métrica para los gráficos comparativos generales:
# "Pe" = probabilidad de error de bit
# "Ps" = probabilidad de error de símbolo
METRICA_COMPARATIVA = "Pe"

# Piso para escala logarítmica cuando haya valores 0
PISO_LOG = 1e-7


# ============================================================
# FUNCIONES TEÓRICAS
# ============================================================

def Q(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def limitar_probabilidad(p: float) -> float:
    return max(0.0, min(1.0, p))


def teorica_fsk(M: int, EbN0_dB: float) -> tuple[float, float]:
    """
    Aproximación teórica para M-FSK ortogonal coherente.

    Devuelve:
        Ps_teorica, Pe_teorica
    """

    k = int(math.log2(M))
    gamma_b = 10 ** (EbN0_dB / 10)

    # Aproximación por cota de unión
    Ps = (M - 1) * Q(math.sqrt(k * gamma_b))
    Ps = limitar_probabilidad(Ps)

    if M == 2:
        Pe = Ps
    else:
        Pe = Ps * M / (2 * (M - 1))

    Pe = limitar_probabilidad(Pe)

    return Ps, Pe


def teorica_qam(M: int, EbN0_dB: float) -> tuple[float, float]:
    """
    Aproximación teórica para M-QAM cuadrada con Gray.

    Devuelve:
        Ps_teorica, Pe_teorica
    """

    gamma_b = 10 ** (EbN0_dB / 10)
    k = int(math.log2(M))

    # Caso especial 4-QAM (QPSK), más exacto
    if M == 4:
        Pe = Q(math.sqrt(2 * gamma_b))
        Ps = 1 - (1 - Pe) ** 2
        return limitar_probabilidad(Ps), limitar_probabilidad(Pe)

    # Caso general cuadrado
    raiz_M = math.sqrt(M)

    argumento = math.sqrt((3 * k / (M - 1)) * gamma_b)
    q = Q(argumento)

    Pe = (4 / k) * (1 - 1 / raiz_M) * q
    Ps = 1 - (1 - 2 * (1 - 1 / raiz_M) * q) ** 2

    Pe = limitar_probabilidad(Pe)
    Ps = limitar_probabilidad(Ps)

    return Ps, Pe


def teorica_modulacion(modulacion: str, M: int, EbN0_dB: float) -> tuple[float, float]:
    modulacion = str(modulacion).upper().replace("_", "-")

    if modulacion in ("M-FSK", "FSK", "MFSK"):
        return teorica_fsk(M, EbN0_dB)

    if modulacion in ("M-QAM", "QAM", "MQAM"):
        return teorica_qam(M, EbN0_dB)

    raise ValueError(f"Modulación no soportada: {modulacion}")


# ============================================================
# LECTURA Y PREPARACIÓN DEL CSV
# ============================================================

def leer_csv_resultados(archivo_csv: str) -> pd.DataFrame:
    tabla = pd.read_csv(archivo_csv)

    renombres = {
        "Nombre modulación": "nombre_modulacion",
        "Modulación": "modulacion",
        "M": "M",
        "EbN0_dB": "EbN0_dB",
        "Ps": "Ps_simulada",
        "Pe": "Pe_simulada",
    }

    tabla = tabla.rename(columns=renombres)

    columnas_necesarias = [
        "nombre_modulacion",
        "modulacion",
        "M",
        "EbN0_dB",
        "Ps_simulada",
        "Pe_simulada",
    ]

    for columna in columnas_necesarias:
        if columna not in tabla.columns:
            raise ValueError(
                f"Falta la columna '{columna}'. "
                f"Columnas disponibles: {list(tabla.columns)}"
            )

    tabla["M"] = tabla["M"].astype(int)
    tabla["EbN0_dB"] = tabla["EbN0_dB"].astype(float)
    tabla["Ps_simulada"] = tabla["Ps_simulada"].astype(float)
    tabla["Pe_simulada"] = tabla["Pe_simulada"].astype(float)

    return tabla.sort_values(by=["modulacion", "M", "EbN0_dB"]).reset_index(drop=True)


def agregar_columnas_teoricas(tabla: pd.DataFrame) -> pd.DataFrame:
    tabla = tabla.copy()

    lista_ps = []
    lista_pe = []

    for _, fila in tabla.iterrows():
        Ps_t, Pe_t = teorica_modulacion(
            fila["modulacion"],
            int(fila["M"]),
            float(fila["EbN0_dB"])
        )
        lista_ps.append(Ps_t)
        lista_pe.append(Pe_t)

    tabla["Ps_teorica"] = lista_ps
    tabla["Pe_teorica"] = lista_pe

    return tabla


def preparar_para_log(serie: pd.Series) -> pd.Series:
    """
    En escala log no se pueden graficar ceros.
    Entonces se reemplazan por un piso pequeño.
    """
    return serie.astype(float).clip(lower=PISO_LOG)


def guardar_csv_enriquecido(tabla: pd.DataFrame) -> None:
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    archivo_salida = CARPETA_SALIDA / "resultados_con_teoricas.csv"

    columnas = [
        "nombre_modulacion",
        "modulacion",
        "M",
        "EbN0_dB",
        "Ps_simulada",
        "Pe_simulada",
        "Ps_teorica",
        "Pe_teorica",
    ]

    tabla[columnas].to_csv(archivo_salida, index=False)

    print(f"CSV enriquecido generado: {archivo_salida}")


# ============================================================
# GRÁFICO GENÉRICO
# ============================================================

def graficar_simulada_y_teorica(
    tabla: pd.DataFrame,
    modulaciones: list[str],
    valores_M: list[int] | None,
    metrica: str,
    titulo: str,
    archivo_salida: Path
) -> None:
    """
    metrica:
        "Ps" -> error de símbolo
        "Pe" -> error de bit
    """

    if metrica not in ("Ps", "Pe"):
        raise ValueError("La métrica debe ser 'Ps' o 'Pe'")

    col_sim = f"{metrica}_simulada"
    col_teo = f"{metrica}_teorica"

    ylabel = (
        "Probabilidad de error de símbolo"
        if metrica == "Ps"
        else "Probabilidad de error de bit"
    )

    datos = tabla[tabla["modulacion"].isin(modulaciones)].copy()

    if valores_M is not None:
        datos = datos[datos["M"].isin(valores_M)].copy()

    if datos.empty:
        print(f"No hay datos para generar: {titulo}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    grupos = (
        datos[["modulacion", "M", "nombre_modulacion"]]
        .drop_duplicates()
        .sort_values(by=["modulacion", "M"])
    )

    for _, grupo in grupos.iterrows():
        modulacion = grupo["modulacion"]
        M = int(grupo["M"])
        nombre = grupo["nombre_modulacion"]

        datos_curva = datos[
            (datos["modulacion"] == modulacion) &
            (datos["M"] == M)
        ].sort_values(by="EbN0_dB")

        linea_sim = ax.semilogy(
            datos_curva["EbN0_dB"],
            preparar_para_log(datos_curva[col_sim]),
            marker="o",
            linestyle="-",
            label=f"{nombre} simulada"
        )[0]

        color = linea_sim.get_color()

        ax.semilogy(
            datos_curva["EbN0_dB"],
            preparar_para_log(datos_curva[col_teo]),
            linestyle="--",
            color=color,
            label=f"{nombre} teórica"
        )

    ax.set_xlabel("Eb/N0 [dB]")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()
    ax.set_ylim(bottom=PISO_LOG)

    archivo_salida.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(archivo_salida, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"Gráfico generado: {archivo_salida}")


# ============================================================
# GRÁFICOS PEDIDOS
# ============================================================

def graficar_fsk_con_teoricas(tabla: pd.DataFrame) -> None:
    """
    Todas las FSK juntas con sus curvas simuladas y teóricas.
    """

    graficar_simulada_y_teorica(
        tabla=tabla,
        modulaciones=["M-FSK"],
        valores_M=[2, 4, 8, 16],
        metrica="Ps",
        titulo="FSK: Probabilidad de error de símbolo vs Eb/N0",
        archivo_salida=CARPETA_SALIDA / "FSK_todas_Ps_simulada_teorica.png"
    )

    graficar_simulada_y_teorica(
        tabla=tabla,
        modulaciones=["M-FSK"],
        valores_M=[2, 4, 8, 16],
        metrica="Pe",
        titulo="FSK: Probabilidad de error de bit vs Eb/N0",
        archivo_salida=CARPETA_SALIDA / "FSK_todas_Pe_simulada_teorica.png"
    )


def graficar_todas_las_modulaciones(tabla: pd.DataFrame, metrica: str) -> None:
    """
    Todas las FSK + todas las QAM en un solo gráfico,
    para una sola métrica elegida.
    """

    graficar_simulada_y_teorica(
        tabla=tabla,
        modulaciones=["M-FSK", "M-QAM"],
        valores_M=None,
        metrica=metrica,
        titulo=f"Todas las modulaciones: {metrica} vs Eb/N0",
        archivo_salida=CARPETA_SALIDA / f"todas_las_modulaciones_{metrica}.png"
    )


def graficar_comparacion_M16(tabla: pd.DataFrame, metrica: str) -> None:
    """
    16-FSK contra 16-QAM
    """

    graficar_simulada_y_teorica(
        tabla=tabla,
        modulaciones=["M-FSK", "M-QAM"],
        valores_M=[16],
        metrica=metrica,
        titulo=f"Comparación M=16: 16-FSK vs 16-QAM ({metrica} vs Eb/N0)",
        archivo_salida=CARPETA_SALIDA / f"comparacion_M16_FSK_QAM_{metrica}.png"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    tabla = leer_csv_resultados(ARCHIVO_CSV)
    tabla = agregar_columnas_teoricas(tabla)

    guardar_csv_enriquecido(tabla)

    # 1) Todas las FSK con sus curvas teóricas
    graficar_fsk_con_teoricas(tabla)

    # 2) Todas las FSK + todas las QAM
    graficar_todas_las_modulaciones(
        tabla,
        metrica=METRICA_COMPARATIVA
    )

    # 3) 16-FSK vs 16-QAM
    graficar_comparacion_M16(
        tabla,
        metrica=METRICA_COMPARATIVA
    )

    print("\nProceso finalizado.")
    print(f"Resultados guardados en: {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()