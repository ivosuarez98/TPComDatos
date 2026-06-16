from __future__ import annotations

from pathlib import Path

import pandas as pd

from settings import Settings
from transmisor import Transmisor
from canal import Canal
from receptor import Receptor

import matplotlib.pyplot as plt


EB_OBJETIVO = 1.0


def calcular_N0_desde_EbN0_dB(EbN0_dB: float, Eb_objetivo: float = EB_OBJETIVO) -> float:
    """
    Convierte Eb/N0 en dB a N0 lineal.

    Eb/N0[dB] = 10 log10(Eb/N0)

    Entonces:

    Eb/N0 lineal = 10^(EbN0_dB/10)

    N0 = Eb / (Eb/N0)
    """

    EbN0_lineal = 10 ** (EbN0_dB / 10)
    N0 = eb_objetivo / EbN0_lineal

    return N0


def deshabilitar_graficos(settings: Settings) -> None:
    """
    Desactiva gráficos e informes individuales para acelerar el análisis.
    """

    atributos_a_desactivar = [
        "guardar_graficos",
        "graficar_constelacion_tx",
        "graficar_constelacion_rx",
        "graficar_constelacion_recibida",
        "graficar_curvas",
        "mostrar_graficos",
        "generar_informe",
    ]

    for atributo in atributos_a_desactivar:
        if hasattr(settings, atributo):
            setattr(settings, atributo, False)


def nombre_caso_analisis(settings: Settings, EbN0_dB: float, usar_codigo: bool) -> str:
    modulacion = settings.modulacion.replace("M-", "").replace("-", "").lower()
    codigo = "con_codigo" if usar_codigo else "sin_codigo"

    return f"{modulacion}_M{settings.M}_{settings.tipo_etiquetado}_{codigo}_EbN0_{EbN0_dB}_dB"


def preparar_rutas_analisis(settings: Settings, EbN0_dB: float, usar_codigo: bool) -> str:
    caso = nombre_caso_analisis(settings, EbN0_dB, usar_codigo)

    carpeta = f"a02output/analisis_sistema/{caso}"

    settings.ruta_archivo_salida = f"{carpeta}/mensaje_recibido.txt"
    settings.ruta_informe = f"{carpeta}/informe.txt"
    settings.carpeta_graficos = carpeta

    Path(carpeta).mkdir(parents=True, exist_ok=True)

    return caso


def ejecutar_caso_analisis(settings: Settings, EbN0_dB: float, usar_codigo: bool) -> dict:
    """
    Ejecuta transmisor -> canal -> receptor usando tus clases existentes.
    No grafica ni genera informes individuales.
    """

    caso = preparar_rutas_analisis(settings, EbN0_dB, usar_codigo)

    transmisor = Transmisor(settings)
    canal = Canal(settings)
    receptor = Receptor(settings)

    senal_tx = transmisor.transmitir()
    senal_rx = canal.transmitir(senal_tx)
    resultado = receptor.recibir(senal_rx, contexto=senal_tx.contexto)

    print("\n" + "=" * 72)
    print(f"CASO: {caso}")
    print("=" * 72)
    print(resultado.resumen())

    return {
        "nombre_caso": caso,
        "settings": settings,
        "senal_tx": senal_tx,
        "senal_rx": senal_rx,
        "resultado": resultado,
        "EbN0_dB": EbN0_dB,
        "N0": settings.N0,
    }


def extraer_fila_resultado(caso: dict) -> dict:
    settings = caso["settings"]
    senal_tx = caso["senal_tx"]
    senal_rx = caso["senal_rx"]
    resultado = caso["resultado"]
    contexto = senal_tx.contexto

    return {
        "Caso": caso["nombre_caso"],

        "Modulación": contexto.modulacion,
        "M": contexto.M,
        "Etiquetado": contexto.tipo_etiquetado,

        "EbN0_dB": caso["EbN0_dB"],
        "N0": caso["N0"],

        "AWGN": settings.usar_awgn,
        "Atenuación": settings.usar_atenuacion,
        "Atenuación aplicada": round(senal_rx.atenuacion, 6),

        "Codificación canal": contexto.usar_codificacion_canal,

        "Bits fuente": len(contexto.bits_fuente),
        "Bits modulador": len(contexto.bits_modulador),
        "Símbolos TX": len(contexto.simbolos_tx),

        "Es teórica": round(contexto.energia_media_simbolo_teorica, 6),
        "Eb teórica": round(contexto.energia_media_bit_teorica, 6),
        "Es estimada": round(contexto.energia_media_simbolo, 6),
        "Eb estimada": round(contexto.energia_media_bit, 6),

        "Errores símbolo": resultado.errores_simbolo,
        "Pe símbolo": round(resultado.probabilidad_error_simbolo, 8),

        "Errores bit": resultado.errores_bit,
        "Pe bit": round(resultado.probabilidad_error_bit, 8),

        "Decodificación": "OK" if resultado.decodificacion_exitosa else "Con errores",
    }


def crear_settings_base(
    modulacion: str,
    M: int,
    tipo_etiquetado: str,
    usar_codigo: bool,
    N0: float
) -> Settings:
    """
    Crea los settings para cada iteración.
    """

    settings = Settings(
        modulacion=modulacion,
        M=M,
        tipo_etiquetado=tipo_etiquetado,

        usar_codificacion_canal=usar_codigo,

        usar_awgn=True,
        usar_atenuacion=False,
        usar_respuesta_impulsiva=False,

        N0=N0,
    )

    deshabilitar_graficos(settings)

    return settings


def analizar_configuracion(
    modulacion: str,
    M: int,
    tipo_etiquetado: str,
    usar_codigo: bool
) -> pd.DataFrame:
    """
    Barre Eb/N0 de 0 a 10 dB para una configuración dada.
    """

    filas = []

    for EbN0_dB in range(0, 11):

        N0 = calcular_N0_desde_EbN0_dB(
            EbN0_dB,
            eb_objetivo=EB_OBJETIVO
        )

        settings = crear_settings_base(
            modulacion=modulacion,
            M=M,
            tipo_etiquetado=tipo_etiquetado,
            usar_codigo=usar_codigo,
            N0=N0
        )

        print("\n" + "#" * 72)
        print(f"Ejecutando {modulacion}, M={M}, Eb/N0={EbN0_dB} dB, N0={N0:.6f}")
        print("#" * 72)

        caso = ejecutar_caso_analisis(
            settings=settings,
            EbN0_dB=EbN0_dB,
            usar_codigo=usar_codigo
        )

        fila = extraer_fila_resultado(caso)
        filas.append(fila)

    return pd.DataFrame(filas)


def guardar_tabla_analisis(tabla: pd.DataFrame, nombre_archivo: str) -> None:
    carpeta = Path("a02output/analisis_sistema")
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_csv = carpeta / f"{nombre_archivo}.csv"
    ruta_txt = carpeta / f"{nombre_archivo}.txt"

    tabla.to_csv(ruta_csv, index=False)

    with open(ruta_txt, "w", encoding="utf-8") as archivo:
        archivo.write("ANÁLISIS DEL SISTEMA\n")
        archivo.write("=" * 72 + "\n\n")
        archivo.write(tabla.to_string(index=False))
        archivo.write("\n")

    print("\nArchivos generados:")
    print(ruta_csv)
    print(ruta_txt)

def calcular_N0_desde_EbN0_dB(EbN0_dB: float, eb_objetivo: float = EB_OBJETIVO) -> float:
    EbN0_lineal = 10 ** (EbN0_dB / 10)
    N0 = eb_objetivo / EbN0_lineal
    return N0

def graficar_probabilidades_error_desde_csv(
    archivo_csv: str,
    archivo_salida: str | None = None
) -> None:
    """
    Lee un CSV de resultados del análisis y grafica:
    - Probabilidad de error de símbolo
    - Probabilidad de error de bit

    Usa como eje x Eb/N0 en dB.
    """

    tabla = pd.read_csv(archivo_csv)

    columnas_necesarias = [
        "EbN0_dB",
        "Pe símbolo",
        "Pe bit"
    ]

    for columna in columnas_necesarias:
        if columna not in tabla.columns:
            raise ValueError(
                f"No se encontró la columna '{columna}' en el CSV. "
                f"Columnas disponibles: {list(tabla.columns)}"
            )

    tabla = tabla.sort_values(by="EbN0_dB")

    x = tabla["EbN0_dB"]
    pe_simbolo = tabla["Pe símbolo"]
    pe_bit = tabla["Pe bit"]

    if archivo_salida is None:
        ruta_csv = Path(archivo_csv)
        archivo_salida = ruta_csv.with_name(
            ruta_csv.stem + "_curvas_error.png"
        )

    Path(archivo_salida).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure()

    plt.semilogy(
        x,
        pe_simbolo.replace(0, float("nan")),
        marker="o",
        label="Probabilidad de error de símbolo"
    )

    plt.semilogy(
        x,
        pe_bit.replace(0, float("nan")),
        marker="s",
        label="Probabilidad de error de bit"
    )

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Probabilidad de error")
    plt.title("Curvas de probabilidad de error")
    plt.grid(True, which="both")
    plt.legend()

    plt.savefig(
        archivo_salida,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfico generado: {archivo_salida}")


def graficar_comparativo_fsk_desde_csv(
    archivo_csv: str,
    archivo_salida_simbolo: str,
    archivo_salida_bit: str
) -> None:
    """
    Lee un CSV general con todos los FSK y grafica:
    - Pe símbolo vs Eb/N0 para M = 2, 4, 8, 16
    - Pe bit vs Eb/N0 para M = 2, 4, 8, 16
    """

    tabla = pd.read_csv(archivo_csv)

    columnas_necesarias = [
        "M",
        "EbN0_dB",
        "Pe símbolo",
        "Pe bit"
    ]

    for columna in columnas_necesarias:
        if columna not in tabla.columns:
            raise ValueError(
                f"No se encontró la columna '{columna}' en el CSV. "
                f"Columnas disponibles: {list(tabla.columns)}"
            )

    Path(archivo_salida_simbolo).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Gráfico de probabilidad de error de símbolo
    # ========================================================

    plt.figure()

    for M in sorted(tabla["M"].unique()):
        datos_M = tabla[tabla["M"] == M].sort_values(by="EbN0_dB")

        plt.semilogy(
            datos_M["EbN0_dB"],
            datos_M["Pe símbolo"].replace(0, float("nan")),
            marker="o",
            label=f"{M}-FSK"
        )

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Probabilidad de error de símbolo")
    plt.title("Probabilidad de error de símbolo - M-FSK")
    plt.grid(True, which="both")
    plt.legend()

    plt.savefig(
        archivo_salida_simbolo,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # Gráfico de probabilidad de error de bit
    # ========================================================

    plt.figure()

    for M in sorted(tabla["M"].unique()):
        datos_M = tabla[tabla["M"] == M].sort_values(by="EbN0_dB")

        plt.semilogy(
            datos_M["EbN0_dB"],
            datos_M["Pe bit"].replace(0, float("nan")),
            marker="s",
            label=f"{M}-FSK"
        )

    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("Probabilidad de error de bit")
    plt.title("Probabilidad de error de bit - M-FSK")
    plt.grid(True, which="both")
    plt.legend()

    plt.savefig(
        archivo_salida_bit,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Gráfico generado: {archivo_salida_simbolo}")
    print(f"Gráfico generado: {archivo_salida_bit}")

def guardar_csv_resumen_probabilidades(
    tabla_general: pd.DataFrame,
    archivo_salida: str = "a02output/analisis_sistema/resumen_probabilidades_error.csv"
) -> None:
    """
    Genera un CSV resumen con:
    - Modulación
    - M
    - Eb/N0 en dB
    - Ps: probabilidad de error de símbolo
    - Pe: probabilidad de error de bit
    """

    columnas_necesarias = [
        "Modulación",
        "M",
        "EbN0_dB",
        "Pe símbolo",
        "Pe bit"
    ]

    for columna in columnas_necesarias:
        if columna not in tabla_general.columns:
            raise ValueError(
                f"No se encontró la columna '{columna}'. "
                f"Columnas disponibles: {list(tabla_general.columns)}"
            )

    resumen = tabla_general[
        [
            "Modulación",
            "M",
            "EbN0_dB",
            "Pe símbolo",
            "Pe bit"
        ]
    ].copy()

    resumen = resumen.rename(
        columns={
            "Pe símbolo": "Ps",
            "Pe bit": "Pe"
        }
    )

    resumen["Nombre modulación"] = resumen["M"].astype(str) + "-" + resumen["Modulación"].str.replace("M-", "")

    resumen = resumen[
        [
            "Nombre modulación",
            "Modulación",
            "M",
            "EbN0_dB",
            "Ps",
            "Pe"
        ]
    ]

    resumen = resumen.sort_values(
        by=[
            "Modulación",
            "M",
            "EbN0_dB"
        ]
    )

    Path(archivo_salida).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    resumen.to_csv(
        archivo_salida,
        index=False
    )

    print(f"CSV resumen generado: {archivo_salida}")

def main():
    carpeta_salida = Path("a02output/analisis_sistema")
    carpeta_salida.mkdir(
        parents=True,
        exist_ok=True
    )

    todos_los_resultados = []

    configuraciones = [
        ("M-FSK", 2, "binario", False),
        ("M-FSK", 4, "binario", False),
        ("M-FSK", 8, "binario", False),
        ("M-FSK", 16, "binario", False),
        ("M-QAM", 4, "gray", False),
        ("M-QAM", 16, "gray", False),
    ]

    for modulacion, M, tipo_etiquetado, usar_codigo in configuraciones:

        print("\n" + "=" * 72)
        print(f"INICIANDO ANÁLISIS {M}-{modulacion.replace('M-', '')}")
        print("=" * 72)

        tabla = analizar_configuracion(
            modulacion=modulacion,
            M=M,
            tipo_etiquetado=tipo_etiquetado,
            usar_codigo=usar_codigo
        )

        nombre_base = f"analisis_{M}{modulacion.replace('M-', '').lower()}_sin_codigo"

        guardar_tabla_analisis(
            tabla,
            nombre_archivo=nombre_base
        )

        graficar_probabilidades_error_desde_csv(
            archivo_csv=f"a02output/analisis_sistema/{nombre_base}.csv",
            archivo_salida=f"a02output/analisis_sistema/{nombre_base}_curvas_error.png"
        )

        todos_los_resultados.append(tabla)

    tabla_general = pd.concat(
        todos_los_resultados,
        ignore_index=True
    )

    tabla_general.to_csv(
        "a02output/analisis_sistema/analisis_todas_modulaciones.csv",
        index=False
    )

    guardar_csv_resumen_probabilidades(
        tabla_general,
        archivo_salida="a02output/analisis_sistema/resumen_probabilidades_error.csv"
    )

    print("\nProceso finalizado.")
    print("CSV completo: a02output/analisis_sistema/analisis_todas_modulaciones.csv")
    print("CSV resumen: a02output/analisis_sistema/resumen_probabilidades_error.csv")
if __name__ == "__main__":
    main()