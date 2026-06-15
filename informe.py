from __future__ import annotations

from pathlib import Path

import pandas as pd
import math

from modelos import ResultadoRecepcion, SenalRecibida, SenalTransmitida
from settings import Settings


def generar_tabla_huffman(probabilidades, codigo_huffman):
    filas = []
    for caracter, probabilidad in probabilidades.items():
        filas.append({
            "Caracter": repr(caracter),
            "Probabilidad": round(probabilidad, 6),
            "Probabilidad (%)": round(probabilidad * 100, 2),
            "Código Huffman": codigo_huffman[caracter],
            "Longitud": len(codigo_huffman[caracter]),
        })
    return pd.DataFrame(filas).sort_values(by="Probabilidad", ascending=False).reset_index(drop=True)


def generar_tabla_metricas_fuente(contexto):
    if contexto.tipo_codificacion_fuente != "huffman":
        return pd.DataFrame([{
            "Métrica": "Codificación de fuente",
            "Valor": "UTF-8 directo, Huffman deshabilitado",
            "Unidad": "-",
        }])

    eficiencia = contexto.entropia / contexto.longitud_promedio if contexto.longitud_promedio else 0
    return pd.DataFrame({
        "Métrica": [
            "Entropía de la fuente",
            "Longitud mínima Huffman",
            "Longitud promedio Huffman",
            "Eficiencia del código Huffman",
            "Longitud código fijo ASCII",
        ],
        "Valor": [
            round(contexto.entropia, 4),
            contexto.longitud_minima,
            round(contexto.longitud_promedio, 4),
            round(eficiencia * 100, 2),
            8,
        ],
        "Unidad": ["bits/símbolo", "bits", "bits/símbolo", "%", "bits/símbolo"],
    })


def generar_tabla_resultado(settings, senal_tx, senal_rx, resultado):
    contexto = senal_tx.contexto

    Es_teorica, Eb_teorica = calcular_energias_teoricas(contexto)

    fila = {
        "Modulación": contexto.modulacion,
        "M": contexto.M,
        "Etiquetado": contexto.tipo_etiquetado,
        "N0": settings.N0,
        "AWGN": settings.usar_awgn,
        "Atenuación": settings.usar_atenuacion,
        "Atenuación aplicada": round(senal_rx.atenuacion, 6),

        "Bits fuente": len(contexto.bits_fuente),
        "Bits modulador": len(contexto.bits_modulador),
        "Símbolos TX": len(contexto.simbolos_tx),

        "Es teórica": round(Es_teorica, 6) if Es_teorica is not None else None,
        "Eb teórica": round(Eb_teorica, 6) if Eb_teorica is not None else None,
        "Es estimada": round(contexto.energia_media_simbolo, 6),
        "Eb estimada": round(contexto.energia_media_bit, 6),

        "Errores símbolo": resultado.errores_simbolo,
        "Pe símbolo": round(resultado.probabilidad_error_simbolo, 8),
        "Errores bit": resultado.errores_bit,
        "Pe bit": round(resultado.probabilidad_error_bit, 8),

        "Decodificación": "OK" if resultado.decodificacion_exitosa else "Con errores",
    }

    return pd.DataFrame([fila])


def escribir_tabla(archivo, titulo, dataframe):
    archivo.write("\n")
    archivo.write("=" * 72 + "\n")
    archivo.write(titulo + "\n")
    archivo.write("=" * 72 + "\n")
    if dataframe is None or dataframe.empty:
        archivo.write("No hay datos disponibles.\n")
    else:
        archivo.write(dataframe.to_string(index=False))
        archivo.write("\n")


def generar_informe(settings: Settings, senal_tx: SenalTransmitida, senal_rx: SenalRecibida, resultado: ResultadoRecepcion, archivo_informe: str | None = None) -> str:
    ruta = Path(archivo_informe or settings.ruta_informe)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    contexto = senal_tx.contexto

    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("INFORME DEL SISTEMA DE COMUNICACIÓN DIGITAL\n")
        archivo.write("=" * 72 + "\n")
        archivo.write(f"Archivo de entrada: {settings.ruta_archivo_entrada}\n")
        archivo.write(f"Archivo de salida: {settings.ruta_archivo_salida}\n")
        archivo.write(f"Cantidad de caracteres: {len(contexto.texto)}\n")
        archivo.write(f"Modulación: {contexto.modulacion}, M={contexto.M}, etiquetado={contexto.tipo_etiquetado}\n")
        archivo.write(f"Canal: AWGN={settings.usar_awgn}, atenuación={settings.usar_atenuacion}, respuesta impulsiva={settings.usar_respuesta_impulsiva}\n")

        escribir_tabla(archivo, "1) Métricas de fuente", generar_tabla_metricas_fuente(contexto))

        if contexto.tipo_codificacion_fuente == "huffman":
            escribir_tabla(archivo, "2) Tabla Huffman", generar_tabla_huffman(contexto.probabilidades, contexto.codigo_huffman))

        escribir_tabla(archivo, "3) Resultado de la cadena", generar_tabla_resultado(settings, senal_tx, senal_rx, resultado))

        if contexto.usar_codificacion_canal:
            archivo.write("\n")
            archivo.write("=" * 72 + "\n")
            archivo.write("4) Código de canal\n")
            archivo.write("=" * 72 + "\n")
            archivo.write(f"n = {settings.n_canal}\n")
            archivo.write(f"k = {settings.k_canal}\n")
            archivo.write(f"dmin = {contexto.dmin}\n")
            archivo.write(f"Errores detectables = {contexto.errores_detectables}\n")
            archivo.write(f"Errores corregibles = {contexto.errores_corregibles}\n")
            archivo.write(f"Palabras con error detectado = {sum(1 for info in resultado.info_errores_canal if info['error_detectado'])}\n")
            archivo.write(f"Palabras corregidas = {sum(1 for info in resultado.info_errores_canal if info['error_corregido'])}\n")

        archivo.write("\n")
        archivo.write("=" * 72 + "\n")
        archivo.write("5) Verificación final\n")
        archivo.write("=" * 72 + "\n")
        archivo.write(resultado.resumen() + "\n")
        if resultado.error_decodificacion:
            archivo.write(f"Detalle: {resultado.error_decodificacion}\n")

    return str(ruta)
def generar_tabla_comparativa_casos(casos):
    filas = []

    for caso in casos:
        nombre_caso = caso["nombre_caso"]
        settings = caso["settings"]
        senal_tx = caso["senal_tx"]
        senal_rx = caso["senal_rx"]
        resultado = caso["resultado"]
        contexto = senal_tx.contexto

        Es_teorica, Eb_teorica = calcular_energias_teoricas(contexto)

        filas.append({
            "Caso": nombre_caso,
            "Modulación": contexto.modulacion,
            "M": contexto.M,
            "Etiquetado": contexto.tipo_etiquetado,
            "N0": settings.N0,
            "AWGN": settings.usar_awgn,
            "Atenuación": settings.usar_atenuacion,
            "Atenuación aplicada": round(senal_rx.atenuacion, 6),
            "Codificación canal": contexto.usar_codificacion_canal,

            "Bits fuente": len(contexto.bits_fuente),
            "Bits modulador": len(contexto.bits_modulador),
            "Símbolos TX": len(contexto.simbolos_tx),

            "Es teórica": round(Es_teorica, 6) if Es_teorica is not None else None,
            "Eb teórica": round(Eb_teorica, 6) if Eb_teorica is not None else None,
            "Es estimada": round(contexto.energia_media_simbolo, 6),
            "Eb estimada": round(contexto.energia_media_bit, 6),

            "Errores símbolo": resultado.errores_simbolo,
            "Pe símbolo": round(resultado.probabilidad_error_simbolo, 8),
            "Errores bit": resultado.errores_bit,
            "Pe bit": round(resultado.probabilidad_error_bit, 8),
            "Decodificación": "OK" if resultado.decodificacion_exitosa else "Con advertencias",
            "Archivo salida": resultado.archivo_salida,
            "Informe individual": settings.ruta_informe,
        })

    return pd.DataFrame(filas)


def generar_informe_conjunto(casos, archivo_informe="a02output/informe_comparativo.txt"):
    ruta = Path(archivo_informe)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write("INFORME COMPARATIVO DEL SISTEMA DE COMUNICACIÓN DIGITAL\n")
        archivo.write("=" * 72 + "\n")

        archivo.write("\n")
        archivo.write(f"Cantidad de casos simulados: {len(casos)}\n")

        if len(casos) == 0:
            archivo.write("No hay casos disponibles para comparar.\n")
            return str(ruta)

        primer_caso = casos[0]
        contexto_base = primer_caso["senal_tx"].contexto

        archivo.write(f"Archivo de entrada: {primer_caso['settings'].ruta_archivo_entrada}\n")
        archivo.write(f"Cantidad de caracteres: {len(contexto_base.texto)}\n")

        escribir_tabla(
            archivo,
            "1) Métricas de fuente",
            generar_tabla_metricas_fuente(contexto_base)
        )

        if contexto_base.tipo_codificacion_fuente == "huffman":
            escribir_tabla(
                archivo,
                "2) Tabla Huffman",
                generar_tabla_huffman(
                    contexto_base.probabilidades,
                    contexto_base.codigo_huffman
                )
            )

        escribir_tabla(
            archivo,
            "3) Tabla comparativa de casos simulados",
            generar_tabla_comparativa_casos(casos)
        )

        archivo.write("\n")
        archivo.write("=" * 72 + "\n")
        archivo.write("4) Resumen individual por caso\n")
        archivo.write("=" * 72 + "\n")

        for caso in casos:
            archivo.write("\n")
            archivo.write("-" * 72 + "\n")
            archivo.write(f"Caso: {caso['nombre_caso']}\n")
            archivo.write("-" * 72 + "\n")
            archivo.write(caso["resultado"].resumen())
            archivo.write("\n")

    return str(ruta)

def calcular_energias_teoricas(contexto):
    """
    Calcula Es y Eb teóricas a partir de la constelación ideal.

    Es teórica: promedio de energía de todos los símbolos posibles.
    Eb teórica: Es teórica dividida por la cantidad de bits por símbolo.
    """

    constelacion = contexto.constelacion

    if constelacion is None or len(constelacion) == 0:
        return None, None

    energias = []

    for simbolo in constelacion.values():
        energia = sum(coordenada ** 2 for coordenada in simbolo)
        energias.append(energia)

    Es_teorica = sum(energias) / len(energias)

    bits_por_simbolo = int(math.log2(contexto.M))
    Eb_teorica = Es_teorica / bits_por_simbolo

    return Es_teorica, Eb_teorica