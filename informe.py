from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def generar_tabla_resultado(settings: Settings, senal_tx: SenalTransmitida, senal_rx: SenalRecibida, resultado: ResultadoRecepcion):
    contexto = senal_tx.contexto
    return pd.DataFrame([{
        "Modulación": contexto.modulacion,
        "M": contexto.M,
        "Etiquetado": contexto.tipo_etiquetado,
        "N0": settings.N0,
        "AWGN": settings.usar_awgn,
        "Atenuación habilitada": settings.usar_atenuacion,
        "Atenuación aplicada": round(senal_rx.atenuacion, 6),
        "Codificación canal": contexto.usar_codificacion_canal,
        "Bits fuente": len(contexto.bits_fuente),
        "Bits modulador": len(contexto.bits_modulador),
        "Símbolos TX": len(contexto.simbolos_tx),
        "Es estimada": round(contexto.energia_media_simbolo, 6),
        "Eb estimada": round(contexto.energia_media_bit, 6),
        "Errores símbolo": resultado.errores_simbolo,
        "Pe símbolo": round(resultado.probabilidad_error_simbolo, 8),
        "Errores bit": resultado.errores_bit,
        "Pe bit": round(resultado.probabilidad_error_bit, 8),
        "Decodificación": "OK" if resultado.decodificacion_exitosa else "Con advertencias",
    }])


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
