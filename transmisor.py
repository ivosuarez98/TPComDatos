from __future__ import annotations

import heapq
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from codificacion_canal import (
    calcular_parametros_codigo,
    calcular_tabla_sindromes,
    codificar_vector_bits,
    obtener_matriz_paridad,
)
from modelos import ContextoTransmision, SenalTransmitida
from modulacion import (
    calcular_energias,
    calcular_energias_constelacion,
    graficar_constelacion,
    modular_bits,
)
from settings import Settings


class Transmisor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def leer_texto(self) -> str:
        with open(self.settings.ruta_archivo_entrada, "r", encoding="utf-8") as archivo:
            texto = archivo.read()
        if not texto:
            raise ValueError("El archivo de entrada está vacío")
        return texto

    def calcular_probabilidades(self, texto: str) -> Dict[str, float]:
        cantidad_total = len(texto)
        conteo = Counter(texto)
        return {caracter: cantidad / cantidad_total for caracter, cantidad in conteo.items()}

    def calcular_entropia(self, probabilidades: Dict[str, float]) -> float:
        return sum(-p * math.log2(p) for p in probabilidades.values())

    def generar_codigo_huffman(self, probabilidades: Dict[str, float]) -> Dict[str, str]:
        heap = []
        contador = 0
        for caracter, probabilidad in probabilidades.items():
            heapq.heappush(heap, (probabilidad, contador, caracter))
            contador += 1

        if len(heap) == 1:
            unico = heapq.heappop(heap)[2]
            return {unico: "0"}

        while len(heap) > 1:
            prob1, _, arbol1 = heapq.heappop(heap)
            prob2, _, arbol2 = heapq.heappop(heap)
            heapq.heappush(heap, (prob1 + prob2, contador, (arbol1, arbol2)))
            contador += 1

        arbol_huffman = heapq.heappop(heap)[2]
        codigo_huffman = {}

        def recorrer(nodo, palabra: str) -> None:
            if isinstance(nodo, str):
                codigo_huffman[nodo] = palabra
            else:
                izquierda, derecha = nodo
                recorrer(izquierda, palabra + "0")
                recorrer(derecha, palabra + "1")

        recorrer(arbol_huffman, "")
        return codigo_huffman

    def calcular_longitudes(self, probabilidades: Dict[str, float], codigo_huffman: Dict[str, str]) -> Tuple[int, float, Dict[str, int]]:
        longitudes = {caracter: len(codigo) for caracter, codigo in codigo_huffman.items()}
        longitud_minima = min(longitudes.values())
        longitud_promedio = sum(probabilidades[caracter] * longitudes[caracter] for caracter in probabilidades)
        return longitud_minima, longitud_promedio, longitudes

    def codificar_texto_huffman(self, texto: str, codigo_huffman: Dict[str, str]) -> Tuple[List[str], List[int]]:
        vector_huffman = [codigo_huffman[caracter] for caracter in texto]
        bits = [int(bit) for palabra in vector_huffman for bit in palabra]
        return vector_huffman, bits

    def codificar_texto_utf8(self, texto: str) -> List[int]:
        bits = []
        for byte in texto.encode("utf-8"):
            bits.extend(int(bit) for bit in format(byte, "08b"))
        return bits

    def procesar_fuente(self) -> ContextoTransmision:
        texto = self.leer_texto()
        contexto = ContextoTransmision(texto=texto)

        if self.settings.usar_codificacion_fuente:
            probabilidades = self.calcular_probabilidades(texto)
            codigo_huffman = self.generar_codigo_huffman(probabilidades)
            longitud_minima, longitud_promedio, longitudes = self.calcular_longitudes(probabilidades, codigo_huffman)
            _, bits_fuente = self.codificar_texto_huffman(texto, codigo_huffman)

            contexto.tipo_codificacion_fuente = "huffman"
            contexto.probabilidades = probabilidades
            contexto.entropia = self.calcular_entropia(probabilidades)
            contexto.codigo_huffman = codigo_huffman
            contexto.longitud_minima = longitud_minima
            contexto.longitud_promedio = longitud_promedio
            contexto.longitudes = longitudes
            contexto.bits_fuente = bits_fuente
        else:
            contexto.tipo_codificacion_fuente = "utf8"
            contexto.bits_fuente = self.codificar_texto_utf8(texto)

        return contexto

    def aplicar_codificacion_canal(self, contexto: ContextoTransmision) -> None:
        contexto.usar_codificacion_canal = self.settings.usar_codificacion_canal

        if not self.settings.usar_codificacion_canal:
            contexto.bits_canal = contexto.bits_fuente.copy()
            contexto.bits_modulador = contexto.bits_canal.copy()
            contexto.relleno_canal = 0
            return

        params = self.settings.params_codigo_canal
        k = int(params["k"])
        n = int(params["n"])
        G = params["G"]

        H = obtener_matriz_paridad(k, n, G)
        S = calcular_tabla_sindromes(H)
        dmin, errores_detectables, errores_corregibles = calcular_parametros_codigo(k, n, G)
        bits_codificados, relleno = codificar_vector_bits(contexto.bits_fuente, k, n, G)

        contexto.bits_canal = bits_codificados.astype(int).tolist()
        contexto.bits_modulador = contexto.bits_canal.copy()
        contexto.relleno_canal = relleno
        contexto.H = H
        contexto.S = S
        contexto.dmin = dmin
        contexto.errores_detectables = errores_detectables
        contexto.errores_corregibles = errores_corregibles

    def modular(self, contexto: ContextoTransmision) -> None:
        simbolos, constelacion, grupos, relleno, bits_por_simbolo = modular_bits(
            contexto.bits_modulador,
            self.settings.modulacion,
            self.settings.M,
            self.settings.tipo_etiquetado,
        )

        contexto.modulacion = self.settings.modulacion
        contexto.M = self.settings.M
        contexto.tipo_etiquetado = self.settings.tipo_etiquetado
        contexto.bits_por_simbolo = bits_por_simbolo
        contexto.constelacion = constelacion
        contexto.grupos_bits = grupos
        contexto.bits_relleno_modulacion = relleno
        contexto.simbolos_tx = simbolos
        contexto.energia_media_simbolo, contexto.energia_media_bit = calcular_energias(simbolos, bits_por_simbolo)
        contexto.energia_media_simbolo_teorica, contexto.energia_media_bit_teorica = calcular_energias_constelacion(constelacion, bits_por_simbolo)

    def transmitir(self) -> SenalTransmitida:
        contexto = self.procesar_fuente()
        self.aplicar_codificacion_canal(contexto)
        self.modular(contexto)

        if self.settings.graficar_constelacion_tx and self.settings.guardar_graficos:
            graficar_constelacion(
                contexto.constelacion,
                contexto.modulacion,
                contexto.M,
                contexto.tipo_etiquetado,
                self.settings.archivo_constelacion_tx,
            )

        return SenalTransmitida(simbolos=contexto.simbolos_tx, contexto=contexto)
