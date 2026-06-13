from __future__ import annotations

import math
from typing import List

import numpy as np

from modelos import SenalRecibida, SenalTransmitida
from modulacion import graficar_constelacion_recibida
from settings import Settings


class Canal:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self.atenuacion = 1.0
        self.ruido: List[List[float]] = []
        self.simbolos_rx: List[List[float]] = []

    def generar_ruido_awgn(self, dimension: int) -> List[float]:
        if not self.settings.usar_awgn:
            return [0.0] * dimension
        sigma = math.sqrt(self.settings.N0 / 2)
        return self.rng.normal(loc=0.0, scale=sigma, size=dimension).astype(float).tolist()

    def generar_atenuacion(self) -> float:
        if not self.settings.usar_atenuacion:
            self.atenuacion = 1.0
        else:
            self.atenuacion = float(self.rng.uniform(self.settings.atenuacion_min, self.settings.atenuacion_max))
        return self.atenuacion

    def aplicar_respuesta_impulsiva(self, simbolos: List[List[float]]) -> List[List[float]]:
        if not self.settings.usar_respuesta_impulsiva:
            return simbolos

        h = np.array(self.settings.respuesta_impulsiva, dtype=float)
        if h.size == 0 or np.allclose(h, [1.0]):
            return simbolos

        matriz = np.array(simbolos, dtype=float)
        salida = np.zeros_like(matriz)
        for dimension in range(matriz.shape[1]):
            salida[:, dimension] = np.convolve(matriz[:, dimension], h, mode="full")[:matriz.shape[0]]
        return salida.tolist()

    def transmitir(self, senal_tx: SenalTransmitida) -> SenalRecibida:
        simbolos_procesados = self.aplicar_respuesta_impulsiva(senal_tx.simbolos)
        self.generar_atenuacion()

        self.simbolos_rx = []
        self.ruido = []
        for simbolo in simbolos_procesados:
            ruido_simbolo = self.generar_ruido_awgn(len(simbolo))
            simbolo_rx = [self.atenuacion * coordenada + ruido for coordenada, ruido in zip(simbolo, ruido_simbolo)]
            self.simbolos_rx.append(simbolo_rx)
            self.ruido.append(ruido_simbolo)

        return SenalRecibida(
            simbolos=self.simbolos_rx,
            contexto=senal_tx.contexto,
            atenuacion=self.atenuacion,
            ruido=self.ruido,
        )

    def graficar_constelacion_recibida(self, senal_rx: SenalRecibida, archivo_figura: str | None = None) -> None:
        contexto = senal_rx.contexto
        graficar_constelacion_recibida(
            senal_rx.simbolos,
            contexto.constelacion,
            contexto.modulacion,
            contexto.M,
            contexto.tipo_etiquetado,
            archivo_figura or self.settings.archivo_constelacion_rx,
            senal_rx.atenuacion,
            self.settings.N0,
        )
