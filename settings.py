from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


def generar_matriz_golay_sistematica() -> np.ndarray:
    G_golay_dada = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=int)

    k_canal = 12
    P_golay = G_golay_dada[:, :12]
    return np.concatenate((np.eye(k_canal, dtype=int), P_golay), axis=1)


@dataclass
class Settings:
    # Rutas
    ruta_archivo_entrada: str = "a01input/test.txt"
    ruta_archivo_salida: str = "a02output/mensaje_recibido.txt"
    ruta_informe: str = "a02output/informe.txt"
    carpeta_graficos: str = "a02output"

    # Llaves de funcionalidad
    usar_codificacion_fuente: bool = True
    usar_codificacion_canal: bool = False
    graficar_constelacion_tx: bool = True
    graficar_constelacion_rx: bool = True
    graficar_curvas_error: bool = True
    guardar_graficos: bool = True
    generar_informe: bool = True

    # Modulación
    modulacion: str = "M-QAM"
    M: int = 16
    tipo_etiquetado: str = "gray"  # "binario" o "gray"

    # Canal
    usar_awgn: bool = True
    usar_atenuacion: bool = False
    usar_respuesta_impulsiva: bool = False
    N0: float = 0.157
    Eb_N0_dB: float | None = None
    atenuacion_min: float = 0.5
    atenuacion_max: float = 0.9
    respuesta_impulsiva: List[float] = field(default_factory=lambda: [1.0])

    # Código lineal de bloques
    n_canal: int = 24
    k_canal: int = 12
    G: np.ndarray = field(default_factory=generar_matriz_golay_sistematica)

    # Simulación
    seed: int | None = 1234
    cantidad_maxima_puntos_grafico: int = 8000

    @property
    def archivo_constelacion_tx(self) -> str:
        nombre = f"constelacion_tx_{self.modulacion.replace('-', '_')}_M{self.M}.png"
        return str(Path(self.carpeta_graficos) / nombre)

    @property
    def archivo_constelacion_rx(self) -> str:
        nombre = f"constelacion_rx_{self.modulacion.replace('-', '_')}_M{self.M}.png"
        return str(Path(self.carpeta_graficos) / nombre)

    @property
    def params_codigo_canal(self) -> dict:
        return {
            "k": self.k_canal,
            "n": self.n_canal,
            "G": self.G,
            "usar_codificacion_canal": self.usar_codificacion_canal,
        }

    @classmethod
    def ejemplo_qam(cls, usar_codificacion_canal: bool = False) -> "Settings":
        return cls(
            modulacion="M-QAM",
            M=16,
            tipo_etiquetado="gray",
            N0=0.157,
            usar_codificacion_canal=usar_codificacion_canal,
            graficar_constelacion_tx=True,
            graficar_constelacion_rx=True,
        )

    @classmethod
    def ejemplo_mfsk(cls, usar_codificacion_canal: bool = False) -> "Settings":
        return cls(
            modulacion="M-FSK",
            M=4,
            tipo_etiquetado="binario",
            N0=0.125,
            usar_codificacion_canal=usar_codificacion_canal,
            graficar_constelacion_tx=True,
            graficar_constelacion_rx=True,
        )
