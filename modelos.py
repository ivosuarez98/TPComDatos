from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextoTransmision:
    texto: str = ""
    tipo_codificacion_fuente: str = "huffman"

    probabilidades: Dict[str, float] = field(default_factory=dict)
    entropia: float = 0.0
    codigo_huffman: Dict[str, str] = field(default_factory=dict)
    longitud_minima: int = 0
    longitud_promedio: float = 0.0
    longitudes: Dict[str, int] = field(default_factory=dict)

    bits_fuente: List[int] = field(default_factory=list)
    bits_modulador: List[int] = field(default_factory=list)
    bits_relleno_modulacion: int = 0

    usar_codificacion_canal: bool = False
    bits_canal: List[int] = field(default_factory=list)
    relleno_canal: int = 0
    H: Any = None
    S: Any = None
    dmin: Optional[int] = None
    errores_detectables: Optional[int] = None
    errores_corregibles: Optional[int] = None

    modulacion: str = "M-QAM"
    M: int = 16
    tipo_etiquetado: str = "gray"
    bits_por_simbolo: int = 4
    constelacion: Dict[str, List[float]] = field(default_factory=dict)
    grupos_bits: List[str] = field(default_factory=list)
    simbolos_tx: List[List[float]] = field(default_factory=list)
    energia_media_simbolo: float = 0.0
    energia_media_bit: float = 0.0
    energia_media_simbolo_teorica: float = 0.0
    energia_media_bit_teorica: float = 0.0


@dataclass
class SenalTransmitida:
    simbolos: List[List[float]]
    contexto: ContextoTransmision


@dataclass
class SenalRecibida:
    simbolos: List[List[float]]
    contexto: ContextoTransmision
    atenuacion: float = 1.0
    ruido: List[List[float]] = field(default_factory=list)


@dataclass
class ResultadoRecepcion:
    texto_salida: str
    archivo_salida: str
    bits_demodulados: List[int]
    bits_rx: List[int]
    simbolos_demodulados: List[List[float]]
    vector_huffman_rx: List[str] = field(default_factory=list)
    buffer_huffman_pendiente: str = ""
    decodificacion_exitosa: bool = False
    error_decodificacion: str = ""

    errores_simbolo: int = 0
    probabilidad_error_simbolo: float = 0.0
    errores_bit: int = 0
    probabilidad_error_bit: float = 0.0
    errores_bit_demodulador: int = 0
    probabilidad_error_bit_demodulador: float = 0.0

    palabras_canal_corregidas: Any = None
    info_errores_canal: List[Dict[str, Any]] = field(default_factory=list)

    def resumen(self) -> str:
        estado = "OK" if self.decodificacion_exitosa else "CON ADVERTENCIAS"
        return (
            "===== RESUMEN DE RECEPCIÓN =====\n"
            f"Estado decodificación: {estado}\n"
            f"Archivo de salida: {self.archivo_salida}\n"
            f"Errores de símbolo: {self.errores_simbolo}\n"
            f"Probabilidad de error de símbolo: {self.probabilidad_error_simbolo:.6f}\n"
            f"Errores de bit: {self.errores_bit}\n"
            f"Probabilidad de error de bit: {self.probabilidad_error_bit:.6f}\n"
            f"Errores de bit a la salida del demodulador: {self.errores_bit_demodulador}\n"
            f"Probabilidad de error de bit a la salida del demodulador: {self.probabilidad_error_bit_demodulador:.6f}"
        )
