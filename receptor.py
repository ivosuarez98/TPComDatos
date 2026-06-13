from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from codificacion_canal import decodificar_vector_bits
from modelos import ContextoTransmision, ResultadoRecepcion, SenalRecibida
from modulacion import demodular_simbolos, distancia_cuadratica
from settings import Settings


class Receptor:
    def __init__(self, settings: Settings):
        self.settings = settings

    def decodificar_canal(self, bits_demodulados: List[int], contexto: ContextoTransmision):
        if not contexto.usar_codificacion_canal:
            return bits_demodulados.copy(), None, []
        if contexto.H is None or contexto.S is None:
            raise ValueError("Para decodificar canal se necesitan H y la tabla de síndromes S")

        bits_decodificados, palabras_corregidas, info_errores = decodificar_vector_bits(
            contexto.H,
            contexto.S,
            bits_demodulados,
            relleno=contexto.relleno_canal,
        )
        return bits_decodificados.astype(int).tolist(), palabras_corregidas, info_errores

    def decodificar_huffman(self, bits: List[int], contexto: ContextoTransmision) -> Tuple[str, List[str], str, bool, str]:
        codigo_inverso = {codigo: caracter for caracter, codigo in contexto.codigo_huffman.items()}
        vector_huffman_rx = []
        caracteres = []
        buffer = ""

        for bit in bits:
            buffer += str(int(bit))
            if buffer in codigo_inverso:
                vector_huffman_rx.append(buffer)
                caracteres.append(codigo_inverso[buffer])
                buffer = ""

        texto_salida = "".join(caracteres)
        if buffer:
            return texto_salida, vector_huffman_rx, buffer, False, "Quedaron bits sin decodificar con Huffman"
        return texto_salida, vector_huffman_rx, "", True, ""

    def decodificar_utf8(self, bits: List[int]) -> Tuple[str, List[str], str, bool, str]:
        resto = len(bits) % 8
        if resto:
            bits = bits[:-resto]

        bytes_rx = []
        for i in range(0, len(bits), 8):
            byte = int("".join(str(int(bit)) for bit in bits[i:i + 8]), 2)
            bytes_rx.append(byte)

        try:
            return bytes(bytes_rx).decode("utf-8"), [], "", True, ""
        except UnicodeDecodeError as exc:
            return bytes(bytes_rx).decode("utf-8", errors="replace"), [], "", False, str(exc)

    def decodificar_fuente(self, bits: List[int], contexto: ContextoTransmision):
        if contexto.tipo_codificacion_fuente == "huffman":
            return self.decodificar_huffman(bits, contexto)
        return self.decodificar_utf8(bits)

    def escribir_archivo_salida(self, texto: str) -> None:
        Path(self.settings.ruta_archivo_salida).parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings.ruta_archivo_salida, "w", encoding="utf-8") as archivo:
            archivo.write(texto)

    def calcular_probabilidad_error_simbolo(self, simbolos_tx: List[List[float]], simbolos_demodulados: List[List[float]]) -> Tuple[int, float]:
        if not simbolos_tx:
            return 0, 0.0
        cantidad = min(len(simbolos_tx), len(simbolos_demodulados))
        errores = sum(1 for tx, rx in zip(simbolos_tx[:cantidad], simbolos_demodulados[:cantidad]) if distancia_cuadratica(tx, rx) > 1e-18)
        errores += abs(len(simbolos_tx) - len(simbolos_demodulados))
        return errores, errores / len(simbolos_tx)

    def calcular_probabilidad_error_bit(self, bits_tx: List[int], bits_rx: List[int]) -> Tuple[int, float]:
        if not bits_tx:
            return 0, 0.0
        cantidad = min(len(bits_tx), len(bits_rx))
        errores = sum(1 for tx, rx in zip(bits_tx[:cantidad], bits_rx[:cantidad]) if int(tx) != int(rx))
        errores += abs(len(bits_tx) - len(bits_rx))
        return errores, errores / len(bits_tx)

    def recibir(self, senal_rx: SenalRecibida, contexto: ContextoTransmision | None = None) -> ResultadoRecepcion:
        contexto = contexto or senal_rx.contexto

        bits_demodulados, simbolos_demodulados, _ = demodular_simbolos(
            senal_rx.simbolos,
            contexto.modulacion,
            contexto.M,
            contexto.tipo_etiquetado,
            bits_relleno=contexto.bits_relleno_modulacion,
        )

        bits_rx, palabras_corregidas, info_errores = self.decodificar_canal(bits_demodulados, contexto)
        texto_salida, vector_huffman_rx, buffer_huffman_pendiente, decodificacion_exitosa, error_decodificacion = self.decodificar_fuente(bits_rx, contexto)
        self.escribir_archivo_salida(texto_salida)

        errores_simbolo, pe_simbolo = self.calcular_probabilidad_error_simbolo(contexto.simbolos_tx, simbolos_demodulados)
        errores_bit, pe_bit = self.calcular_probabilidad_error_bit(contexto.bits_fuente, bits_rx)
        errores_bit_demod, pe_bit_demod = self.calcular_probabilidad_error_bit(contexto.bits_modulador, bits_demodulados)

        if decodificacion_exitosa and texto_salida != contexto.texto:
            decodificacion_exitosa = False
            error_decodificacion = "El texto reconstruido no coincide con el texto original"

        return ResultadoRecepcion(
            texto_salida=texto_salida,
            archivo_salida=self.settings.ruta_archivo_salida,
            bits_demodulados=bits_demodulados,
            bits_rx=bits_rx,
            simbolos_demodulados=simbolos_demodulados,
            vector_huffman_rx=vector_huffman_rx,
            buffer_huffman_pendiente=buffer_huffman_pendiente,
            decodificacion_exitosa=decodificacion_exitosa,
            error_decodificacion=error_decodificacion,
            errores_simbolo=errores_simbolo,
            probabilidad_error_simbolo=pe_simbolo,
            errores_bit=errores_bit,
            probabilidad_error_bit=pe_bit,
            errores_bit_demodulador=errores_bit_demod,
            probabilidad_error_bit_demodulador=pe_bit_demod,
            palabras_canal_corregidas=palabras_corregidas,
            info_errores_canal=info_errores,
        )
