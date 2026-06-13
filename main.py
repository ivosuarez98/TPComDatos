from settings import Settings
from transmisor import Transmisor
from canal import Canal
from receptor import Receptor
from informe import generar_informe


def ejecutar_caso(settings):
    transmisor = Transmisor(settings)
    canal = Canal(settings)
    receptor = Receptor(settings)

    senal_tx = transmisor.transmitir()
    senal_rx = canal.transmitir(senal_tx)
    resultado = receptor.recibir(senal_rx, contexto=senal_tx.contexto)

    if settings.generar_informe:
        generar_informe(settings, senal_tx, senal_rx, resultado)

    print(resultado.resumen())

    return resultado


def main():
    configuraciones = [
        Settings(
            modulacion="M-QAM",
            M=4,
            tipo_etiquetado="gray",
            ruta_archivo_salida="a02output/mensaje_qam4.txt",
            generar_informe=True,
        ),
        Settings(
            modulacion="M-QAM",
            M=16,
            tipo_etiquetado="gray",
            ruta_archivo_salida="a02output/mensaje_qam16.txt",
            generar_informe=True,
        ),
        Settings(
            modulacion="M-FSK",
            M=2,
            tipo_etiquetado="binario",
            ruta_archivo_salida="a02output/mensaje_fsk2.txt",
            generar_informe=True,
        ),
        Settings(
            modulacion="M-FSK",
            M=4,
            tipo_etiquetado="binario",
            ruta_archivo_salida="a02output/mensaje_fsk4.txt",
            generar_informe=True,
        ),
    ]

    resultados = []

    for settings in configuraciones:
        resultado = ejecutar_caso(settings)
        resultados.append(resultado)


if __name__ == "__main__":
    main()