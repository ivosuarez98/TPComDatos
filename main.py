from settings import Settings
from transmisor import Transmisor
from canal import Canal
from receptor import Receptor
from informe import generar_informe, generar_informe_conjunto


def nombre_caso(settings: Settings) -> str:
    modulacion = settings.modulacion.replace("M-", "").replace("-", "").lower()
    codigo = "con_codigo" if settings.usar_codificacion_canal else "sin_codigo"
    canal = "awgn" if settings.usar_awgn else "ideal"

    return f"{modulacion}_M{settings.M}_{settings.tipo_etiquetado}_{codigo}_{canal}"


def preparar_rutas(settings: Settings) -> str:
    caso = nombre_caso(settings)

    settings.ruta_archivo_salida = f"a02output/{caso}/mensaje_recibido.txt"
    settings.ruta_informe = f"a02output/{caso}/informe.txt"
    settings.carpeta_graficos = f"a02output/{caso}"

    return caso


def ejecutar_caso(settings: Settings) -> dict:
    caso = preparar_rutas(settings)

    transmisor = Transmisor(settings)
    canal = Canal(settings)
    receptor = Receptor(settings)

    senal_tx = transmisor.transmitir()
    senal_rx = canal.transmitir(senal_tx)
    resultado = receptor.recibir(senal_rx, contexto=senal_tx.contexto)

    if settings.graficar_constelacion_rx and settings.guardar_graficos:
        canal.graficar_constelacion_recibida(senal_rx)

    if settings.generar_informe:
        generar_informe(settings, senal_tx, senal_rx, resultado)

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
    }


def main():
    configuraciones = [
        Settings(
            modulacion="M-QAM",
            M=16,
            tipo_etiquetado="gray",
            usar_codificacion_canal=False,
            usar_awgn=True,
            usar_atenuacion=False,
            usar_respuesta_impulsiva=False,
            N0=0.2512, 
        ),
    ]

    resultados = []

    for settings in configuraciones:
        resultado_caso = ejecutar_caso(settings)
        resultados.append(resultado_caso)

    generar_informe_conjunto(
        resultados,
        archivo_informe="a02output/informe_comparativo.txt"
    )

    print("\nProceso finalizado.")
    print("Se generó un informe individual por caso.")
    print("Se generó el informe comparativo en: a02output/informe_comparativo.txt")


if __name__ == "__main__":
    main()