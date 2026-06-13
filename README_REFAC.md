# Refactorización TP - Taller de Comunicaciones Digitales

Estructura propuesta:

- `settings.py`: configuración centralizada y ejemplos QAM/MFSK.
- `transmisor.py`: lectura, Huffman/UTF-8, codificación de canal opcional, modulación y energías.
- `canal.py`: AWGN, atenuación y respuesta impulsiva opcional.
- `receptor.py`: demodulación, decodificación de canal opcional, decodificación de fuente, escritura y métricas.
- `modulacion.py`: funciones comunes de constelación, modulación, demodulación y gráficos.
- `codificacion_canal.py`: código lineal de bloques, H, síndromes, corrección y parámetros.
- `modelos.py`: dataclasses para intercambiar datos entre bloques.
- `informe.py`: generación de tablas e informe sin ejecutar la simulación.
- `main.py`: orquestación compacta.

Ejecutar desde esta carpeta:

```bash
python main.py
```

Asegurate de que exista el archivo configurado en `Settings.ruta_archivo_entrada`.
