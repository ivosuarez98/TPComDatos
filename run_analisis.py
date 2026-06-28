import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analisis import run_system_analysis, generar_graficos_informe

def main():
    out_dir = "a02output/analysis"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print(" PROCESANDO SIMULACIONES ADAPTADAS PARA CÁTEDRA TA137")
    print("=" * 70)
    
    rango_ebn0 = list(range(0, 11))
    
    print("\n[PASO 1] Corriendo canales puros respetando restricciones de M...")
    dfs_sin = run_system_analysis(ebn0_db_range=rango_ebn0, use_channel_coding=False)
    
    print("\n[PASO 2] Corriendo canales con codificación lineal de bloques...")
    dfs_con = run_system_analysis(ebn0_db_range=rango_ebn0, use_channel_coding=True)
    
    print("\n[PASO 3] Generando reportes visuales con/sin código...")
    generar_graficos_informe(dfs_sin, dfs_con, output_dir=out_dir)
    
    print("\n" + "=" * 70)
    print(f" ¡PROCESO COMPLETADO! Gráficos generados correctamente en: {out_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    main()