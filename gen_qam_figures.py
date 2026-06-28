"""
Genera figuras QAM para el Módulo F usando entrada pequeña.
"""
import sys, os, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

BACKUP = "a01input/test.txt.bak"
ORIG = "a01input/test.txt"
SMALL = "a01input/test2.txt"

shutil.copy(ORIG, BACKUP)
shutil.copy(SMALL, ORIG)

try:
    from analisis import run_system_analysis, generar_graficos_informe
    out_dir = "a02output/analysis"
    os.makedirs(out_dir, exist_ok=True)
    rango = list(range(0, 11))

    print("[1/3] Sin codificación...")
    dfs_sin = run_system_analysis(ebn0_db_range=rango, use_channel_coding=False)
    
    print("[2/3] Con codificación...")
    dfs_con = run_system_analysis(ebn0_db_range=rango, use_channel_coding=True)
    
    print("[3/3] Generando gráficos...")
    generar_graficos_informe(dfs_sin, dfs_con, output_dir=out_dir)
    print("OK! Figuras:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")
finally:
    shutil.move(BACKUP, ORIG)
