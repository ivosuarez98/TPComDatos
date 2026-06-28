"""
Módulo F - Análisis del Sistema (Versión Exclusiva M-QAM / M-FSK)
Genera los gráficos solicitados en los puntos a, b, c, d y e.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.special import erfc
from math import log2, pi, sin, sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from enum import Enum

from settings import Settings
from transmisor import Transmisor
from canal import Canal
from receptor import Receptor

class SymbolRep(Enum):
    SYMBOL = 0
    BIT = 1
    def __str__(self):
        return "symb" if self == SymbolRep.SYMBOL else "bit"

def q_function(x: float) -> float:
    return 0.5 * erfc(x / np.sqrt(2))

# --- FÓRMULAS TEÓRICAS CONFIGURADAS ---

def pe_psk_theoretical(M: int, ebn0_linear: float) -> float:
    k = log2(M)
    arg = np.sqrt(2 * k * ebn0_linear) * sin(pi / M)
    return (1 if M == 2 else 2) * q_function(arg)

def pe_qam_theoretical(M: int, ebn0_linear: float) -> float:
    if M == 2: return pe_psk_theoretical(2, ebn0_linear)
    k = log2(M)
    esn0 = ebn0_linear * k
    factor = 2 * (1 - 1 / sqrt(M))
    arg = sqrt(3 * esn0 / (M - 1))
    p_coor = factor * q_function(arg)
    return 1.0 - (1.0 - p_coor) ** 2

def pb_qam_theoretical(M: int, ebn0_linear: float) -> float:
    return pe_qam_theoretical(M, ebn0_linear) / log2(M)

def pe_fsk_theoretical(M: int, ebn0_linear: float) -> float:
    k = log2(M)
    arg = np.sqrt(k * ebn0_linear)
    return (M - 1) * q_function(arg)

def pb_fsk_theoretical(M: int, ebn0_linear: float) -> float:
    return 0.5 * M * pe_fsk_theoretical(M, ebn0_linear) / (M - 1)


def run_system_analysis(
    ebn0_db_range: List[float] = list(range(0, 11)),
    use_channel_coding: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Ejecuta el barrido analizando únicamente M-QAM y M-FSK con sus órdenes válidos.
    Muestra los errores reales en consola si la simulación falla o se desvía.
    """
    config_modulaciones = {
        "M-QAM": [4,16],        
        "M-FSK": [2,4,8,16]   
    }
    
    results = {}

    for scheme, M_list in config_modulaciones.items():
        scheme_result = []
        for M in M_list:
            k_bits = log2(M) # Cantidad de bits por símbolo
            
            for ebn0_db in ebn0_db_range:
                ebn0_linear = 10**(ebn0_db / 10.0)
                
                # --- DESPLAZAMIENTO TEÓRICO DE REFERENCIA ---
                # Si el sistema usa codificación de canal, la curva teórica se debe evaluar 
                # con la Eb efectiva (atenuada por la tasa de código Rc = 12/24)
                ebn0_th_eff = ebn0_linear * (12.0 / 24.0) if use_channel_coding else ebn0_linear

                if scheme == "M-QAM":
                    Pe_th = pe_qam_theoretical(M, ebn0_th_eff)
                    Pb_th = pb_qam_theoretical(M, ebn0_th_eff)
                else:
                    Pe_th = pe_fsk_theoretical(M, ebn0_th_eff)
                    Pb_th = pb_fsk_theoretical(M, ebn0_th_eff)

                try:
                    # --- CORRECCIÓN DE N0 PARA ACUMULAR LA PENALIZACIÓN DE REDUNDANCIA ---
                    # Como el transmisor/canal no alteran la energía geométrica, 
                    # aumentamos el ruido (N0) para reflejar la pérdida por tasa de código (Rc = 12/24)
                    N0_dinamico = 1.0 / ebn0_linear
                    if use_channel_coding:
                        Rc = 12.0 / 24.0
                        N0_dinamico = N0_dinamico / Rc  # Incrementa el ruido en el canal físico
                        
                    cfg = Settings(
                        modulacion=scheme,
                        M=M,
                        tipo_etiquetado="gray",
                        N0=N0_dinamico,              
                        usar_awgn=True,              
                        usar_codificacion_canal=use_channel_coding,
                        graficar_constelacion_tx=False,
                        graficar_constelacion_rx=False,
                        generar_informe=False
                    )
                    
                    transmisor = Transmisor(cfg)
                    canal = Canal(cfg)
                    canal.eb_n0_db = ebn0_db  
                    receptor = Receptor(cfg)

                    # --- ENTRADA DE DATOS DESDE LA FUENTE POR DEFECTO ---
                    contexto_tx = transmisor.procesar_fuente()
                    
                    transmisor.aplicar_codificacion_canal(contexto_tx)
                    transmisor.modular(contexto_tx)
                    
                    from modelos import SenalTransmitida
                    senal_tx = SenalTransmitida(
                        simbolos=contexto_tx.simbolos_tx,
                        contexto=contexto_tx
                    )
                    
                    senal_rx = canal.transmitir(senal_tx)
                    resultado_rx = receptor.recibir(
                        senal_rx,
                        contexto=senal_tx.contexto
                    )
                    
                    # --- CÁLCULO DE MÉTRICAS ---
                    bits_enviados_fuente = senal_tx.contexto.bits_fuente
                    bits_recibidos_fuente = resultado_rx.bits_rx
                    
                    if use_channel_coding:
                        errores_reales = np.sum(
                            np.array(bits_enviados_fuente) != np.array(bits_recibidos_fuente)
                        )
                        Pb_sim = errores_reales / len(bits_enviados_fuente)
                        Pe_sim = 1.0 - (1.0 - Pb_sim) ** k_bits
                    else:
                        Pb_sim = getattr(
                            resultado_rx,
                            "probabilidad_error_bit_demodulador",
                            0.0
                        )
                        Pe_sim = getattr(
                            resultado_rx,
                            "probabilidad_error_simbolo",
                            0.0
                        )
                    
                    if Pb_sim == 0:
                        Pb_sim = 1e-6
                    
                    if Pe_sim == 0:
                        Pe_sim = 1e-6
                    
                except Exception as e:
                    print(f"[ALERTA SIM] Falló {scheme} M={M} en Eb/N0={ebn0_db}dB. Error: {e}")
                    # CORRECCIÓN CLAVE: Si se rompe o da error de dimensiones, se calcula una 
                    # aproximación penalizada, nunca la curva teórica ideal limpia.
                    Pb_sim, Pe_sim = 0.5, 0.5

                scheme_result.append({
                    "M": M,
                    "ebn0_db": ebn0_db,
                    "theory_bit": Pb_th,
                    "theory_symb": Pe_th,
                    "sim_bit": Pb_sim,
                    "sim_symb": Pe_sim
                })
        results[scheme] = pd.DataFrame(scheme_result)
    return results

# =====================================================================
# --- RUTINAS DE GRAFICACIÓN AJUSTADAS A QAM / FSK ------
# =====================================================================

def generar_graficos_informe(dfs_sin: Dict[str, pd.DataFrame], dfs_con: Dict[str, pd.DataFrame], output_dir: str = "data/analysis"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    colors = {2: 'darkblue', 4: 'forestgreen', 8: 'darkorange', 16: 'firebrick'}

    # -----------------------------------------------------------------
    # PUNTO A y B: Curvas Pe y Pb independientes (M-QAM usa 4,16 y M-FSK usa 2,4,8,16)
    # -----------------------------------------------------------------
    for scheme, df in dfs_sin.items():
        M_actuales = df["M"].unique()
        
        # Gráfico Punto A: Pe (Símbolo)
        fig_a, ax_a = plt.subplots(figsize=(9, 6))
        for M in M_actuales:
            df_m = df[df["M"] == M].sort_values("ebn0_db")
            ax_a.semilogy(df_m["ebn0_db"], df_m["theory_symb"], linestyle="--", color=colors[M], label=f"M={M} Teórico")
            ax_a.scatter(df_m["ebn0_db"], df_m["sim_symb"], marker="o", color=colors[M], edgecolor='black', s=50, label=f"M={M} Simulado")
        ax_a.set_xlabel("$E_b/N_0$ (dB)")
        ax_a.set_ylabel("Probabilidad de Error de Símbolo ($P_e$)")
        ax_a.set_title(f"Rendimiento $P_e$ para {scheme} (Sin Codificación)")
        ax_a.grid(True, which="both", alpha=0.4, linestyle=":")
        ax_a.legend(loc="lower left")
        ax_a.set_ylim(1e-6, 1.1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/punto_a_{scheme}_Pe.png", dpi=150)
        plt.close()

        # Gráfico Punto B: Pb (Bit)
        fig_b, ax_b = plt.subplots(figsize=(9, 6))
        for M in M_actuales:
            df_m = df[df["M"] == M].sort_values("ebn0_db")
            ax_b.semilogy(df_m["ebn0_db"], df_m["theory_bit"], linestyle="--", color=colors[M], label=f"M={M} Teórico")
            ax_b.scatter(df_m["ebn0_db"], df_m["sim_bit"], marker="s", color=colors[M], edgecolor='black', s=50, label=f"M={M} Simulado")
        ax_b.set_xlabel("$E_b/N_0$ (dB)")
        ax_b.set_ylabel("Probabilidad de Error de Bit ($P_b$)")
        ax_b.set_title(f"Rendimiento $P_b$ para {scheme} (Sin Codificación)")
        ax_b.grid(True, which="both", alpha=0.4, linestyle=":")
        ax_b.legend(loc="lower left")
        ax_b.set_ylim(1e-6, 1.1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/punto_b_{scheme}_Pb.png", dpi=150)
        plt.close()

    # -----------------------------------------------------------------
    # PUNTO C: Comparación QAM vs FSK para M=4 y M=16
    # -----------------------------------------------------------------
    
    fig_c, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, M in zip(axs, [4, 16]):
    
        for sch, col, mark in zip(
            ["M-QAM", "M-FSK"],
            ["navy", "crimson"],
            ["o", "s"]
        ):
            df = dfs_sin[sch][dfs_sin[sch]["M"] == M].sort_values("ebn0_db")
    
            ax.semilogy(
                df["ebn0_db"],
                df["theory_bit"],
                linestyle="-",
                color=col,
                label=f"{sch} Teórico"
            )
    
            ax.scatter(
                df["ebn0_db"],
                df["sim_bit"],
                marker=mark,
                color=col,
                s=60,
                edgecolor="black",
                label=f"{sch} Simulado"
            )
    
        ax.set_xlabel("$E_b/N_0$ (dB)")
        ax.set_title(f"M = {M}")
        ax.grid(True, which="both", alpha=0.5)
        ax.set_ylim(1e-6, 1.1)
        ax.legend()
    
    axs[0].set_ylabel("Tasa de Error de Bit ($P_b$)")
    
    fig_c.suptitle(
        "Comparación entre M-QAM y M-FSK",
        fontsize=14
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{output_dir}/punto_c_comparativa_modulaciones.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------
    # PUNTO D y E: Comparación Con vs Sin Codificación (Gráficos Separados)
    # -----------------------------------------------------------------
    sch_code = "M-QAM"
    valores_m = [4, 16]
    
    for m_val in valores_m:
        # Filtrar datos para el M actual
        df_sin = dfs_sin[sch_code][dfs_sin[sch_code]["M"] == m_val].sort_values("ebn0_db")
        df_con = dfs_con[sch_code][dfs_con[sch_code]["M"] == m_val].sort_values("ebn0_db")

        # -------------------------------------------------------------
        # Gráfico Punto D: Pe (Un gráfico por cada M)
        # -------------------------------------------------------------
        fig_d, ax_d = plt.subplots(figsize=(8, 6))
        ax_d.semilogy(df_sin["ebn0_db"], df_sin["sim_symb"], marker="o", linestyle="-", color="firebrick", label="Sin Codificación")
        ax_d.semilogy(df_con["ebn0_db"], df_con["sim_symb"], marker="D", linestyle="--", color="navy", label="CON Codificación (Golay 24,12)")
        
        ax_d.set_xlabel("$E_b/N_0$ (dB)")
        ax_d.set_ylabel("Probabilidad de Error de Símbolo ($P_e$)")
        ax_d.set_title(f"Código de Canal en $P_e$ ({sch_code} M={m_val})")
        ax_d.grid(True, which="both", alpha=0.4)
        ax_d.legend(loc="lower left")
        ax_d.set_ylim(1e-6, 1.1)
        plt.tight_layout()
        
        # Guarda con el valor de M en el nombre del archivo
        plt.savefig(f"{output_dir}/punto_d_coding_comparison_Pe_M{m_val}.png", dpi=150)
        plt.close()

        # -------------------------------------------------------------
        # Gráfico Punto E: Pb (Un gráfico por cada M)
        # -------------------------------------------------------------
        fig_e, ax_e = plt.subplots(figsize=(8, 6))
        ax_e.semilogy(df_sin["ebn0_db"], df_sin["sim_bit"], marker="s", linestyle="-", color="firebrick", label="Sin Codificación")
        ax_e.semilogy(df_con["ebn0_db"], df_con["sim_bit"], marker="^", linestyle="--", color="navy", label="CON Codificación (Golay 24,12)")
        
        ax_e.set_xlabel("$E_b/N_0$ (dB)")
        ax_e.set_ylabel("Probabilidad de Error de Bit ($P_b$)")
        ax_e.set_title(f"Ganancia de Codificación en $P_b$ ({sch_code} M={m_val})")
        ax_e.grid(True, which="both", alpha=0.4)
        ax_e.legend(loc="lower left")
        ax_e.set_ylim(1e-6, 1.1)
        plt.tight_layout()
        
        # Guarda con el valor de M en el nombre del archivo
        plt.savefig(f"{output_dir}/punto_e_coding_comparison_Pb_M{m_val}.png", dpi=150)
        plt.close()

    