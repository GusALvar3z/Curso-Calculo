#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROYECTO EMPRENDEDOR: OPTICLOUD
Plataforma de optimización de costos y recursos en servidores cloud para MYPES
mediante modelos de cálculo multivariable y ecuaciones diferenciales en Python.

Curso: Cálculo Integral y Ecuaciones Diferenciales
Universidad César Vallejo - Escuela Profesional de Ingeniería de Sistemas

Autores:
- Alvarez Osorio, Gustavo Dhair
- Ayala Rojas, Rubi Antuane
- Perez Portugal, Paul Alexander
- Quiñonez Mendoza, Nathaly Vanessa
- Vivar Gomez, Elias Jesus

Asesor:
- Dr. Rivera Crisostomo, Renee

Modo de ejecución:
1. Interfaz Web Interactiva (Streamlit):
   streamlit run opticloud_app.py
2. Modo Consola / Demostración y Pruebas Automáticas (sin requerir Streamlit):
   python3 opticloud_app.py
================================================================================
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


# ==============================================================================
# 1. MÓDULO DE INTEGRACIÓN DEFINIDA (CÁLCULO INTEGRAL CON SYMPY)
# ==============================================================================
@dataclass
class ResultadoIntegracion:
    expresion_tasa_simbolica: str
    expresion_tasa_latex: str
    primitiva_simbolica: str
    primitiva_latex: str
    trafico_diario_exacto_gb: float
    trafico_mensual_exacto_gb: float
    estimacion_media_simple_mensual_gb: float
    error_distorsion_porcentual: float
    costo_red_mensual_usd: float
    intervalo_horas: Tuple[float, float]


class ModuloIntegracionTrafico:
    """
    Calcula el volumen acumulado de tráfico y transferencia de datos a lo largo
    de intervalos continuos de tiempo mediante integrales definidas en SymPy,
    evitando la distorsión producida por una media aritmética simple.
    """

    def __init__(self) -> None:
        self.t = sp.Symbol("t", real=True, nonnegative=True)

    def construir_funcion_tasa(
        self,
        trafico_base_gb_h: float,
        trafico_pico_gb_h: float,
        periodo_horas: float = 24.0,
        perfil: str = "sinusoidal_diario",
    ) -> sp.Expr:
        """
        Modela la tasa instantánea de transferencia de datos R(t) en GB/hora:
        - 'sinusoidal_diario': R(t) = R_base + (R_pico - R_base) * sin^2(pi * t / T)
        - 'doble_pico_comercial': R(t) = R_base + (R_pico - R_base) * (0.65*sin^2(pi*t/T) + 0.35*sin^2(2*pi*t/T))
        - 'polinomico_suave': R(t) = R_base + (R_pico - R_base) * 16 * (t/T)^2 * (1 - t/T)^2
        """
        t = self.t
        T = sp.Rational(int(round(periodo_horas * 100)), 100)
        r_base = sp.Rational(int(round(trafico_base_gb_h * 1000)), 1000)
        amplitud = sp.Rational(int(round((trafico_pico_gb_h - trafico_base_gb_h) * 1000)), 1000)

        if perfil == "doble_pico_comercial":
            expr = r_base + amplitud * (
                sp.Rational(65, 100) * sp.sin(sp.pi * t / T) ** 2
                + sp.Rational(35, 100) * sp.sin(2 * sp.pi * t / T) ** 2
            )
        elif perfil == "polinomico_suave":
            u = t / T
            expr = r_base + amplitud * 16 * (u**2) * ((1 - u) ** 2)
        else:
            expr = r_base + amplitud * sp.sin(sp.pi * t / T) ** 2

        return sp.simplify(expr)

    def calcular_transferencia_acumulada(
        self,
        trafico_base_gb_h: float,
        trafico_pico_gb_h: float,
        t_inicio: float = 0.0,
        t_fin: float = 24.0,
        costo_gb_usd: float = 0.05,
        gb_libres_mes: float = 150.0,
        perfil: str = "sinusoidal_diario",
    ) -> Tuple[ResultadoIntegracion, sp.Expr]:
        """
        Resuelve simbólicamente V = int_{t_inicio}^{t_fin} R(t) dt con SymPy
        y cuantifica el costo variable mensual de transferencia de red.
        """
        t = self.t
        r_t = self.construir_funcion_tasa(
            trafico_base_gb_h=trafico_base_gb_h,
            trafico_pico_gb_h=trafico_pico_gb_h,
            periodo_horas=max(t_fin - t_inicio, 1.0),
            perfil=perfil,
        )

        # Antiderivada F(t) = \int R(t) dt
        primitiva = sp.integrate(r_t, t)

        # Integral definida exacta en [t_inicio, t_fin]
        integral_definida = sp.integrate(r_t, (t, t_inicio, t_fin))
        trafico_diario_gb = float(integral_definida.evalf())
        trafico_mensual_gb = trafico_diario_gb * 30.0

        # Comparación frente a una estimación empírica distorsionada
        # (las mypes suelen dimensionar asumiendo el 65% del pico sostenido)
        duracion = float(t_fin - t_inicio)
        media_ingenua_hora = 0.35 * trafico_base_gb_h + 0.65 * trafico_pico_gb_h
        estimacion_media_simple_mensual = media_ingenua_hora * duracion * 30.0

        error_distorsion = (
            abs(estimacion_media_simple_mensual - trafico_mensual_gb)
            / max(trafico_mensual_gb, 1e-6)
            * 100.0
        )

        gb_facturables = max(0.0, trafico_mensual_gb - gb_libres_mes)
        costo_red_usd = gb_facturables * costo_gb_usd

        resultado = ResultadoIntegracion(
            expresion_tasa_simbolica=str(r_t),
            expresion_tasa_latex=sp.latex(r_t),
            primitiva_simbolica=str(primitiva),
            primitiva_latex=sp.latex(primitiva),
            trafico_diario_exacto_gb=round(trafico_diario_gb, 4),
            trafico_mensual_exacto_gb=round(trafico_mensual_gb, 2),
            estimacion_media_simple_mensual_gb=round(estimacion_media_simple_mensual, 2),
            error_distorsion_porcentual=round(error_distorsion, 2),
            costo_red_mensual_usd=round(costo_red_usd, 2),
            intervalo_horas=(t_inicio, t_fin),
        )
        return resultado, r_t


# ==============================================================================
# 2. MÓDULO DE OPTIMIZACIÓN DE COSTOS Y RECURSOS (CÁLCULO MULTIVARIABLE)
# ==============================================================================
@dataclass
class ResultadoOptimizacion:
    funcion_costo_latex: str
    restriccion_latencia_latex: str
    costo_reducido_1var_latex: str
    derivada_primera_latex: str
    derivada_segunda_latex: str
    vcpu_optimo_continuo: float
    ram_optima_continua_gb: float
    vcpu_recomendado_discreto: int
    ram_recomendada_discreta_gb: int
    latencia_estimada_optima_ms: float
    costo_computo_actual_usd: float
    costo_computo_optimo_usd: float
    costo_total_actual_usd: float
    costo_total_optimo_usd: float
    ahorro_mensual_usd: float
    ahorro_mensual_pen: float
    porcentaje_ahorro: float


class ModuloOptimizacionCostos:
    """
    Formula la función de costo multivariable C(x, y) dependiente de vCPU (x)
    y memoria RAM en GB (y). Mediante el análisis de derivadas simbólicas en SymPy
    identifica la configuración que minimiza el costo total sujeto a restricciones
    de latencia máxima admisible.
    """

    def __init__(self) -> None:
        self.x = sp.Symbol("x", real=True, positive=True)  # Cantidad de vCPU
        self.y = sp.Symbol("y", real=True, positive=True)  # Memoria RAM (GB)

    def optimizar_recursos(
        self,
        demanda_pico_req_s: float,
        latencia_max_ms: float,
        latencia_base_ms: float,
        coef_carga_latencia: float,
        costo_unitario_vcpu_mes_usd: float,
        costo_unitario_ram_gb_mes_usd: float,
        vcpu_actual: int,
        ram_actual_gb: int,
        costo_red_mensual_usd: float = 0.0,
        tipo_cambio_pen: float = 3.75,
    ) -> ResultadoOptimizacion:
        """
        Minimiza:
            C(x, y) = c_cpu * x + c_ram * y + C_red
        Sujeto a la restricción de latencia de respuesta:
            L(x, y) = L_base + (alpha * D_pico) / (x * y) <= L_max
        De donde la frontera activa de rendimiento exige capacidad conjunta mínima:
            x * y = K_min, con K_min = (alpha * D_pico) / (L_max - L_base)
        """
        x, y = self.x, self.y

        c_cpu = sp.Rational(int(round(costo_unitario_vcpu_mes_usd * 100)), 100)
        c_ram = sp.Rational(int(round(costo_unitario_ram_gb_mes_usd * 100)), 100)
        c_red = sp.Rational(int(round(costo_red_mensual_usd * 100)), 100)

        margen_latencia = max(latencia_max_ms - latencia_base_ms, 5.0)
        k_min_val = (coef_carga_latencia * demanda_pico_req_s) / margen_latencia
        K_min = sp.Rational(int(round(k_min_val * 1000)), 1000)

        # Función de costo multivariable C(x, y)
        costo_xy = c_cpu * x + c_ram * y + c_red

        # Expresión simbólica de latencia L(x, y)
        alpha_d = sp.Rational(int(round(coef_carga_latencia * demanda_pico_req_s * 100)), 100)
        l_base_sym = sp.Rational(int(round(latencia_base_ms * 100)), 100)
        latencia_xy = l_base_sym + alpha_d / (x * y)

        # Sustitución de la frontera activa y(x) = K_min / x en C(x, y)
        y_en_funcion_de_x = K_min / x
        costo_1var = sp.simplify(costo_xy.subs(y, y_en_funcion_de_x))

        # Derivada primera dC/dx y segunda derivada d^2C/dx^2 (criterio de convexidad)
        dC_dx = sp.diff(costo_1var, x)
        d2C_dx2 = sp.diff(dC_dx, x)

        # Resolución simbólica de dC/dx = 0 para x > 0
        soluciones_x = sp.solve(sp.Eq(dC_dx, 0), x)
        x_opt_continuo = (
            float(soluciones_x[0].evalf())
            if soluciones_x
            else math.sqrt(float(K_min * c_ram / c_cpu))
        )
        y_opt_continuo = float((K_min / x_opt_continuo).evalf())

        # Búsqueda de configuración discreta óptima comercialmente viable en la nube
        # Evaluamos candidatos enteros que cumplan estrictamente L(x, y) <= L_max
        # y mantengan una arquitectura balanceada (entre 1 GB y 4 GB de RAM por vCPU)
        mejor_par = (
            max(1, int(math.ceil(x_opt_continuo))),
            max(2, int(math.ceil(y_opt_continuo))),
        )
        mejor_costo_computo = float("inf")

        x_min_busqueda = max(1, int(math.floor(x_opt_continuo * 0.5)))
        x_max_busqueda = max(vcpu_actual + 4, int(math.ceil(x_opt_continuo * 2.5)) + 2)

        for cand_x in range(x_min_busqueda, x_max_busqueda + 1):
            ram_min_req = math.ceil(k_min_val / cand_x)
            cand_y = max(2, ram_min_req)
            if cand_y < cand_x:
                cand_y = cand_x
            if cand_y > 4 * cand_x:
                continue

            lat_cand = latencia_base_ms + (coef_carga_latencia * demanda_pico_req_s) / (
                cand_x * cand_y
            )
            if lat_cand <= latencia_max_ms + 1e-6:
                costo_cand = (
                    costo_unitario_vcpu_mes_usd * cand_x
                    + costo_unitario_ram_gb_mes_usd * cand_y
                )
                if costo_cand < mejor_costo_computo:
                    mejor_costo_computo = costo_cand
                    mejor_par = (cand_x, cand_y)

        vcpu_rec, ram_rec = mejor_par
        latencia_optima_ms = latencia_base_ms + (
            coef_carga_latencia * demanda_pico_req_s
        ) / (vcpu_rec * ram_rec)

        costo_computo_actual = (
            costo_unitario_vcpu_mes_usd * vcpu_actual
            + costo_unitario_ram_gb_mes_usd * ram_actual_gb
        )
        costo_computo_optimo = (
            costo_unitario_vcpu_mes_usd * vcpu_rec
            + costo_unitario_ram_gb_mes_usd * ram_rec
        )

        costo_total_actual = costo_computo_actual + costo_red_mensual_usd
        costo_total_optimo = costo_computo_optimo + costo_red_mensual_usd

        ahorro_usd = max(0.0, costo_total_actual - costo_total_optimo)
        ahorro_pen = ahorro_usd * tipo_cambio_pen
        porcentaje_ahorro = (ahorro_usd / max(costo_total_actual, 1e-6)) * 100.0

        return ResultadoOptimizacion(
            funcion_costo_latex=sp.latex(costo_xy),
            restriccion_latencia_latex=sp.latex(latencia_xy)
            + r" \le "
            + f"{latencia_max_ms:.1f}",
            costo_reducido_1var_latex=sp.latex(costo_1var),
            derivada_primera_latex=sp.latex(dC_dx),
            derivada_segunda_latex=sp.latex(d2C_dx2),
            vcpu_optimo_continuo=round(x_opt_continuo, 3),
            ram_optima_continua_gb=round(y_opt_continuo, 3),
            vcpu_recomendado_discreto=vcpu_rec,
            ram_recomendada_discreta_gb=ram_rec,
            latencia_estimada_optima_ms=round(latencia_optima_ms, 2),
            costo_computo_actual_usd=round(costo_computo_actual, 2),
            costo_computo_optimo_usd=round(costo_computo_optimo, 2),
            costo_total_actual_usd=round(costo_total_actual, 2),
            costo_total_optimo_usd=round(costo_total_optimo, 2),
            ahorro_mensual_usd=round(ahorro_usd, 2),
            ahorro_mensual_pen=round(ahorro_pen, 2),
            porcentaje_ahorro=round(porcentaje_ahorro, 2),
        )


# ==============================================================================
# 3. MÓDULO DE ECUACIONES DIFERENCIALES DE PRIMER ORDEN (SATURACIÓN Y AUTOESCALADO)
# ==============================================================================
@dataclass
class ResultadoEcuacionDiferencial:
    edo_simbolica_latex: str
    solucion_particular_simbolica: str
    solucion_particular_latex: str
    capacidad_equilibrio_asintotico: float
    habra_saturacion: bool
    tiempo_autoescalado_seg: Optional[float]
    tiempo_saturacion_critica_seg: Optional[float]
    umbral_autoescalado_peticiones: float
    capacidad_max_cola: float


class ModuloEcuacionesDiferenciales:
    """
    Modela la dinámica de la cola de peticiones y la tasa de llenado del búfer
    de memoria mediante una Ecuación Diferencial Ordinaria (EDO) lineal de
    primer orden resuelta simbólicamente con sympy.dsolve:

        dQ(t)/dt + k * Q(t) = lambda_sobrecarga,   con Q(0) = Q_0

    Permite anticipar el instante exacto en que el sistema alcanza el umbral
    preventivo de autoescalado horizontal y el momento de saturación total.
    """

    def __init__(self) -> None:
        self.t = sp.Symbol("t", real=True, nonnegative=True)
        self.Q = sp.Function("Q")

    def analizar_saturacion_cola(
        self,
        tasa_llegada_rafaga_req_s: float,
        capacidad_base_req_s: float,
        coef_disipacion_k: float,
        cola_inicial_q0: float,
        capacidad_max_buffer: float,
        fraccion_alerta_autoescalado: float = 0.80,
    ) -> Tuple[ResultadoEcuacionDiferencial, sp.Expr]:
        """
        Resuelve el problema de valor inicial (PVI) de primer orden:
            dQ/dt + k * Q(t) = lambda_neto
        donde lambda_neto = max(tasa_llegada_rafaga - capacidad_base, 0).
        """
        t = self.t
        Q = self.Q

        flujo_neto_entrada = max(tasa_llegada_rafaga_req_s - capacidad_base_req_s, 1.0)
        k_sym = sp.Rational(int(round(coef_disipacion_k * 1000)), 1000)
        lam_sym = sp.Rational(int(round(flujo_neto_entrada * 100)), 100)
        q0_sym = sp.Rational(int(round(cola_inicial_q0 * 100)), 100)

        # Definición simbólica de la EDO de primer orden
        edo = sp.Eq(sp.diff(Q(t), t) + k_sym * Q(t), lam_sym)

        # Resolución analítica exacta mediante sympy.dsolve con condición inicial Q(0) = Q_0
        solucion_edo = sp.dsolve(edo, Q(t), ics={Q(0): q0_sym})
        q_t_expr = sp.simplify(solucion_edo.rhs)

        # Equilibrio asintótico cuando t -> infinito: Q_eq = lambda_neto / k
        q_equilibrio = float((lam_sym / k_sym).evalf())
        umbral_autoescalado = fraccion_alerta_autoescalado * capacidad_max_buffer

        # Cálculo exacto del tiempo de autoescalado Q(t_scale) = umbral_autoescalado
        tiempo_autoescalado: Optional[float] = None
        tiempo_saturacion: Optional[float] = None
        habra_saturacion = q_equilibrio > capacidad_max_buffer

        if q_equilibrio > umbral_autoescalado and umbral_autoescalado > cola_inicial_q0:
            t_scale_val = -(1.0 / coef_disipacion_k) * math.log(
                (q_equilibrio - umbral_autoescalado) / (q_equilibrio - cola_inicial_q0)
            )
            tiempo_autoescalado = round(max(0.0, t_scale_val), 2)

        if habra_saturacion and capacidad_max_buffer > cola_inicial_q0:
            t_sat_val = -(1.0 / coef_disipacion_k) * math.log(
                (q_equilibrio - capacidad_max_buffer) / (q_equilibrio - cola_inicial_q0)
            )
            tiempo_saturacion = round(max(0.0, t_sat_val), 2)

        resultado = ResultadoEcuacionDiferencial(
            edo_simbolica_latex=sp.latex(edo),
            solucion_particular_simbolica=str(q_t_expr),
            solucion_particular_latex=sp.latex(solucion_edo),
            capacidad_equilibrio_asintotico=round(q_equilibrio, 2),
            habra_saturacion=habra_saturacion,
            tiempo_autoescalado_seg=tiempo_autoescalado,
            tiempo_saturacion_critica_seg=tiempo_saturacion,
            umbral_autoescalado_peticiones=round(umbral_autoescalado, 2),
            capacidad_max_cola=round(capacidad_max_buffer, 2),
        )
        return resultado, q_t_expr


# ==============================================================================
# 4. MOTOR INTEGRADO Y BANCO DE PRUEBAS CON DATOS SINTÉTICOS (SECCIÓN 5 Y ANEXOS)
# ==============================================================================
ESCENARIOS_MYPES_SINTETICOS: List[Dict[str, object]] = [
    {
        "id": "MYPE-01",
        "nombre": "E-Commerce Textil Gamarra (Tienda Web)",
        "perfil_trafico": "doble_pico_comercial",
        "trafico_base_gb_h": 1.2,
        "trafico_pico_gb_h": 6.8,
        "demanda_pico_req_s": 180.0,
        "latencia_max_ms": 120.0,
        "latencia_base_ms": 25.0,
        "coef_carga_latencia": 24.0,
        "vcpu_actual": 8,
        "ram_actual_gb": 16,
        "costo_vcpu_usd": 18.0,
        "costo_ram_usd": 4.5,
        "rafaga_req_s": 310.0,
        "capacidad_base_req_s": 190.0,
        "coef_disipacion_k": 0.045,
        "cola_inicial_q0": 50.0,
        "buffer_max_req": 1500.0,
    },
    {
        "id": "MYPE-02",
        "nombre": "Sistema SaaS Facturación Electrónica SUNAT",
        "perfil_trafico": "sinusoidal_diario",
        "trafico_base_gb_h": 2.0,
        "trafico_pico_gb_h": 9.5,
        "demanda_pico_req_s": 260.0,
        "latencia_max_ms": 100.0,
        "latencia_base_ms": 20.0,
        "coef_carga_latencia": 26.0,
        "vcpu_actual": 12,
        "ram_actual_gb": 24,
        "costo_vcpu_usd": 18.0,
        "costo_ram_usd": 4.5,
        "rafaga_req_s": 420.0,
        "capacidad_base_req_s": 270.0,
        "coef_disipacion_k": 0.050,
        "cola_inicial_q0": 80.0,
        "buffer_max_req": 2000.0,
    },
    {
        "id": "MYPE-03",
        "nombre": "App Delivery Gastronómico Lima Norte",
        "perfil_trafico": "doble_pico_comercial",
        "trafico_base_gb_h": 0.8,
        "trafico_pico_gb_h": 5.2,
        "demanda_pico_req_s": 140.0,
        "latencia_max_ms": 130.0,
        "latencia_base_ms": 30.0,
        "coef_carga_latencia": 22.0,
        "vcpu_actual": 6,
        "ram_actual_gb": 12,
        "costo_vcpu_usd": 18.0,
        "costo_ram_usd": 4.5,
        "rafaga_req_s": 250.0,
        "capacidad_base_req_s": 150.0,
        "coef_disipacion_k": 0.040,
        "cola_inicial_q0": 30.0,
        "buffer_max_req": 1200.0,
    },
    {
        "id": "MYPE-04",
        "nombre": "Plataforma EdTech Academia Preuniversitaria",
        "perfil_trafico": "polinomico_suave",
        "trafico_base_gb_h": 3.0,
        "trafico_pico_gb_h": 12.0,
        "demanda_pico_req_s": 320.0,
        "latencia_max_ms": 110.0,
        "latencia_base_ms": 22.0,
        "coef_carga_latencia": 28.0,
        "vcpu_actual": 16,
        "ram_actual_gb": 32,
        "costo_vcpu_usd": 18.0,
        "costo_ram_usd": 4.5,
        "rafaga_req_s": 520.0,
        "capacidad_base_req_s": 330.0,
        "coef_disipacion_k": 0.055,
        "cola_inicial_q0": 100.0,
        "buffer_max_req": 2500.0,
    },
    {
        "id": "MYPE-05",
        "nombre": "Microservicio ERP Logístico y Almacenes",
        "perfil_trafico": "sinusoidal_diario",
        "trafico_base_gb_h": 1.0,
        "trafico_pico_gb_h": 4.4,
        "demanda_pico_req_s": 110.0,
        "latencia_max_ms": 140.0,
        "latencia_base_ms": 28.0,
        "coef_carga_latencia": 20.0,
        "vcpu_actual": 4,
        "ram_actual_gb": 12,
        "costo_vcpu_usd": 18.0,
        "costo_ram_usd": 4.5,
        "rafaga_req_s": 195.0,
        "capacidad_base_req_s": 120.0,
        "coef_disipacion_k": 0.038,
        "cola_inicial_q0": 25.0,
        "buffer_max_req": 1000.0,
    },
]


def ejecutar_simulacion_completa(parametros: Dict[str, object]) -> Dict[str, object]:
    """
    Integra el flujo de datos entre los tres módulos matemáticos de OptiCloud:
    1. Módulo de Integración -> Volumen acumulado de red y costo variable.
    2. Módulo de Optimización -> Dimensionamiento óptimo de vCPU y RAM y ahorro mensual.
    3. Módulo de Ecuaciones Diferenciales -> Tiempo de autoescalado y saturación de cola.
    """
    mod_int = ModuloIntegracionTrafico()
    mod_opt = ModuloOptimizacionCostos()
    mod_edo = ModuloEcuacionesDiferenciales()

    res_int, expr_r_t = mod_int.calcular_transferencia_acumulada(
        trafico_base_gb_h=float(parametros["trafico_base_gb_h"]),
        trafico_pico_gb_h=float(parametros["trafico_pico_gb_h"]),
        t_inicio=0.0,
        t_fin=24.0,
        costo_gb_usd=float(parametros.get("costo_gb_usd", 0.05)),
        gb_libres_mes=float(parametros.get("gb_libres_mes", 150.0)),
        perfil=str(parametros.get("perfil_trafico", "sinusoidal_diario")),
    )

    res_opt = mod_opt.optimizar_recursos(
        demanda_pico_req_s=float(parametros["demanda_pico_req_s"]),
        latencia_max_ms=float(parametros["latencia_max_ms"]),
        latencia_base_ms=float(parametros["latencia_base_ms"]),
        coef_carga_latencia=float(parametros["coef_carga_latencia"]),
        costo_unitario_vcpu_mes_usd=float(parametros["costo_vcpu_usd"]),
        costo_unitario_ram_gb_mes_usd=float(parametros["costo_ram_usd"]),
        vcpu_actual=int(parametros["vcpu_actual"]),
        ram_actual_gb=int(parametros["ram_actual_gb"]),
        costo_red_mensual_usd=res_int.costo_red_mensual_usd,
        tipo_cambio_pen=float(parametros.get("tipo_cambio_pen", 3.75)),
    )

    res_edo, expr_q_t = mod_edo.analizar_saturacion_cola(
        tasa_llegada_rafaga_req_s=float(parametros["rafaga_req_s"]),
        capacidad_base_req_s=float(parametros["capacidad_base_req_s"]),
        coef_disipacion_k=float(parametros["coef_disipacion_k"]),
        cola_inicial_q0=float(parametros["cola_inicial_q0"]),
        capacidad_max_buffer=float(parametros["buffer_max_req"]),
        fraccion_alerta_autoescalado=float(parametros.get("fraccion_alerta", 0.80)),
    )

    return {
        "integracion": res_int,
        "expr_r_t": expr_r_t,
        "optimizacion": res_opt,
        "edo": res_edo,
        "expr_q_t": expr_q_t,
    }


def generar_tabla_pruebas_sinteticas() -> pd.DataFrame:
    """
    Ejecuta las pruebas internas con datos sintéticos de los 5 escenarios MYPE
    y retorna un DataFrame consolidado para validación y exportación.
    """
    filas = []
    for esc in ESCENARIOS_MYPES_SINTETICOS:
        out = ejecutar_simulacion_completa(esc)
        r_int: ResultadoIntegracion = out["integracion"]
        r_opt: ResultadoOptimizacion = out["optimizacion"]
        r_edo: ResultadoEcuacionDiferencial = out["edo"]

        filas.append(
            {
                "ID": esc["id"],
                "Escenario MYPE": esc["nombre"],
                "Tráfico Mensual Integral (GB)": r_int.trafico_mensual_exacto_gb,
                "Config. Actual (vCPU / RAM)": f"{esc['vcpu_actual']} vCPU / {esc['ram_actual_gb']} GB",
                "Config. Óptima OptiCloud": (
                    f"{r_opt.vcpu_recomendado_discreto} vCPU / {r_opt.ram_recomendada_discreta_gb} GB"
                ),
                "Latencia Óptima (ms)": f"{r_opt.latencia_estimada_optima_ms:.1f} / {esc['latencia_max_ms']:.0f}",
                "Costo Actual (USD/mes)": r_opt.costo_total_actual_usd,
                "Costo Óptimo (USD/mes)": r_opt.costo_total_optimo_usd,
                "Ahorro Mensual (USD)": r_opt.ahorro_mensual_usd,
                "Ahorro Mensual (S/)": r_opt.ahorro_mensual_pen,
                "Reducción Gasto (%)": r_opt.porcentaje_ahorro,
                "Alerta Autoescalado (s)": r_edo.tiempo_autoescalado_seg,
                "Tiempo Saturación EDO (s)": r_edo.tiempo_saturacion_critica_seg,
            }
        )
    return pd.DataFrame(filas)


# ==============================================================================
# 5. GENERACIÓN DE GRÁFICAS DE LOS TRES MÓDULOS MATEMÁTICOS
# ==============================================================================
def crear_figura_panel_opticloud(
    parametros: Dict[str, object], resultados: Dict[str, object]
) -> plt.Figure:
    """
    Genera una figura con 3 paneles analíticos:
    1. Integral Definida del Tráfico Acumulado R(t) en 24 horas.
    2. Curva de Optimización de Costo C(x) bajo restricción de latencia.
    3. Dinámica de Saturación del Búfer Q(t) mediante EDO de primer orden.
    """
    res_int: ResultadoIntegracion = resultados["integracion"]
    expr_r_t: sp.Expr = resultados["expr_r_t"]
    res_opt: ResultadoOptimizacion = resultados["optimizacion"]
    res_edo: ResultadoEcuacionDiferencial = resultados["edo"]
    expr_q_t: sp.Expr = resultados["expr_q_t"]

    t_sym = sp.Symbol("t", real=True, nonnegative=True)
    f_r_num = sp.lambdify(t_sym, expr_r_t, modules=["numpy"])
    f_q_num = sp.lambdify(t_sym, expr_q_t, modules=["numpy"])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), dpi=130)
    fig.patch.set_facecolor("#F8FAFC")

    # --- PANEL 1: Integración Definida de Tráfico ---
    ax1 = axes[0]
    t_vals = np.linspace(0, 24, 240)
    r_vals = np.array([float(f_r_num(tv)) for tv in t_vals])
    ax1.plot(t_vals, r_vals, color="#0284C7", linewidth=2.5, label="Tasa instantánea R(t)")
    ax1.fill_between(
        t_vals,
        0,
        r_vals,
        color="#38BDF8",
        alpha=0.28,
        label=f"Integral diaria: {res_int.trafico_diario_exacto_gb:.1f} GB",
    )
    media_ingenua = (
        0.35 * float(parametros["trafico_base_gb_h"])
        + 0.65 * float(parametros["trafico_pico_gb_h"])
    )
    ax1.axhline(
        media_ingenua,
        color="#DC2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Estimación empírica ({media_ingenua:.2f} GB/h)",
    )
    ax1.set_title("1. Integración Definida: Tráfico R(t)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Tiempo t (Horas del día)")
    ax1.set_ylabel("Tasa de transferencia (GB/h)")
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, max(r_vals) * 1.2)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # --- PANEL 2: Optimización Multivariable de Costos ---
    ax2 = axes[1]
    margen_lat = max(
        float(parametros["latencia_max_ms"]) - float(parametros["latencia_base_ms"]), 5.0
    )
    k_min_val = (
        float(parametros["coef_carga_latencia"]) * float(parametros["demanda_pico_req_s"])
    ) / margen_lat

    x_grid = np.linspace(1.5, max(float(parametros["vcpu_actual"]) * 1.25, 10.0), 200)
    c_cpu = float(parametros["costo_vcpu_usd"])
    c_ram = float(parametros["costo_ram_usd"])
    c_red = res_int.costo_red_mensual_usd
    costo_curva = c_cpu * x_grid + c_ram * (k_min_val / x_grid) + c_red

    ax2.plot(
        x_grid,
        costo_curva,
        color="#4F46E5",
        linewidth=2.5,
        label="Costo C(x, y(x)) en frontera L_max",
    )
    costo_opt_cont = (
        c_cpu * res_opt.vcpu_optimo_continuo
        + c_ram * res_opt.ram_optima_continua_gb
        + c_red
    )
    ax2.scatter(
        [res_opt.vcpu_optimo_continuo],
        [costo_opt_cont],
        color="#16A34A",
        s=85,
        zorder=5,
        label=f"Mínimo analítico ({res_opt.vcpu_optimo_continuo:.2f} vCPU)",
    )
    ax2.scatter(
        [res_opt.vcpu_recomendado_discreto],
        [res_opt.costo_total_optimo_usd],
        color="#059669",
        marker="D",
        s=80,
        zorder=6,
        label=f"Óptimo discreto ({res_opt.vcpu_recomendado_discreto} vCPU): ${res_opt.costo_total_optimo_usd:.0f}",
    )
    ax2.scatter(
        [float(parametros["vcpu_actual"])],
        [res_opt.costo_total_actual_usd],
        color="#DC2626",
        marker="X",
        s=95,
        zorder=6,
        label=f"Sobredimensionado ({parametros['vcpu_actual']} vCPU): ${res_opt.costo_total_actual_usd:.0f}",
    )
    ax2.set_title("2. Optimización: Costo vs. vCPU (dC/dx = 0)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Procesadores virtuales x (vCPU)")
    ax2.set_ylabel("Costo Operativo Mensual (USD)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    # --- PANEL 3: EDO de Primer Orden para Saturación de Cola ---
    ax3 = axes[2]
    t_max_plot = (
        res_edo.tiempo_saturacion_critica_seg * 1.25
        if res_edo.tiempo_saturacion_critica_seg
        else 90.0
    )
    t_edo = np.linspace(0, max(t_max_plot, 30.0), 240)
    q_vals = np.array([float(f_q_num(tv)) for tv in t_edo])

    ax3.plot(t_edo, q_vals, color="#7C3AED", linewidth=2.5, label="Cola de peticiones Q(t)")
    ax3.axhline(
        res_edo.capacidad_max_cola,
        color="#DC2626",
        linestyle="-",
        linewidth=1.8,
        label=f"Saturación Q_max ({res_edo.capacidad_max_cola:.0f} req)",
    )
    ax3.axhline(
        res_edo.umbral_autoescalado_peticiones,
        color="#D97706",
        linestyle="--",
        linewidth=1.6,
        label=f"Autoescalado 80% ({res_edo.umbral_autoescalado_peticiones:.0f} req)",
    )
    if res_edo.tiempo_autoescalado_seg is not None:
        ax3.axvline(
            res_edo.tiempo_autoescalado_seg,
            color="#D97706",
            linestyle=":",
            linewidth=1.5,
        )
        ax3.scatter(
            [res_edo.tiempo_autoescalado_seg],
            [res_edo.umbral_autoescalado_peticiones],
            color="#D97706",
            s=75,
            zorder=5,
            label=f"Activar Auto-Scaling: t = {res_edo.tiempo_autoescalado_seg:.1f} s",
        )
    if res_edo.tiempo_saturacion_critica_seg is not None:
        ax3.scatter(
            [res_edo.tiempo_saturacion_critica_seg],
            [res_edo.capacidad_max_cola],
            color="#DC2626",
            s=75,
            zorder=5,
            label=f"Colapso de cola: t = {res_edo.tiempo_saturacion_critica_seg:.1f} s",
        )

    ax3.set_title("3. EDO 1er Orden: Dinámica de Cola Q(t)", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Tiempo desde inicio de ráfaga t (segundos)")
    ax3.set_ylabel("Peticiones acumuladas en búfer Q(t)")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=8.0)

    fig.tight_layout()
    return fig


# ==============================================================================
# 6. INTERFAZ WEB INTERACTIVA EN STREAMLIT
# ==============================================================================
def ejecutar_interfaz_streamlit() -> None:
    """
    Construye y despliega la interfaz web interactiva en Streamlit para OptiCloud.
    """
    import streamlit as st

    st.set_page_config(
        page_title="OptiCloud | Optimización Cloud para MYPES",
        page_icon="☁️",
        layout="wide",
    )

    st.title("☁️ OptiCloud: Plataforma de Optimización de Costos y Recursos Cloud para MYPES")
    st.markdown(
        "**Modelos de Cálculo Integral, Optimización Multivariable y Ecuaciones Diferenciales de 1er Orden con SymPy**"
    )

    # Selector rápido de escenario preconfigurado o personalizado en barra lateral
    st.sidebar.header("⚙️ Parámetros del Servidor Cloud")
    nombres_escenarios = [f"{e['id']} - {e['nombre']}" for e in ESCENARIOS_MYPES_SINTETICOS] + [
        "Personalizado"
    ]
    seleccion = st.sidebar.selectbox("Plantilla de carga MYPE:", nombres_escenarios, index=0)

    if seleccion != "Personalizado":
        idx = nombres_escenarios.index(seleccion)
        base_cfg = dict(ESCENARIOS_MYPES_SINTETICOS[idx])
    else:
        base_cfg = dict(ESCENARIOS_MYPES_SINTETICOS[0])

    st.sidebar.subheader("1. Tráfico de Red (Integración)")
    perfil_trafico = st.sidebar.selectbox(
        "Perfil de tráfico diario R(t):",
        ["sinusoidal_diario", "doble_pico_comercial", "polinomico_suave"],
        index=["sinusoidal_diario", "doble_pico_comercial", "polinomico_suave"].index(
            str(base_cfg["perfil_trafico"])
        ),
    )
    trafico_base = st.sidebar.slider(
        "Tráfico valle/base (GB/hora):", 0.2, 10.0, float(base_cfg["trafico_base_gb_h"]), 0.1
    )
    trafico_pico = st.sidebar.slider(
        "Tráfico pico (GB/hora):",
        max(trafico_base + 0.5, 1.0),
        25.0,
        float(base_cfg["trafico_pico_gb_h"]),
        0.2,
    )

    st.sidebar.subheader("2. Recursos y Costos (Optimización)")
    vcpu_actual = st.sidebar.number_input(
        "vCPU contratados actualmente:", min_value=1, max_value=64, value=int(base_cfg["vcpu_actual"])
    )
    ram_actual = st.sidebar.number_input(
        "RAM contratada actualmente (GB):",
        min_value=2,
        max_value=256,
        value=int(base_cfg["ram_actual_gb"]),
    )
    demanda_pico = st.sidebar.slider(
        "Demanda pico concurrente (req/s):", 20.0, 800.0, float(base_cfg["demanda_pico_req_s"]), 10.0
    )
    latencia_max = st.sidebar.slider(
        "Latencia máxima admisible SLA (ms):", 50.0, 250.0, float(base_cfg["latencia_max_ms"]), 5.0
    )
    costo_vcpu = st.sidebar.number_input(
        "Costo unitario vCPU (USD/mes):",
        min_value=2.0,
        max_value=60.0,
        value=float(base_cfg["costo_vcpu_usd"]),
        step=0.5,
    )
    costo_ram = st.sidebar.number_input(
        "Costo unitario GB RAM (USD/mes):",
        min_value=1.0,
        max_value=25.0,
        value=float(base_cfg["costo_ram_usd"]),
        step=0.5,
    )

    st.sidebar.subheader("3. Dinámica de Cola y Búfer (EDO)")
    rafaga_req = st.sidebar.slider(
        "Tasa de entrada en ráfaga (req/s):",
        float(demanda_pico),
        1000.0,
        float(base_cfg["rafaga_req_s"]),
        10.0,
    )
    buffer_max = st.sidebar.slider(
        "Capacidad máxima del búfer Q_max (peticiones):",
        500.0,
        5000.0,
        float(base_cfg["buffer_max_req"]),
        100.0,
    )
    coef_k = st.sidebar.slider(
        "Coeficiente de disipación k (1/s):",
        0.010,
        0.150,
        float(base_cfg["coef_disipacion_k"]),
        0.005,
    )

    parametros_usuario: Dict[str, object] = {
        "perfil_trafico": perfil_trafico,
        "trafico_base_gb_h": trafico_base,
        "trafico_pico_gb_h": trafico_pico,
        "demanda_pico_req_s": demanda_pico,
        "latencia_max_ms": latencia_max,
        "latencia_base_ms": float(base_cfg["latencia_base_ms"]),
        "coef_carga_latencia": float(base_cfg["coef_carga_latencia"]),
        "vcpu_actual": int(vcpu_actual),
        "ram_actual_gb": int(ram_actual),
        "costo_vcpu_usd": float(costo_vcpu),
        "costo_ram_usd": float(costo_ram),
        "rafaga_req_s": float(rafaga_req),
        "capacidad_base_req_s": float(base_cfg["capacidad_base_req_s"]),
        "coef_disipacion_k": float(coef_k),
        "cola_inicial_q0": float(base_cfg["cola_inicial_q0"]),
        "buffer_max_req": float(buffer_max),
    }

    resultados = ejecutar_simulacion_completa(parametros_usuario)
    r_int: ResultadoIntegracion = resultados["integracion"]
    r_opt: ResultadoOptimizacion = resultados["optimizacion"]
    r_edo: ResultadoEcuacionDiferencial = resultados["edo"]

    # Métricas superiores ejecutivas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Configuración Recomendada",
        f"{r_opt.vcpu_recomendado_discreto} vCPU / {r_opt.ram_recomendada_discreta_gb} GB",
        delta=f"Actual: {vcpu_actual} vCPU / {ram_actual} GB",
        delta_color="off",
    )
    col2.metric(
        "Costo Mensual Optimizado",
        f"${r_opt.costo_total_optimo_usd:.2f} USD",
        delta=f"-${r_opt.ahorro_mensual_usd:.2f} USD (-{r_opt.porcentaje_ahorro:.1f}%)",
        delta_color="normal",
    )
    col3.metric(
        "Ahorro Mensual en Soles",
        f"S/ {r_opt.ahorro_mensual_pen:.2f}",
        delta=f"Tráfico Exacto: {r_int.trafico_mensual_exacto_gb:.0f} GB/mes",
    )
    col4.metric(
        "Instante de Autoescalado (EDO)",
        f"{r_edo.tiempo_autoescalado_seg:.1f} seg"
        if r_edo.tiempo_autoescalado_seg is not None
        else "Estable (Sin saturación)",
        delta=(
            f"Colapso en: {r_edo.tiempo_saturacion_critica_seg:.1f} s"
            if r_edo.tiempo_saturacion_critica_seg is not None
            else "Capacidad suficiente"
        ),
        delta_color="inverse",
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Panel Gráfico y Simulación",
            "🧮 Modelo Matemático Simbólico (SymPy)",
            "🧪 Pruebas con Datos Sintéticos (Anexo)",
            "📥 Exportar Configuración y Flujo",
        ]
    )

    with tab1:
        fig = crear_figura_panel_opticloud(parametros_usuario, resultados)
        st.pyplot(fig)

    with tab2:
        st.subheader("1. Integración Definida del Volumen de Tráfico (SymPy)")
        st.markdown("Función de tasa de transferencia instantánea $R(t)$ en GB/hora:")
        st.latex(r"R(t) = " + r_int.expresion_tasa_latex)
        st.markdown("Antiderivada simbólica $F(t) = \\int R(t)\\,dt$:")
        st.latex(r"F(t) = " + r_int.primitiva_latex)
        st.markdown(
            f"**Volumen diario exacto:** $\\int_{{0}}^{{24}} R(t)\\,dt = {r_int.trafico_diario_exacto_gb:.2f}\\text{{ GB/día}}$ "
            f"($30\\text{{ días}} = {r_int.trafico_mensual_exacto_gb:.2f}\\text{{ GB/mes}}$). "
            f"Se corrige una distorsión de **{r_int.error_distorsion_porcentual:.2f}%** frente al cálculo por media simple."
        )

        st.subheader("2. Optimización Multivariable de Costos bajo Restricción de Latencia")
        st.latex(r"C(x, y) = " + r_opt.funcion_costo_latex)
        st.latex(r"L(x, y) = " + r_opt.restriccion_latencia_latex)
        st.markdown("Sustituyendo la frontera activa de rendimiento $y(x)$ en $C(x, y)$ y derivando con SymPy:")
        st.latex(r"C(x) = " + r_opt.costo_reducido_1var_latex)
        st.latex(
            r"\frac{dC}{dx} = "
            + r_opt.derivada_primera_latex
            + r" = 0 \quad ; \quad \frac{d^2C}{dx^2} = "
            + r_opt.derivada_segunda_latex
            + r" > 0"
        )
        st.markdown(
            f"**Punto crítico continuo:** $x^* = {r_opt.vcpu_optimo_continuo:.2f}\\text{{ vCPU}}$, "
            f"$y^* = {r_opt.ram_optima_continua_gb:.2f}\\text{{ GB RAM}}$. "
            f"**Configuración discreta recomendada:** `{r_opt.vcpu_recomendado_discreto} vCPU` y `{r_opt.ram_recomendada_discreta_gb} GB RAM` "
            f"(Latencia esperada: `{r_opt.latencia_estimada_optima_ms:.1f} ms`)."
        )

        st.subheader("3. Ecuación Diferencial Ordinaria de Primer Orden (Dinámica de Búfer)")
        st.latex(r_edo.edo_simbolica_latex)
        st.markdown("Solución analítica exacta obtenida con `sympy.dsolve` para $Q(0) = Q_0$:")
        st.latex(r_edo.solucion_particular_latex)

    with tab3:
        st.subheader("Resultados de Pruebas Internas con Datos Sintéticos (5 Escenarios MYPE)")
        df_sintetico = generar_tabla_pruebas_sinteticas()
        st.dataframe(df_sintetico, use_container_width=True)

    with tab4:
        st.subheader("Exportación de Recomendación y Configuración Cloud")
        reporte_exportable = {
            "plataforma": "OptiCloud v1.0",
            "parametros_entrada": parametros_usuario,
            "integracion_trafico": asdict(r_int),
            "optimizacion_recursos": asdict(r_opt),
            "prediccion_saturacion_edo": asdict(r_edo),
        }
        json_str = json.dumps(reporte_exportable, indent=2, ensure_ascii=False)
        st.download_button(
            label="⬇️ Descargar Reporte de Dimensionamiento (JSON)",
            data=json_str,
            file_name="opticloud_recomendacion.json",
            mime="application/json",
        )
        st.code(json_str, language="json")


# ==============================================================================
# 7. EJECUCIÓN EN MODO CONSOLA / DEMOSTRACIÓN AUTOMÁTICA DE ANEXOS
# ==============================================================================
def ejecutar_modo_consola(directorio_salida: Optional[str] = None) -> None:
    """
    Ejecuta todos los módulos simbólicos en consola, imprime el desarrollo
    matemático de SymPy, genera la tabla de pruebas sintéticas y guarda
    las gráficas de evidencia.
    """
    print("=" * 84)
    print(" OPTICLOUD - DEMOSTRACIÓN DEL PROTOTIPO FUNCIONAL (SYMPY + MODELOS MATEMÁTICOS)")
    print("=" * 84)

    escenario_principal = ESCENARIOS_MYPES_SINTETICOS[0]
    resultados = ejecutar_simulacion_completa(escenario_principal)
    r_int: ResultadoIntegracion = resultados["integracion"]
    r_opt: ResultadoOptimizacion = resultados["optimizacion"]
    r_edo: ResultadoEcuacionDiferencial = resultados["edo"]

    print(f"\n[CASO DEMOSTRATIVO]: {escenario_principal['id']} - {escenario_principal['nombre']}")
    print("-" * 84)
    print("1) MÓDULO DE INTEGRACIÓN DEFINIDA (CÁLCULO INTEGRAL CON SYMPY)")
    print(f"   - Función de tasa R(t) [GB/h]   : R(t) = {r_int.expresion_tasa_simbolica}")
    print(f"   - Antiderivada simbólica F(t)   : F(t) = {r_int.primitiva_simbolica}")
    print(f"   - Integral definida [0, 24h]    : {r_int.trafico_diario_exacto_gb:.2f} GB/día")
    print(f"   - Tráfico acumulado mensual     : {r_int.trafico_mensual_exacto_gb:.2f} GB/mes")
    print(f"   - Estimación media simple       : {r_int.estimacion_media_simple_mensual_gb:.2f} GB/mes")
    print(f"   - Distorsión evitada            : {r_int.error_distorsion_porcentual:.2f}%")
    print(f"   - Costo variable de red         : ${r_int.costo_red_mensual_usd:.2f} USD/mes")

    print("\n2) MÓDULO DE OPTIMIZACIÓN MULTIVARIABLE (DERIVADAS CON SYMPY)")
    print(f"   - Función costo C(x) en frontera: C(x) = {r_opt.costo_reducido_1var_latex}")
    print(f"   - Primera derivada dC/dx = 0    : {r_opt.derivada_primera_latex} = 0")
    print(f"   - Segunda derivada d2C/dx2 > 0  : {r_opt.derivada_segunda_latex} > 0 (Mínimo global)")
    print(
        f"   - Óptimo analítico continuo     : x* = {r_opt.vcpu_optimo_continuo:.2f} vCPU, "
        f"y* = {r_opt.ram_optima_continua_gb:.2f} GB RAM"
    )
    print(
        f"   - Configuración actual (MYPE)   : {escenario_principal['vcpu_actual']} vCPU / "
        f"{escenario_principal['ram_actual_gb']} GB RAM -> ${r_opt.costo_total_actual_usd:.2f} USD/mes"
    )
    print(
        f"   - Configuración recomendada     : {r_opt.vcpu_recomendado_discreto} vCPU / "
        f"{r_opt.ram_recomendada_discreta_gb} GB RAM -> ${r_opt.costo_total_optimo_usd:.2f} USD/mes"
    )
    print(
        f"   - Ahorro mensual logrado        : ${r_opt.ahorro_mensual_usd:.2f} USD/mes "
        f"(S/ {r_opt.ahorro_mensual_pen:.2f} PEN) -> Reducción del {r_opt.porcentaje_ahorro:.2f}%"
    )

    print("\n3) MÓDULO DE ECUACIONES DIFERENCIALES DE 1ER ORDEN (SYMPY DSOLVE)")
    print(f"   - EDO lineal de 1er orden       : {r_edo.edo_simbolica_latex}")
    print(f"   - Solución analítica Q(t)       : Q(t) = {r_edo.solucion_particular_simbolica}")
    print(f"   - Umbral de autoescalado (80%)  : {r_edo.umbral_autoescalado_peticiones:.0f} peticiones")
    print(f"   - Tiempo p/ activar autoescalado: t = {r_edo.tiempo_autoescalado_seg} segundos")
    print(f"   - Tiempo de saturación total    : t = {r_edo.tiempo_saturacion_critica_seg} segundos")

    print("\n4) TABLA DE PRUEBAS INTERNAS CON DATOS SINTÉTICOS DE TRÁFICO (SECCIÓN 5 Y 8)")
    print("-" * 84)
    df_pruebas = generar_tabla_pruebas_sinteticas()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_pruebas.to_string(index=False))
    print("=" * 84)

    if directorio_salida:
        os.makedirs(directorio_salida, exist_ok=True)
        ruta_fig = os.path.join(directorio_salida, "opticloud_graficas_evidencia.png")
        fig = crear_figura_panel_opticloud(escenario_principal, resultados)
        fig.savefig(ruta_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Gráfica de evidencia exportada en: {ruta_fig}")


def _ejecutando_en_streamlit() -> bool:
    """Detecta si el script fue invocado mediante `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _ejecutando_en_streamlit():
        ejecutar_interfaz_streamlit()
    else:
        ejecutar_modo_consola()
