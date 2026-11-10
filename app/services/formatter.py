import numpy as np
from typing import List, Dict, Optional, Tuple
from ..schemas.simplex_schema import SimplexStep

def clean_float(val: float, precision: int = 4) -> float:
    """Evita ceros negativos y redondea a la precisión solicitada."""
    if abs(val) < 1e-9:
        return 0.0
    return round(float(val), precision)

def format_row_operation(target_row: int, factor: float, pivot_row: int, is_pivot_row: bool = False) -> str:
    """Genera la notación formal de la operación elemental entre filas."""
    r_target = f"R_{target_row + 1}"
    r_pivot = f"R_{pivot_row + 1}"
    
    if is_pivot_row:
        return f"{r_target}' = {r_target} / {clean_float(factor)}"
    
    clean_fac = clean_float(factor)
    if clean_fac == 0.0:
        return f"{r_target}' = {r_target}"
    elif clean_fac > 0:
        return f"{r_target}' = {r_target} - ({clean_fac}) * {r_pivot}'"
    else:
        return f"{r_target}' = {r_target} + ({abs(clean_fac)}) * {r_pivot}'"

def create_step_snapshot(
    step_number: int,
    phase: int,
    description: str,
    headers: List[str],
    tableau_matrix: np.ndarray,
    basic_vars: List[str],
    cb: List[float],
    cj: List[float],
    zj: List[float],
    zj_cj: List[float],
    objective_name: str = "Z",
    indicator_label: str = "Z_j - C_j",
    pivot_row: Optional[int] = None,
    pivot_col: Optional[int] = None,
    entering_var: Optional[str] = None,
    leaving_var: Optional[str] = None,
    ratios: Optional[List[Optional[float]]] = None,
    row_operations: Optional[List[str]] = None,
    precision: int = 4
) -> SimplexStep:
    """Construye un snapshot estructurado del tablero para el frontend."""
    # Redondear la matriz
    matrix_clean = [
        [clean_float(cell, precision) for cell in row]
        for row in tableau_matrix
    ]
    
    cb_clean = [clean_float(c, precision) for c in cb]
    cj_clean = [clean_float(c, precision) for c in cj]
    zj_clean = [clean_float(z, precision) for z in zj]
    zj_cj_clean = [clean_float(z_c, precision) for z_c in zj_cj]
    
    ratios_clean = None
    if ratios is not None:
        ratios_clean = [clean_float(r, precision) if r is not None else None for r in ratios]
        
    pivot_elem = None
    if pivot_row is not None and pivot_col is not None:
        pivot_elem = clean_float(tableau_matrix[pivot_row, pivot_col], precision)

    phase_title = "Fase 1: Búsqueda de Solución Básica Factible" if phase == 1 else "Fase 2: Optimización de la Función Objetivo Original"

    return SimplexStep(
        step_number=step_number,
        phase=phase,
        phase_title=phase_title,
        description=description,
        objective_name=objective_name,
        indicator_label=indicator_label,
        headers=headers,
        matrix=matrix_clean,
        basic_vars=list(basic_vars),
        cb=cb_clean,
        cj=cj_clean,
        zj=zj_clean,
        zj_cj=zj_cj_clean,
        pivot_row=pivot_row,
        pivot_col=pivot_col,
        pivot_element=pivot_elem,
        entering_var=entering_var,
        leaving_var=leaving_var,
        ratios=ratios_clean,
        row_operations=row_operations or []
    )
