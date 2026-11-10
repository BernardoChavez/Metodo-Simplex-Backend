import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from ..schemas.simplex_schema import (
    SimplexRequest,
    SimplexResponse,
    SimplexStep,
    ConstraintAnalysis,
    SensitivityRange,
    DualProblem,
    OptimizationType,
    ConstraintOperator,
)
from .formatter import clean_float, format_row_operation, create_step_snapshot

class TwoPhaseSimplexSolver:
    def __init__(self, request: SimplexRequest):
        self.req = request
        self.opt_type = request.optimization_type
        self.orig_c = list(request.c)
        self.num_vars = len(self.orig_c)
        self.num_constraints = len(request.constraints)
        self.precision = request.decimal_precision
        self.max_iter = request.max_iterations

        # Nombres de variables de decisión
        if request.variable_names and len(request.variable_names) == self.num_vars:
            self.decision_var_names = list(request.variable_names)
        else:
            self.decision_var_names = [f"X{j+1}" for j in range(self.num_vars)]

        self.steps: List[SimplexStep] = []
        self.step_counter = 0

    def solve(self) -> SimplexResponse:
        try:
            return self._execute()
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SimplexResponse(
                status="ERROR",
                message=f"Error durante la ejecución del algoritmo: {str(e)}",
                optimization_type=self.opt_type.value,
                optimal_z=None,
                solution={},
                slack_surplus={},
                shadow_prices=None,
                constraints_analysis=[],
                cost_sensitivity=[],
                rhs_sensitivity=[],
                dual_problem=None,
                has_alternative_optima=False,
                alternative_optima_vars=[],
                total_iterations=self.step_counter,
                phase1_iterations=0,
                phase2_iterations=0,
                steps=self.steps,
            )

    def _execute(self) -> SimplexResponse:
        # Paso 1: Estandarización de restricciones (asegurar RHS >= 0)
        a_rows = []
        b_vec = []
        operators = []
        constr_names = []

        for i, constr in enumerate(self.req.constraints):
            row = list(constr.coefficients)
            rhs = float(constr.rhs)
            op = constr.operator
            c_name = constr.name or f"Restricción {i+1}"

            if rhs < 0:
                row = [-val for val in row]
                rhs = -rhs
                if op == ConstraintOperator.LEQ:
                    op = ConstraintOperator.GEQ
                elif op == ConstraintOperator.GEQ:
                    op = ConstraintOperator.LEQ

            a_rows.append(row)
            b_vec.append(rhs)
            operators.append(op)
            constr_names.append(c_name)

        # Paso 2: Identificar variables adicionales (holgura, exceso, artificiales)
        all_var_names = list(self.decision_var_names)
        slack_map: Dict[str, int] = {}    # Nombre -> índice de fila
        surplus_map: Dict[str, int] = {}  # Nombre -> índice de fila
        artif_map: Dict[str, int] = {}    # Nombre -> índice de fila

        basic_vars: List[str] = []

        s_count = 0
        e_count = 0
        a_count = 0

        for i, op in enumerate(operators):
            if op == ConstraintOperator.LEQ:
                s_count += 1
                name = f"S{s_count}"
                all_var_names.append(name)
                slack_map[name] = i
                basic_vars.append(name)
            elif op == ConstraintOperator.GEQ:
                e_count += 1
                e_name = f"E{e_count}"
                all_var_names.append(e_name)
                surplus_map[e_name] = i

                a_count += 1
                a_name = f"A{a_count}"
                all_var_names.append(a_name)
                artif_map[a_name] = i
                basic_vars.append(a_name)
            elif op == ConstraintOperator.EQ:
                a_count += 1
                a_name = f"A{a_count}"
                all_var_names.append(a_name)
                artif_map[a_name] = i
                basic_vars.append(a_name)

        # Construir matriz inicial completa
        total_cols = len(all_var_names)
        A = np.zeros((self.num_constraints, total_cols), dtype=float)

        # Llenar variables de decisión
        for i in range(self.num_constraints):
            for j in range(self.num_vars):
                A[i, j] = a_rows[i][j]

        # Llenar variables adicionales
        for name, row_idx in slack_map.items():
            col_idx = all_var_names.index(name)
            A[row_idx, col_idx] = 1.0

        for name, row_idx in surplus_map.items():
            col_idx = all_var_names.index(name)
            A[row_idx, col_idx] = -1.0

        for name, row_idx in artif_map.items():
            col_idx = all_var_names.index(name)
            A[row_idx, col_idx] = 1.0

        # Vector RHS como columna
        b = np.array(b_vec, dtype=float).reshape(-1, 1)
        tableau = np.hstack([A, b])

        headers = list(all_var_names) + ["RHS"]

        phase1_iters = 0
        phase2_iters = 0

        # Si hay variables artificiales, se requiere Fase 1
        has_artificials = len(artif_map) > 0

        if has_artificials:
            # ==========================================
            # FASE 1: Minimizar W = sum(A_i)
            # ==========================================
            cj_phase1 = np.zeros(total_cols, dtype=float)
            for name in artif_map:
                col_idx = all_var_names.index(name)
                cj_phase1[col_idx] = 1.0

            cb_phase1 = np.zeros(self.num_constraints, dtype=float)
            for i, b_var in enumerate(basic_vars):
                if b_var in artif_map:
                    cb_phase1[i] = 1.0

            # Iteraciones de Fase 1
            while True:
                self.step_counter += 1
                phase1_iters += 1

                # Z_j = sum(CB_i * a_ij)
                zj = np.dot(cb_phase1, tableau[:, :-1])
                # Para Minimizar: indicador = Z_j - C_j. Entra el mayor positivo.
                zj_cj = zj - cj_phase1
                w_val = np.dot(cb_phase1, tableau[:, -1])

                # Condición de Optimalidad Fase 1 (Minimización):
                # Todos zj_cj <= 1e-7
                pos_candidates = [(idx, val) for idx, val in enumerate(zj_cj) if val > 1e-7]

                if not pos_candidates:
                    # Óptimo de Fase 1 alcanzado
                    step_desc = f"Fin de la Fase 1. Se alcanzó la condición de optimalidad con W = {clean_float(w_val, self.precision)}."
                    self.steps.append(
                        create_step_snapshot(
                            step_number=self.step_counter,
                            phase=1,
                            description=step_desc,
                            objective_name="W",
                            indicator_label="W_j - C_j",
                            headers=headers,
                            tableau_matrix=tableau,
                            basic_vars=basic_vars,
                            cb=cb_phase1.tolist(),
                            cj=cj_phase1.tolist(),
                            zj=zj.tolist(),
                            zj_cj=zj_cj.tolist(),
                            precision=self.precision
                        )
                    )
                    break

                if self.step_counter > self.max_iter:
                    return SimplexResponse(
                        status="MAX_ITERATIONS_REACHED",
                        message=f"Se alcanzó el límite máximo de iteraciones ({self.max_iter}) en Fase 1.",
                        optimization_type=self.opt_type.value,
                        optimal_z=None,
                        solution={},
                        slack_surplus={},
                        shadow_prices=None,
                        constraints_analysis=[],
                        cost_sensitivity=[],
                        rhs_sensitivity=[],
                        dual_problem=None,
                        has_alternative_optima=False,
                        alternative_optima_vars=[],
                        total_iterations=self.step_counter,
                        phase1_iterations=phase1_iters,
                        phase2_iterations=0,
                        steps=self.steps,
                    )

                # Variable que entra (mayor Z_j - C_j positivo)
                pivot_col = max(pos_candidates, key=lambda x: x[1])[0]
                entering_var = all_var_names[pivot_col]

                # Prueba de la razón mínima (Test de la Razón)
                col_vals = tableau[:, pivot_col]
                rhs_vals = tableau[:, -1]
                ratios: List[Optional[float]] = []

                best_ratio = float("inf")
                pivot_row = None

                for r_idx in range(self.num_constraints):
                    a_val = col_vals[r_idx]
                    b_val = rhs_vals[r_idx]
                    if a_val > 1e-7:
                        ratio = b_val / a_val
                        ratios.append(ratio)
                        if ratio < best_ratio - 1e-9:
                            best_ratio = ratio
                            pivot_row = r_idx
                    else:
                        ratios.append(None)

                if pivot_row is None:
                    return SimplexResponse(
                        status="UNBOUNDED",
                        message="El problema no está acotado durante la Fase 1.",
                        optimization_type=self.opt_type.value,
                        optimal_z=None,
                        solution={},
                        slack_surplus={},
                        shadow_prices=None,
                        constraints_analysis=[],
                        cost_sensitivity=[],
                        rhs_sensitivity=[],
                        dual_problem=None,
                        has_alternative_optima=False,
                        alternative_optima_vars=[],
                        total_iterations=self.step_counter,
                        phase1_iterations=phase1_iters,
                        phase2_iterations=0,
                        steps=self.steps,
                    )

                leaving_var = basic_vars[pivot_row]
                pivot_elem = tableau[pivot_row, pivot_col]

                # Fórmulas Gauss-Jordan
                row_ops = []
                row_ops.append(format_row_operation(pivot_row, pivot_elem, pivot_row, is_pivot_row=True))
                for r_idx in range(self.num_constraints):
                    if r_idx != pivot_row:
                        factor = tableau[r_idx, pivot_col] / pivot_elem
                        row_ops.append(format_row_operation(r_idx, factor, pivot_row, is_pivot_row=False))

                step_desc = (
                    f"Fase 1 - Iteración {phase1_iters}: Variable entrante: {entering_var} (columna {pivot_col+1}), "
                    f"Variable saliente: {leaving_var} (fila {pivot_row+1}). Elemento pivote: {clean_float(pivot_elem, self.precision)}."
                )

                self.steps.append(
                    create_step_snapshot(
                        step_number=self.step_counter,
                        phase=1,
                        description=step_desc,
                        objective_name="W",
                        indicator_label="W_j - C_j",
                        headers=headers,
                        tableau_matrix=tableau,
                        basic_vars=basic_vars,
                        cb=cb_phase1.tolist(),
                        cj=cj_phase1.tolist(),
                        zj=zj.tolist(),
                        zj_cj=zj_cj.tolist(),
                        pivot_row=pivot_row,
                        pivot_col=pivot_col,
                        entering_var=entering_var,
                        leaving_var=leaving_var,
                        ratios=ratios,
                        row_operations=row_ops,
                        precision=self.precision
                    )
                )

                # Pivoteo Gauss-Jordan
                new_tableau = np.copy(tableau)
                new_tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_elem
                for r_idx in range(self.num_constraints):
                    if r_idx != pivot_row:
                        factor = tableau[r_idx, pivot_col]
                        new_tableau[r_idx, :] = tableau[r_idx, :] - factor * new_tableau[pivot_row, :]

                tableau = new_tableau
                basic_vars[pivot_row] = entering_var
                cb_phase1[pivot_row] = cj_phase1[pivot_col]

            # Verificar factibilidad al final de Fase 1
            w_final = np.dot(cb_phase1, tableau[:, -1])
            if w_final > 1e-5:
                return SimplexResponse(
                    status="INFEASIBLE",
                    message=(
                        f"El modelo NO ES FACTIBLE. En la Fase 1 la suma de variables artificiales "
                        f"es W = {clean_float(w_final, self.precision)} > 0. Las restricciones nutricionales o técnicas son incompatibles."
                    ),
                    optimization_type=self.opt_type.value,
                    optimal_z=None,
                    solution={},
                    slack_surplus={},
                    shadow_prices=None,
                    constraints_analysis=[],
                    cost_sensitivity=[],
                    rhs_sensitivity=[],
                    dual_problem=None,
                    has_alternative_optima=False,
                    alternative_optima_vars=[],
                    total_iterations=self.step_counter,
                    phase1_iterations=phase1_iters,
                    phase2_iterations=0,
                    steps=self.steps,
                )

            # Manejo de variables artificiales degeneradas en la base al final de Fase 1 (W = 0)
            for r_idx, b_var in enumerate(basic_vars):
                if b_var in artif_map:
                    candidate_cols = [
                        c_idx for c_idx, name in enumerate(all_var_names)
                        if name not in artif_map and abs(tableau[r_idx, c_idx]) > 1e-7
                    ]
                    if candidate_cols:
                        p_col = candidate_cols[0]
                        p_val = tableau[r_idx, p_col]
                        tableau[r_idx, :] = tableau[r_idx, :] / p_val
                        for o_idx in range(self.num_constraints):
                            if o_idx != r_idx:
                                factor = tableau[o_idx, p_col]
                                tableau[o_idx, :] = tableau[o_idx, :] - factor * tableau[r_idx, :]
                        basic_vars[r_idx] = all_var_names[p_col]

            # ==========================================
            # TRANSICIÓN A FASE 2:
            # Eliminar columnas artificiales
            # ==========================================
            non_artif_indices = [idx for idx, name in enumerate(all_var_names) if name not in artif_map]
            all_var_names = [all_var_names[idx] for idx in non_artif_indices]

            # Matriz sin columnas de artificiales (manteniendo RHS)
            keep_cols = non_artif_indices + [tableau.shape[1] - 1]
            tableau = tableau[:, keep_cols]
            headers = list(all_var_names) + ["RHS"]

        # ==========================================
        # FASE 2: Optimización de la función Z original
        # ==========================================
        total_cols_phase2 = len(all_var_names)
        cj_phase2 = np.zeros(total_cols_phase2, dtype=float)

        # Llenar coeficientes originales de Z
        for j in range(self.num_vars):
            name = self.decision_var_names[j]
            col_idx = all_var_names.index(name)
            cj_phase2[col_idx] = self.orig_c[j]

        # Costos en la base
        cb_phase2 = np.zeros(self.num_constraints, dtype=float)
        for i, b_var in enumerate(basic_vars):
            if b_var in self.decision_var_names:
                d_idx = self.decision_var_names.index(b_var)
                cb_phase2[i] = self.orig_c[d_idx]
            else:
                cb_phase2[i] = 0.0

        is_min = (self.opt_type == OptimizationType.MIN)
        indicator_lbl = "Z_j - C_j" if is_min else "C_j - Z_j"

        while True:
            self.step_counter += 1
            phase2_iters += 1

            zj = np.dot(cb_phase2, tableau[:, :-1])
            z_val = np.dot(cb_phase2, tableau[:, -1])

            # Indicadores de optimalidad
            if is_min:
                indicator = zj - cj_phase2
                pos_candidates = [(idx, val) for idx, val in enumerate(indicator) if val > 1e-7]
            else:
                indicator = cj_phase2 - zj
                pos_candidates = [(idx, val) for idx, val in enumerate(indicator) if val > 1e-7]

            if not pos_candidates:
                # Solución Óptima alcanzada
                step_desc = (
                    f"¡Solución Óptima encontrada en Fase 2! "
                    f"Todos los indicadores ({indicator_lbl}) son <= 0. "
                    f"Valor óptimo Z = {clean_float(z_val, self.precision)}."
                )
                self.steps.append(
                    create_step_snapshot(
                        step_number=self.step_counter,
                        phase=2,
                        description=step_desc,
                        objective_name="Z",
                        indicator_label=indicator_lbl,
                        headers=headers,
                        tableau_matrix=tableau,
                        basic_vars=basic_vars,
                        cb=cb_phase2.tolist(),
                        cj=cj_phase2.tolist(),
                        zj=zj.tolist(),
                        zj_cj=indicator.tolist(),
                        precision=self.precision
                    )
                )
                break

            if self.step_counter > self.max_iter:
                sol_partial = self._extract_solution(basic_vars, tableau[:, -1])
                return SimplexResponse(
                    status="MAX_ITERATIONS_REACHED",
                    message=f"Se alcanzó el límite máximo de iteraciones ({self.max_iter}) en Fase 2.",
                    optimization_type=self.opt_type.value,
                    optimal_z=clean_float(z_val, self.precision),
                    solution=sol_partial,
                    slack_surplus={},
                    shadow_prices=None,
                    constraints_analysis=[],
                    cost_sensitivity=[],
                    rhs_sensitivity=[],
                    dual_problem=None,
                    has_alternative_optima=False,
                    alternative_optima_vars=[],
                    total_iterations=self.step_counter,
                    phase1_iterations=phase1_iters,
                    phase2_iterations=phase2_iters,
                    steps=self.steps,
                )

            # Variable entrante: mayor valor del indicador
            pivot_col = max(pos_candidates, key=lambda x: x[1])[0]
            entering_var = all_var_names[pivot_col]

            # Prueba de la razón mínima
            col_vals = tableau[:, pivot_col]
            rhs_vals = tableau[:, -1]
            ratios: List[Optional[float]] = []

            best_ratio = float("inf")
            pivot_row = None

            for r_idx in range(self.num_constraints):
                a_val = col_vals[r_idx]
                b_val = rhs_vals[r_idx]
                if a_val > 1e-7:
                    ratio = b_val / a_val
                    ratios.append(ratio)
                    if ratio < best_ratio - 1e-9:
                        best_ratio = ratio
                        pivot_row = r_idx
                else:
                    ratios.append(None)

            if pivot_row is None:
                return SimplexResponse(
                    status="UNBOUNDED",
                    message="El problema no está acotado (la función objetivo puede crecer o decrecer indefinidamente sin violar restricciones).",
                    optimization_type=self.opt_type.value,
                    optimal_z=None,
                    solution={},
                    slack_surplus={},
                    shadow_prices=None,
                    constraints_analysis=[],
                    cost_sensitivity=[],
                    rhs_sensitivity=[],
                    dual_problem=None,
                    has_alternative_optima=False,
                    alternative_optima_vars=[],
                    total_iterations=self.step_counter,
                    phase1_iterations=phase1_iters,
                    phase2_iterations=phase2_iters,
                    steps=self.steps,
                )

            leaving_var = basic_vars[pivot_row]
            pivot_elem = tableau[pivot_row, pivot_col]

            # Operaciones entre filas
            row_ops = []
            row_ops.append(format_row_operation(pivot_row, pivot_elem, pivot_row, is_pivot_row=True))
            for r_idx in range(self.num_constraints):
                if r_idx != pivot_row:
                    factor = tableau[r_idx, pivot_col] / pivot_elem
                    row_ops.append(format_row_operation(r_idx, factor, pivot_row, is_pivot_row=False))

            step_desc = (
                f"Fase 2 - Iteración {phase2_iters}: Variable entrante: {entering_var} (columna {pivot_col+1}), "
                f"Variable saliente: {leaving_var} (fila {pivot_row+1}). Elemento pivote: {clean_float(pivot_elem, self.precision)}."
            )

            self.steps.append(
                create_step_snapshot(
                    step_number=self.step_counter,
                    phase=2,
                    description=step_desc,
                    objective_name="Z",
                    indicator_label=indicator_lbl,
                    headers=headers,
                    tableau_matrix=tableau,
                    basic_vars=basic_vars,
                    cb=cb_phase2.tolist(),
                    cj=cj_phase2.tolist(),
                    zj=zj.tolist(),
                    zj_cj=indicator.tolist(),
                    pivot_row=pivot_row,
                    pivot_col=pivot_col,
                    entering_var=entering_var,
                    leaving_var=leaving_var,
                    ratios=ratios,
                    row_operations=row_ops,
                    precision=self.precision
                )
            )

            # Gauss-Jordan
            new_tableau = np.copy(tableau)
            new_tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_elem
            for r_idx in range(self.num_constraints):
                if r_idx != pivot_row:
                    factor = tableau[r_idx, pivot_col]
                    new_tableau[r_idx, :] = tableau[r_idx, :] - factor * new_tableau[pivot_row, :]

            tableau = new_tableau
            basic_vars[pivot_row] = entering_var
            cb_phase2[pivot_row] = cj_phase2[pivot_col]

        # Extraer solución óptima
        opt_z = float(np.dot(cb_phase2, tableau[:, -1]))
        decision_sol = self._extract_solution(basic_vars, tableau[:, -1])

        # Conjuntos explícitos de variables de holgura y exceso
        slack_names = set(slack_map.keys())
        surplus_names = set(surplus_map.keys())
        slack_surplus_sol = self._extract_slack_surplus(slack_names, surplus_names, basic_vars, tableau[:, -1])

        # Detección de soluciones múltiples (Óptimos Alternativos)
        alt_vars = []
        for c_idx, var_name in enumerate(all_var_names):
            if var_name not in basic_vars:
                if abs(indicator[c_idx]) < 1e-5:
                    alt_vars.append(var_name)
        has_alt = len(alt_vars) > 0

        # Análisis completo de restricciones y cálculo del vector dual y = c_B * B^-1
        constraints_analysis, shadow_prices_dict, B_inv, y_vec = self._compute_constraints_and_duals(
            a_rows=a_rows,
            b_vec=b_vec,
            operators=operators,
            constr_names=constr_names,
            slack_map=slack_map,
            surplus_map=surplus_map,
            basic_vars=basic_vars,
            cb_phase2=cb_phase2,
            solution=decision_sol
        )

        # Análisis de Sensibilidad (Rangos de c_j y b_i) y Formulación Dual
        cost_sens, rhs_sens, dual_problem = self._compute_sensitivity_and_dual(
            a_rows=a_rows,
            b_vec=b_vec,
            operators=operators,
            constr_names=constr_names,
            slack_map=slack_map,
            surplus_map=surplus_map,
            basic_vars=basic_vars,
            cb_phase2=cb_phase2,
            B_inv=B_inv,
            tableau_phase2=tableau[:, :-1],
            headers_phase2=headers[:-1],
            y_vec=y_vec,
            opt_z=clean_float(opt_z, self.precision),
            is_min=is_min
        )

        return SimplexResponse(
            status="OPTIMAL",
            message=f"Solución óptima encontrada exitosamente en {self.step_counter} iteraciones.",
            optimization_type=self.opt_type.value,
            optimal_z=clean_float(opt_z, self.precision),
            solution=decision_sol,
            slack_surplus=slack_surplus_sol,
            shadow_prices=shadow_prices_dict,
            constraints_analysis=constraints_analysis,
            cost_sensitivity=cost_sens,
            rhs_sensitivity=rhs_sens,
            dual_problem=dual_problem,
            has_alternative_optima=has_alt,
            alternative_optima_vars=alt_vars,
            total_iterations=self.step_counter,
            phase1_iterations=phase1_iters,
            phase2_iterations=phase2_iters,
            steps=self.steps,
        )

    def _extract_solution(self, basic_vars: List[str], rhs: np.ndarray) -> Dict[str, float]:
        sol = {}
        for name in self.decision_var_names:
            if name in basic_vars:
                r_idx = basic_vars.index(name)
                sol[name] = clean_float(rhs[r_idx], self.precision)
            else:
                sol[name] = 0.0
        return sol

    def _extract_slack_surplus(
        self,
        slack_names: set,
        surplus_names: set,
        basic_vars: List[str],
        rhs: np.ndarray
    ) -> Dict[str, float]:
        res = {}
        all_aux = sorted(list(slack_names)) + sorted(list(surplus_names))
        for var in all_aux:
            if var in basic_vars:
                r_idx = basic_vars.index(var)
                res[var] = clean_float(rhs[r_idx], self.precision)
            else:
                res[var] = 0.0
        return res

    def _compute_constraints_and_duals(
        self,
        a_rows: List[List[float]],
        b_vec: List[float],
        operators: List[ConstraintOperator],
        constr_names: List[str],
        slack_map: Dict[str, int],
        surplus_map: Dict[str, int],
        basic_vars: List[str],
        cb_phase2: np.ndarray,
        solution: Dict[str, float]
    ) -> Tuple[List[ConstraintAnalysis], Dict[str, float], np.ndarray, np.ndarray]:
        num_c = self.num_constraints
        B = np.zeros((num_c, num_c), dtype=float)

        for col_idx, b_var in enumerate(basic_vars):
            if b_var in self.decision_var_names:
                d_idx = self.decision_var_names.index(b_var)
                for r in range(num_c):
                    B[r, col_idx] = a_rows[r][d_idx]
            elif b_var in slack_map:
                r_idx = slack_map[b_var]
                B[r_idx, col_idx] = 1.0
            elif b_var in surplus_map:
                r_idx = surplus_map[b_var]
                B[r_idx, col_idx] = -1.0
            else:
                for r in range(num_c):
                    B[r, col_idx] = 1.0 if r == col_idx else 0.0

        try:
            B_inv = np.linalg.inv(B)
            y = np.dot(cb_phase2, B_inv)
        except Exception:
            B_inv = np.eye(num_c)
            y = np.zeros(num_c)

        analysis: List[ConstraintAnalysis] = []
        shadow_prices_dict: Dict[str, float] = {}

        for i in range(num_c):
            c_name = constr_names[i]
            op_str = operators[i].value
            req_rhs = b_vec[i]

            lhs = sum(
                a_rows[i][j] * solution.get(self.decision_var_names[j], 0.0)
                for j in range(self.num_vars)
            )

            diff = abs(lhs - req_rhs)
            is_binding = diff < 5e-3
            sp = clean_float(abs(y[i]), self.precision)

            analysis.append(
                ConstraintAnalysis(
                    name=c_name,
                    operator=op_str,
                    rhs=clean_float(req_rhs, self.precision),
                    lhs=clean_float(lhs, self.precision),
                    slack_or_surplus=clean_float(diff, self.precision),
                    is_binding=is_binding,
                    shadow_price=sp
                )
            )

            shadow_prices_dict[f"Restricción {i+1} ({c_name})"] = sp

        return analysis, shadow_prices_dict, B_inv, y

    def _compute_sensitivity_and_dual(
        self,
        a_rows: List[List[float]],
        b_vec: List[float],
        operators: List[ConstraintOperator],
        constr_names: List[str],
        slack_map: Dict[str, int],
        surplus_map: Dict[str, int],
        basic_vars: List[str],
        cb_phase2: np.ndarray,
        B_inv: np.ndarray,
        tableau_phase2: np.ndarray,
        headers_phase2: List[str],
        y_vec: np.ndarray,
        opt_z: float,
        is_min: bool
    ) -> Tuple[List[SensitivityRange], List[SensitivityRange], DualProblem]:
        num_c = self.num_constraints
        num_v = self.num_vars
        c_vec = np.array(self.orig_c, dtype=float)

        # 1. Rangos de sensibilidad para Lado Derecho (b_k)
        x_B = np.dot(B_inv, b_vec)
        rhs_sens: List[SensitivityRange] = []

        for k in range(num_c):
            delta_min = -float('inf')
            delta_max = float('inf')
            col_k = B_inv[:, k]

            for i in range(num_c):
                if col_k[i] > 1e-7:
                    val = -x_B[i] / col_k[i]
                    if val > delta_min:
                        delta_min = val
                elif col_k[i] < -1e-7:
                    val = -x_B[i] / col_k[i]
                    if val < delta_max:
                        delta_max = val

            min_b = clean_float(b_vec[k] + delta_min, self.precision) if delta_min != -float('inf') else None
            max_b = clean_float(b_vec[k] + delta_max, self.precision) if delta_max != float('inf') else None

            rhs_sens.append(
                SensitivityRange(
                    name=f"R{k+1} ({constr_names[k]})",
                    current_value=clean_float(b_vec[k], self.precision),
                    min_allowable=min_b,
                    max_allowable=max_b,
                    is_basic=None
                )
            )

        # 2. Rangos de sensibilidad para Coeficientes de Costo (c_j)
        total_cols = len(headers_phase2)
        c_all = np.zeros(total_cols, dtype=float)
        for j, h in enumerate(headers_phase2):
            if h in self.decision_var_names:
                c_all[j] = c_vec[self.decision_var_names.index(h)]
            else:
                c_all[j] = 0.0

        zj = np.dot(cb_phase2, tableau_phase2)
        zj_cj = zj - c_all

        cost_sens: List[SensitivityRange] = []

        for j, vname in enumerate(self.decision_var_names):
            cur_c = c_vec[j]
            is_basic = vname in basic_vars

            if not is_basic:
                col_idx = headers_phase2.index(vname)
                rc = zj_cj[col_idx]
                if is_min:
                    min_c = clean_float(cur_c + rc, self.precision)
                    max_c = None
                else:
                    min_c = None
                    max_c = clean_float(cur_c - (c_all[col_idx] - zj[col_idx]), self.precision)

                cost_sens.append(
                    SensitivityRange(
                        name=f"X{j+1} ({vname})",
                        current_value=clean_float(cur_c, self.precision),
                        min_allowable=min_c,
                        max_allowable=max_c,
                        is_basic=False
                    )
                )
            else:
                r = basic_vars.index(vname)
                delta_min = -float('inf')
                delta_max = float('inf')

                for k in range(total_cols):
                    if headers_phase2[k] not in basic_vars:
                        a_bar = tableau_phase2[r, k]
                        if is_min:
                            rc = zj_cj[k]
                            rhs_val = -rc
                            if a_bar > 1e-7:
                                val = rhs_val / a_bar
                                if val < delta_max:
                                    delta_max = val
                            elif a_bar < -1e-7:
                                val = rhs_val / a_bar
                                if val > delta_min:
                                    delta_min = val
                        else:
                            rc = c_all[k] - zj[k]
                            rhs_val = -rc
                            if a_bar > 1e-7:
                                val = rhs_val / a_bar
                                if val > delta_min:
                                    delta_min = val
                            elif a_bar < -1e-7:
                                val = rhs_val / a_bar
                                if val < delta_max:
                                    delta_max = val

                min_c = clean_float(cur_c + delta_min, self.precision) if delta_min != -float('inf') else None
                max_c = clean_float(cur_c + delta_max, self.precision) if delta_max != float('inf') else None

                cost_sens.append(
                    SensitivityRange(
                        name=f"X{j+1} ({vname})",
                        current_value=clean_float(cur_c, self.precision),
                        min_allowable=min_c,
                        max_allowable=max_c,
                        is_basic=True
                    )
                )

        # 3. Formulación del Problema Dual
        dual_type = "max" if is_min else "min"
        dual_vars = [f"Y{i+1}" for i in range(num_c)]
        obj_terms = [f"{b_vec[i]} · Y{i+1}" for i in range(num_c)]
        dual_obj = f"{dual_type.upper()} W = " + " + ".join(obj_terms)

        dual_constraints: List[str] = []
        op_dual = "<=" if is_min else ">="

        for j in range(num_v):
            lhs_terms = [f"{a_rows[i][j]} Y{i+1}" for i in range(num_c) if abs(a_rows[i][j]) > 1e-9]
            lhs_str = " + ".join(lhs_terms)
            dual_constraints.append(f"{lhs_str} {op_dual} {c_vec[j]}   [{self.decision_var_names[j]}]")

        dual_sol = {
            f"Y{i+1} ({constr_names[i]})": clean_float(abs(y_vec[i]), self.precision)
            for i in range(num_c)
        }

        dual_problem = DualProblem(
            objective_type=dual_type,
            objective_function=dual_obj,
            decision_variables=dual_vars,
            constraints=dual_constraints,
            optimal_w=opt_z,
            dual_solution=dual_sol
        )

        return cost_sens, rhs_sens, dual_problem
