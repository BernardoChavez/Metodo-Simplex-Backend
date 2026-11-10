from enum import Enum
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator

class OptimizationType(str, Enum):
    MIN = "min"
    MAX = "max"

class ConstraintOperator(str, Enum):
    LEQ = "<="
    GEQ = ">="
    EQ = "="

class ConstraintInput(BaseModel):
    name: Optional[str] = Field(default=None, description="Nombre descriptivo de la restricción")
    coefficients: List[float] = Field(..., description="Coeficientes de cada variable de decisión en la restricción")
    operator: ConstraintOperator = Field(..., description="Operador de desigualdad o igualdad (<=, >=, =)")
    rhs: float = Field(..., description="Lado derecho (Right-Hand Side)")

class SimplexRequest(BaseModel):
    optimization_type: OptimizationType = Field(default=OptimizationType.MIN, description="min para minimizar, max para maximizar")
    c: List[float] = Field(..., description="Vector de coeficientes de la función objetivo Z")
    variable_names: Optional[List[str]] = Field(default=None, description="Nombres descriptivos de las variables (ej: Maíz, Sorgo)")
    constraints: List[ConstraintInput] = Field(..., description="Lista de restricciones lineales")
    max_iterations: int = Field(default=50, ge=1, le=200, description="Límite máximo de iteraciones para evitar bucles")
    decimal_precision: int = Field(default=4, ge=1, le=8, description="Número de decimales para redondeo en la presentación")

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v, info):
        c_len = len(info.data.get("c", [])) if "c" in info.data else None
        if c_len:
            for idx, constr in enumerate(v):
                if len(constr.coefficients) != c_len:
                    raise ValueError(f"La restricción {idx+1} tiene {len(constr.coefficients)} coeficientes, pero la función objetivo tiene {c_len}.")
        return v

class SimplexStep(BaseModel):
    step_number: int
    phase: int = Field(..., description="1 para Fase 1, 2 para Fase 2")
    phase_title: str
    description: str
    objective_name: str = Field(default="Z", description="'W' en Fase 1, 'Z' en Fase 2")
    indicator_label: str = Field(default="Z_j - C_j", description="Etiqueta del indicador ('W_j - C_j', 'Z_j - C_j' o 'C_j - Z_j')")
    headers: List[str]
    matrix: List[List[float]]
    basic_vars: List[str]
    cb: List[float]
    cj: List[float]
    zj: List[float]
    zj_cj: List[float]
    pivot_row: Optional[int] = None
    pivot_col: Optional[int] = None
    pivot_element: Optional[float] = None
    entering_var: Optional[str] = None
    leaving_var: Optional[str] = None
    ratios: Optional[List[Optional[float]]] = None
    row_operations: List[str] = Field(default_factory=list)

class ConstraintAnalysis(BaseModel):
    name: str = Field(..., description="Nombre o identificador de la restricción")
    operator: str = Field(..., description="Operador original (<=, >=, =)")
    rhs: float = Field(..., description="Valor requerido (Lado Derecho)")
    lhs: float = Field(..., description="Valor alcanzado con la mezcla óptima (Lado Izquierdo)")
    slack_or_surplus: float = Field(..., description="Holgura o exceso absoluto")
    is_binding: bool = Field(..., description="True si la restricción es activa / cuello de botella")
    shadow_price: float = Field(..., description="Precio sombra dual asociado ($/unidad o Bs/unidad)")

class SensitivityRange(BaseModel):
    name: str = Field(..., description="Nombre de la variable o restricción")
    current_value: float = Field(..., description="Valor actual (c_j o b_i)")
    min_allowable: Optional[float] = Field(default=None, description="Límite inferior permisible")
    max_allowable: Optional[float] = Field(default=None, description="Límite superior permisible")
    is_basic: Optional[bool] = Field(default=None, description="True si la variable está en la base óptima")

class DualProblem(BaseModel):
    objective_type: str = Field(..., description="max o min para el problema dual")
    objective_function: str = Field(..., description="Ecuación de la función objetivo dual W")
    decision_variables: List[str] = Field(default_factory=list, description="Variables duales Y_1 ... Y_m")
    constraints: List[str] = Field(default_factory=list, description="Restricciones del problema dual")
    optimal_w: Optional[float] = Field(default=None, description="Valor óptimo dual W* = Z*")
    dual_solution: Dict[str, float] = Field(default_factory=dict, description="Valores óptimos de las variables duales Y_i")

class SimplexResponse(BaseModel):
    status: Literal["OPTIMAL", "INFEASIBLE", "UNBOUNDED", "MAX_ITERATIONS_REACHED", "ERROR"]
    message: str
    optimization_type: str
    optimal_z: Optional[float] = None
    solution: Dict[str, float] = Field(default_factory=dict, description="Valores óptimos de las variables de decisión")
    slack_surplus: Dict[str, float] = Field(default_factory=dict, description="Valores de holguras y excesos")
    shadow_prices: Optional[Dict[str, float]] = Field(default=None, description="Precios sombra / costos de oportunidad")
    constraints_analysis: List[ConstraintAnalysis] = Field(default_factory=list, description="Análisis detallado de cada restricción y recurso")
    cost_sensitivity: List[SensitivityRange] = Field(default_factory=list, description="Rangos de sensibilidad para coeficientes c_j")
    rhs_sensitivity: List[SensitivityRange] = Field(default_factory=list, description="Rangos de sensibilidad para lados derechos b_i")
    dual_problem: Optional[DualProblem] = Field(default=None, description="Formulación y solución matemática del problema dual")
    has_alternative_optima: bool = Field(default=False, description="Indica si existen múltiples soluciones óptimas alternativas")
    alternative_optima_vars: List[str] = Field(default_factory=list, description="Variables no básicas con costo reducido cero")
    total_iterations: int
    phase1_iterations: int
    phase2_iterations: int
    steps: List[SimplexStep]
