from fastapi import APIRouter, HTTPException, status
from ...schemas.simplex_schema import SimplexRequest, SimplexResponse, ConstraintInput, ConstraintOperator, OptimizationType
from ...services.two_phase import TwoPhaseSimplexSolver

router = APIRouter(prefix="/simplex", tags=["Simplex Two-Phase"])

@router.post(
    "/solve",
    response_model=SimplexResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolver modelo de Programación Lineal mediante Método Simplex Dos Fases",
    description="Procesa funciones objetivo de Minimización o Maximización con restricciones <=, >= y =, retornando el historial paso a paso de cada iteración, holguras, precios sombra duales y análisis de recursos."
)
def solve_simplex(request: SimplexRequest) -> SimplexResponse:
    try:
        solver = TwoPhaseSimplexSolver(request)
        response = solver.solve()
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al resolver el problema Simplex: {str(e)}"
        )

@router.get(
    "/sample-canes",
    response_model=SimplexRequest,
    summary="Obtener datos precargados del problema de raciones para canes (Proyecto IO1)",
    description="Retorna el modelo con 7 materias primas (Maíz amarillo, Sorgo granífero, Afrecho de arroz, Afrecho de trigo, Torta de soya, Harina de carne/hueso, Aceite de soya) y sus restricciones nutricionales para un lote de 100 kg."
)
def get_sample_canes() -> SimplexRequest:
    return SimplexRequest(
        optimization_type=OptimizationType.MIN,
        variable_names=[
            "Maíz amarillo",
            "Sorgo granífero",
            "Afrecho de arroz",
            "Afrecho de trigo",
            "Torta de soya",
            "Harina de carne/hueso",
            "Aceite de soya"
        ],
        # Costos en Bs/kg según la formulación del proyecto
        c=[2.95, 1.73, 1.00, 1.35, 3.58, 5.80, 9.50],
        constraints=[
            # Proteína (mínimo 18% para 100 kg -> >= 1800)
            ConstraintInput(
                name="Proteína Cruda (mínimo 18% -> >= 1800)",
                coefficients=[8.5, 9.0, 12.5, 15.0, 45.0, 50.0, 0.0],
                operator=ConstraintOperator.GEQ,
                rhs=1800.0
            ),
            # Grasa (mínimo 5.5% para 100 kg -> >= 550)
            ConstraintInput(
                name="Grasa Cruda (mínimo 5.5% -> >= 550)",
                coefficients=[3.8, 3.3, 13.0, 4.0, 1.5, 10.0, 99.0],
                operator=ConstraintOperator.GEQ,
                rhs=550.0
            ),
            # Fibra (mínimo 3.5% para 100 kg -> >= 350)
            ConstraintInput(
                name="Fibra Cruda (mínimo 3.5% -> >= 350)",
                coefficients=[2.2, 2.5, 11.0, 10.0, 6.0, 1.0, 0.0],
                operator=ConstraintOperator.GEQ,
                rhs=350.0
            ),
            # Calcio (mínimo 0.5% para 100 kg -> >= 50)
            ConstraintInput(
                name="Calcio (mínimo 0.5% -> >= 50)",
                coefficients=[0.02, 0.04, 0.05, 0.12, 0.28, 8.00, 0.0],
                operator=ConstraintOperator.GEQ,
                rhs=50.0
            ),
            # Producción total del lote (= 100 kg)
            ConstraintInput(
                name="Producción total del lote (= 100 kg)",
                coefficients=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                operator=ConstraintOperator.EQ,
                rhs=100.0
            )
        ],
        decimal_precision=4,
        max_iterations=50
    )
