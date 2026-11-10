import unittest
from app.schemas.simplex_schema import (
    SimplexRequest,
    ConstraintInput,
    ConstraintOperator,
    OptimizationType
)
from app.services.two_phase import TwoPhaseSimplexSolver
from app.api.v1.simplex import get_sample_canes

class TestTwoPhaseSimplex(unittest.TestCase):

    def test_canes_project_optimal(self):
        """Verifica que el modelo de canes del proyecto IO1 resuelva correctamente."""
        req = get_sample_canes()
        solver = TwoPhaseSimplexSolver(req)
        resp = solver.solve()

        self.assertEqual(resp.status, "OPTIMAL")
        self.assertIsNotNone(resp.optimal_z)
        self.assertAlmostEqual(resp.optimal_z, 153.4138, places=2)

        # Verificar que Sorgo granífero NO está en slack_surplus
        self.assertNotIn("Sorgo granífero", resp.slack_surplus)

        # Verificar peso total = 100 kg
        total_kg = sum(resp.solution.values())
        self.assertAlmostEqual(total_kg, 100.0, places=2)

        # Verificar análisis de restricciones
        self.assertEqual(len(resp.constraints_analysis), 5)
        # Proteína debe ser activa
        self.assertTrue(resp.constraints_analysis[0].is_binding)
        self.assertGreater(resp.constraints_analysis[0].shadow_price, 0)

        # Restricción 5 (Producción total = 100 kg) debe tener precio sombra calculado
        self.assertTrue(resp.constraints_analysis[4].is_binding)
        self.assertGreater(resp.constraints_analysis[4].shadow_price, 0)

    def test_standard_maximization(self):
        """Prueba de maximización estándar con restricciones <="""
        req = SimplexRequest(
            optimization_type=OptimizationType.MAX,
            c=[3.0, 5.0],
            variable_names=["x1", "x2"],
            constraints=[
                ConstraintInput(name="R1", coefficients=[1.0, 0.0], operator=ConstraintOperator.LEQ, rhs=4.0),
                ConstraintInput(name="R2", coefficients=[0.0, 2.0], operator=ConstraintOperator.LEQ, rhs=12.0),
                ConstraintInput(name="R3", coefficients=[3.0, 2.0], operator=ConstraintOperator.LEQ, rhs=18.0)
            ]
        )
        resp = TwoPhaseSimplexSolver(req).solve()
        self.assertEqual(resp.status, "OPTIMAL")
        self.assertAlmostEqual(resp.optimal_z, 36.0, places=3)
        self.assertAlmostEqual(resp.solution["x1"], 2.0, places=3)
        self.assertAlmostEqual(resp.solution["x2"], 6.0, places=3)

    def test_infeasible_problem(self):
        """Prueba de detección de infactibilidad."""
        req = SimplexRequest(
            optimization_type=OptimizationType.MAX,
            c=[1.0, 1.0],
            variable_names=["x1", "x2"],
            constraints=[
                ConstraintInput(name="R1", coefficients=[1.0, 1.0], operator=ConstraintOperator.LEQ, rhs=2.0),
                ConstraintInput(name="R2", coefficients=[1.0, 1.0], operator=ConstraintOperator.GEQ, rhs=5.0)
            ]
        )
        resp = TwoPhaseSimplexSolver(req).solve()
        self.assertEqual(resp.status, "INFEASIBLE")

    def test_unbounded_problem(self):
        """Prueba de detección de problema no acotado."""
        req = SimplexRequest(
            optimization_type=OptimizationType.MAX,
            c=[2.0, 1.0],
            variable_names=["x1", "x2"],
            constraints=[
                ConstraintInput(name="R1", coefficients=[1.0, -1.0], operator=ConstraintOperator.LEQ, rhs=10.0)
            ]
        )
        resp = TwoPhaseSimplexSolver(req).solve()
        self.assertEqual(resp.status, "UNBOUNDED")

    def test_alternative_optima(self):
        """Prueba de detección de soluciones óptimas múltiples."""
        req = SimplexRequest(
            optimization_type=OptimizationType.MAX,
            c=[2.0, 4.0],
            variable_names=["x1", "x2"],
            constraints=[
                ConstraintInput(name="R1", coefficients=[1.0, 2.0], operator=ConstraintOperator.LEQ, rhs=6.0),
                ConstraintInput(name="R2", coefficients=[1.0, 1.0], operator=ConstraintOperator.LEQ, rhs=4.0)
            ]
        )
        resp = TwoPhaseSimplexSolver(req).solve()
        self.assertEqual(resp.status, "OPTIMAL")
        self.assertAlmostEqual(resp.optimal_z, 12.0, places=3)
        self.assertTrue(resp.has_alternative_optima)

if __name__ == "__main__":
    unittest.main()

