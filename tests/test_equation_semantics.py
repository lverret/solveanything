import ast
import math
import unittest

import torch

import solveanything as solver


class PolynomialModel(torch.nn.Module):
    def forward(self, x, y):
        return x**2 + 2 * y


class ZeroModel(torch.nn.Module):
    def forward(self, x, y):
        return 0 * x + 0 * y


def residual(formula, model):
    variables, _ = solver.parse_equations([formula])
    field_indices = {variable: index for index, variable in enumerate(variables)}
    x = torch.tensor([[0.2], [0.7]], requires_grad=True)
    y = torch.tensor([[0.3], [0.8]], requires_grad=True)
    lhs, rhs = formula.split("=")
    node = ast.parse(f"{lhs} - ({rhs})", mode="eval").body
    return solver.evaluate(node, {"x": x, "y": y}, model, field_indices)


class EquationSemanticsTests(unittest.TestCase):
    def test_each_field_call_uses_its_own_coordinates(self):
        value = residual("u(0, y) = u(1, y)", PolynomialModel())
        torch.testing.assert_close(value, torch.full_like(value, -1.0))

    def test_derivatives_use_the_fixed_call_coordinate(self):
        value = residual("grad(u(0, y), x) = grad(u(1, y), x)", PolynomialModel())
        torch.testing.assert_close(value, torch.full_like(value, -2.0))

    def test_nested_derivative_at_a_fixed_coordinate(self):
        value = residual("grad(grad(u(0, y), x), x) = 0", PolynomialModel())
        torch.testing.assert_close(value, torch.full_like(value, 2.0))

    def test_conflicting_fixed_coordinates_do_not_overwrite_the_domain(self):
        _, domains = solver.parse_equations(["u(0, y) = u(1, y)"])
        self.assertTrue(math.isnan(domains[0]["x"]))

    def test_single_boundary_coordinate_is_kept(self):
        _, domains = solver.parse_equations(["u(0, y) = 0"])
        self.assertEqual(domains[0]["x"], 0.0)
        self.assertTrue(math.isnan(domains[0]["y"]))


if __name__ == "__main__":
    unittest.main()
