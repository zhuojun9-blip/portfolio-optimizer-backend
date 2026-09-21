import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException

import version2 as api


class SolverFailureTests(unittest.TestCase):
    def test_infeasible_target_return_is_rejected(self):
        with self.assertRaises(HTTPException) as context:
            api.optimize_target_return(
                np.array([0.10, 0.20]),
                np.eye(2),
                target_ret=0.30,
                allow_short=False,
            )

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("infeasible", context.exception.detail)
        self.assertIn("10.00%", context.exception.detail)
        self.assertIn("20.00%", context.exception.detail)

    def test_solver_failure_is_reported(self):
        failed_result = SimpleNamespace(success=False, message="Iteration limit reached")
        with patch.object(api, "minimize", return_value=failed_result):
            with self.assertRaises(HTTPException) as context:
                api.optimize_min_variance(
                    np.array([0.10, 0.20]),
                    np.eye(2),
                    allow_short=False,
                )

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("Minimum-variance optimization failed", context.exception.detail)
        self.assertIn("Iteration limit reached", context.exception.detail)


if __name__ == "__main__":
    unittest.main()