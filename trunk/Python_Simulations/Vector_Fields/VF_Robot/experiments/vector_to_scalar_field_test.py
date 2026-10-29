"""
3-robot magnitude-only (vector to scalar) primitives on one configurable field.

The robots sense only the speed m = |v|, never its direction. Every critical
point is a zero of m, so it is a minimum of m.

    Attraction  scalar_gradient_descent (existing primitive) on m.
    Orbit       new estimate, then the same orbit law (paper Eq. 11) as
                critical_point_orbiter_plane_fitting.

    Magnitude center estimate (per step)
        fit the plane m(x, y) = a x + b y + c0 through the three readings
        g <- (a, b),  m_c <- fitted m at the centroid c
        p_hat <- c - m_c g / |g|^2      (nearest point on the plane's zero line)

Near a critical point m is a cone, m = |J (x - x*)|. When J^T J is a multiple
of I (linear center, linear saddle, gyre center, any potential flow) the
cone is round and, in the small-formation limit, p_hat equals the vector
Newton estimate. Fitting a plane to a cone with a triangle of circumradius R
adds error of order R^2 / r radially and R / 4 sideways, so unlike the vector
method it is not exact in linear fields.

Output: experiments/outputs/vector_to_scalar_field_test/<config>/
Run from VF_Robot, with the same flags as plane_fit_field_test.py:
    venv/bin/python3 experiments/vector_to_scalar_field_test.py
    venv/bin/python3 experiments/vector_to_scalar_field_test.py --field linear_saddle --side 0.1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _field_tests_common as ft

if __name__ == "__main__":
    args = ft.build_parser(__doc__.strip().splitlines()[0],
                           os.path.join(ft.PROJECT_ROOT, "experiments", "outputs", "vector_to_scalar_field_test")
                           ).parse_args()
    ft.run_suite("magnitude", args)
