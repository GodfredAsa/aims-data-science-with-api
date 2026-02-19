"""
NumPy Basics — Introduction for Junior and Senior Levels
=========================================================

NumPy is the foundation for numerical computing in Python and underpins
pandas, scikit-learn, and most ML libraries. This file introduces:

  JUNIOR: Arrays, indexing, basic operations, aggregations.
  SENIOR: Broadcasting, views vs copies, vectorization, linear algebra,
          reshaping, random, and performance awareness.

Run: python numpy_basics.py
Or:  from numpy_basics import run_junior, run_senior, run_all
"""

import numpy as np


# =============================================================================
# PART A — JUNIOR LEVEL
# =============================================================================


def junior_arrays_creation():
    """Create arrays from lists and using constructors."""
    print("\n" + "=" * 60)
    print("JUNIOR: Array creation")
    print("=" * 60)

    # From list
    arr = np.array([1, 2, 3, 4, 5])
    print("np.array([1,2,3,4,5]) =", arr)
    print("  .shape =", arr.shape, "  .dtype =", arr.dtype)

    # 2D (matrix / table of features)
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    print("\n2D array (3 rows, 2 cols):\n", matrix)
    print("  .shape =", matrix.shape)

    # Constructors
    print("\nnp.zeros((2, 3)):\n", np.zeros((2, 3)))
    print("np.ones((2, 2)):\n", np.ones((2, 2)))
    print("np.arange(0, 10, 2) =", np.arange(0, 10, 2))
    print("np.linspace(0, 1, 5) =", np.linspace(0, 1, 5))


def junior_indexing_slicing():
    """Index and slice 1D and 2D arrays."""
    print("\n" + "=" * 60)
    print("JUNIOR: Indexing and slicing")
    print("=" * 60)

    arr = np.array([10, 20, 30, 40, 50])
    print("arr =", arr)
    print("  arr[0] =", arr[0], "  arr[-1] =", arr[-1])
    print("  arr[1:4] =", arr[1:4], "  (start:stop, stop excluded)")

    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nmatrix:\n", matrix)
    print("  matrix[1, 0] =", matrix[1, 0])
    print("  matrix[:, 0] (first column) =", matrix[:, 0])
    print("  matrix[1:, :2] (rows from 1, cols 0–1):\n", matrix[1:, :2])


def junior_operations_aggregations():
    """Element-wise math and simple aggregations."""
    print("\n" + "=" * 60)
    print("JUNIOR: Operations and aggregations")
    print("=" * 60)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    print("a =", a, "  b =", b)
    print("  a + b =", a + b)
    print("  a * 2 =", a * 2)
    print("  np.sqrt(a) =", np.sqrt(a))

    arr = np.array([2, 4, 6, 8, 10])
    print("\nAggregations on", arr, ":")
    print("  .sum() =", arr.sum(), "  .mean() =", arr.mean())
    print("  .min() =", arr.min(), "  .max() =", arr.max(), "  .std() =", round(arr.std(), 2))

    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    print("\naxis: 0 = down columns, 1 = across rows")
    print("  matrix.sum(axis=0) =", matrix.sum(axis=0))
    print("  matrix.mean(axis=1) =", matrix.mean(axis=1))


def junior_boolean_indexing():
    """Filter arrays with boolean conditions."""
    print("\n" + "=" * 60)
    print("JUNIOR: Boolean indexing (filtering)")
    print("=" * 60)

    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("arr =", arr)
    print("  arr[arr > 5] =", arr[arr > 5])
    print("  arr[arr % 2 == 0] =", arr[arr % 2 == 0])
    mask = (arr >= 3) & (arr <= 7)
    print("  arr[(arr>=3) & (arr<=7)] =", arr[mask])


def run_junior():
    """Run all junior-level NumPy sections."""
    junior_arrays_creation()
    junior_indexing_slicing()
    junior_operations_aggregations()
    junior_boolean_indexing()
    print("\n--- End of Junior NumPy ---\n")


# =============================================================================
# PART B — SENIOR LEVEL
# =============================================================================


def senior_broadcasting():
    """Broadcasting rules and examples."""
    print("\n" + "=" * 60)
    print("SENIOR: Broadcasting")
    print("=" * 60)

    # (3, 2) + (2,) → (3, 2): row [10, 20] added to each row
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    row = np.array([10, 20])
    print("matrix (3x2) + row (2,) → each row gets [10, 20] added:\n", matrix + row)

    # (3, 2) + (3, 1) → (3, 2): column broadcast
    col = np.array([[100], [200], [300]])
    print("\nmatrix + col (3x1):\n", matrix + col)

    # Scalar broadcasts to any shape
    print("\nmatrix * 10 (scalar):\n", matrix * 10)
    print("Rule: dimensions compared from the right; size 1 or missing can broadcast.")


def senior_views_vs_copies():
    """When NumPy returns a view (shared memory) vs a copy."""
    print("\n" + "=" * 60)
    print("SENIOR: Views vs copies")
    print("=" * 60)

    arr = np.array([1, 2, 3, 4, 5])
    slice_view = arr[1:4]  # slice is a view
    slice_view[0] = 999
    print("arr = [1,2,3,4,5]; slice_view = arr[1:4]; slice_view[0] = 999")
    print("  arr =", arr, "  (original changed!)")

    arr2 = np.array([1, 2, 3, 4, 5])
    copy_arr = arr2[1:4].copy()
    copy_arr[0] = 888
    print("\ncopy_arr = arr2[1:4].copy(); copy_arr[0] = 888")
    print("  arr2 =", arr2, "  (unchanged)")
    print("  copy_arr =", copy_arr)
    print("Use .copy() when you need an independent array.")


def senior_vectorization():
    """Avoid Python loops; use array operations."""
    print("\n" + "=" * 60)
    print("SENIOR: Vectorization (no loops)")
    print("=" * 60)

    x = np.linspace(0, 10, 5)
    # Slow: for i in range(len(x)): y[i] = x[i]**2 + 2*x[i]
    y = x**2 + 2 * x
    print("x =", x)
    print("y = x**2 + 2*x (vectorized) =", y)

    # Where / conditional
    arr = np.array([1, -2, 3, -4, 5])
    out = np.where(arr > 0, arr, 0)
    print("\nnp.where(arr > 0, arr, 0) =", out)


def senior_linear_algebra():
    """Dot product, matrix multiply, transpose."""
    print("\n" + "=" * 60)
    print("SENIOR: Linear algebra")
    print("=" * 60)

    u = np.array([1.0, 2.0, 3.0])
    v = np.array([4.0, 5.0, 6.0])
    print("u =", u, "  v =", v)
    print("  np.dot(u, v) =", np.dot(u, v))
    print("  u @ v =", u @ v)

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("\nA @ B (matrix multiply):\n", A @ B)
    print("A.T (transpose):\n", A.T)


def senior_reshaping_stacking():
    """reshape, ravel, stack, concatenate."""
    print("\n" + "=" * 60)
    print("SENIOR: Reshaping and stacking")
    print("=" * 60)

    arr = np.arange(12)
    print("arr = np.arange(12) =", arr)
    mat = arr.reshape(3, 4)
    print("arr.reshape(3, 4):\n", mat)
    print("  mat.ravel() =", mat.ravel())

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    print("\nnp.vstack((a, b)) (vertical stack):\n", np.vstack((a, b)))
    print("np.hstack((a, b)) (horizontal stack):\n", np.hstack((a, b)))
    print("np.concatenate([a, b], axis=0) same as vstack")


def senior_random():
    """Reproducible random numbers and common distributions."""
    print("\n" + "=" * 60)
    print("SENIOR: Random (reproducible)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    print("rng = np.random.default_rng(42)  # seed for reproducibility")
    print("  rng.random(5) =", rng.random(5))
    print("  rng.integers(0, 10, size=5) =", rng.integers(0, 10, size=5))
    print("  rng.standard_normal(3) =", np.round(rng.standard_normal(3), 3))
    print("  rng.choice([1,2,3,4,5], size=3, replace=False) =", rng.choice([1, 2, 3, 4, 5], size=3, replace=False))


def senior_nan_handling():
    """NaN-safe aggregations and masking."""
    print("\n" + "=" * 60)
    print("SENIOR: NaN handling")
    print("=" * 60)

    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    print("arr =", arr)
    print("  np.nanmean(arr) =", np.nanmean(arr))
    print("  np.nansum(arr) =", np.nansum(arr))
    print("  np.isnan(arr) =", np.isnan(arr))
    valid = arr[~np.isnan(arr)]
    print("  arr[~np.isnan(arr)] =", valid)


def run_senior():
    """Run all senior-level NumPy sections."""
    senior_broadcasting()
    senior_views_vs_copies()
    senior_vectorization()
    senior_linear_algebra()
    senior_reshaping_stacking()
    senior_random()
    senior_nan_handling()
    print("\n--- End of Senior NumPy ---\n")


# =============================================================================
# RUN
# =============================================================================


def run_all():
    """Run junior then senior sections."""
    print("\n" + "#" * 60)
    print("# NumPy Basics — Junior + Senior")
    print("#" * 60)
    run_junior()
    run_senior()


if __name__ == "__main__":
    run_all()
