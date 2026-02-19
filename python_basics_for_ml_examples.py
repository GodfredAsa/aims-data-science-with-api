"""
Python Basics for Machine Learning — Examples Script
=====================================================

Runnable examples that follow the structure of python_basics_for_ml_guide.txt.
Run this file to see output for each section; uncomment run_all() at the bottom
to execute everything, or call individual section functions.

Usage: python python_basics_for_ml_examples.py
"""

# -----------------------------------------------------------------------------
# PART 1: CORE PYTHON
# -----------------------------------------------------------------------------


def section_1_variables_and_types():
    """1.1 Variables and data types — essential for ML (features, targets, flags)."""
    print("\n" + "=" * 60)
    print("PART 1.1 — Variables and Data Types")
    print("=" * 60)

    x = 5
    name = "hello"
    price = 19.99
    is_trained = False
    missing = None

    print("x =", x, "  type:", type(x))
    print("name =", name, "  type:", type(name))
    print("price =", price, "  type:", type(price))
    print("is_trained =", is_trained, "  type:", type(is_trained))
    print("missing =", missing, "  type:", type(missing))

    # Type conversion (e.g. string from CSV → number for model)
    str_number = "42"
    as_int = int(str_number)
    as_float = float(str_number)
    print("\nConvert '42' → int:", as_int, "→ float:", as_float)

    # ML-style: feature as float, index as int
    feature_value = float("3.14")
    row_index = int(2)
    print("feature_value:", feature_value, "row_index:", row_index)


def section_2_lists():
    """1.2 Lists — ordered sequences, indexing, slicing, list comprehensions."""
    print("\n" + "=" * 60)
    print("PART 1.2 — Lists")
    print("=" * 60)

    scores = [85, 90, 78, 92]
    mixed = [1, "two", 3.0, True]
    print("scores:", scores)
    print("mixed:", mixed)

    print("\nIndexing: scores[0] =", scores[0], ", scores[-1] =", scores[-1])
    print("Slicing: scores[1:3] =", scores[1:3], "(start:stop, stop excluded)")
    scores.append(88)
    print("After append(88):", scores)
    print("Length:", len(scores))

    # List comprehensions (very common in ML)
    squared = [x**2 for x in scores]
    print("\nList comp [x**2 for x in scores]:", squared)
    evens = [x for x in range(10) if x % 2 == 0]
    print("Evens in range(10):", evens)
    # Extra: normalize to 0–1 (min-max) — like a simple scaler
    min_s, max_s = min(scores), max(scores)
    normalized = [(x - min_s) / (max_s - min_s) for x in scores]
    print("Normalized scores (0-1):", [round(x, 2) for x in normalized])


def section_3_dictionaries():
    """1.3 Dictionaries — configs, hyperparameters, metadata."""
    print("\n" + "=" * 60)
    print("PART 1.3 — Dictionaries")
    print("=" * 60)

    params = {"learning_rate": 0.01, "epochs": 100}
    params["batch_size"] = 32
    print("params:", params)
    value = params.get("lr", 0.001)  # key "lr" missing → default 0.001
    print("params.get('lr', 0.001) =", value)

    # Iterate (e.g. to log or grid-search)
    print("\nLoop over items:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    # Extra: nested dict (e.g. model config)
    config = {"model": {"type": "RandomForest", "n_estimators": 100}, "seed": 42}
    print("\nNested config:", config)
    print("config['model']['n_estimators'] =", config["model"]["n_estimators"])


def section_4_control_flow():
    """1.4 Control flow — if/elif/else, for, while."""
    print("\n" + "=" * 60)
    print("PART 1.4 — Control Flow")
    print("=" * 60)

    score = 0.85
    if score > 0.9:
        print("score > 0.9 → Good model")
    elif score > 0.7:
        print("score > 0.7 → Okay")
    else:
        print("→ Retrain")
    print("(score was", score, ")")

    print("\nfor i in range(5):")
    for i in range(5):
        print("  i =", i)

    params = {"lr": 0.01, "epochs": 100}
    print("\nfor key, value in params.items():")
    for key, value in params.items():
        print(f"  {key} = {value}")

    # Extra: loop over list with index (enumerate — useful for batches)
    labels = ["cat", "dog", "bird"]
    print("\nenumerate(labels):")
    for idx, label in enumerate(labels):
        print(f"  index {idx}: {label}")


def section_5_functions():
    """1.5 Functions — reuse logic (preprocessing, metrics, splits)."""
    print("\n" + "=" * 60)
    print("PART 1.5 — Functions")
    print("=" * 60)

    def normalize(x, min_val=0, max_val=1):
        """Scale x to [min_val, max_val] (default 0–1)."""
        return (x - min_val) / (max_val - min_val) if max_val != min_val else 0

    print("normalize(50, min_val=0, max_val=100) =", normalize(50, 0, 100))
    print("normalize(0.5) =", normalize(0.5))  # default 0, 1

    def split_data(data, ratio=0.8):
        """Split list/array into train and test portions."""
        n = int(len(data) * ratio)
        return data[:n], data[n:]

    data = list(range(10))
    train, test = split_data(data, ratio=0.8)
    print("\nsplit_data(range(10), 0.8) → train:", train, "test:", test)

    # Extra: return dict (e.g. metrics)
    def compute_metrics(y_true, y_pred):
        """Placeholder: return dict of metrics."""
        errors = [abs(t - p) for t, p in zip(y_true, y_pred)]
        return {"mae": sum(errors) / len(errors), "n": len(y_true)}

    m = compute_metrics([1, 2, 3], [1.1, 2.2, 2.9])
    print("compute_metrics([1,2,3], [1.1,2.2,2.9]) =", m)


def section_6_imports():
    """1.6 Modules and imports — math, numpy, pandas."""
    print("\n" + "=" * 60)
    print("PART 1.6 — Modules and Imports")
    print("=" * 60)

    import math
    print("import math → math.sqrt(16) =", math.sqrt(16))

    from math import sqrt
    print("from math import sqrt → sqrt(16) =", sqrt(16))

    # Convention: numpy as np, pandas as pd
    import numpy as np
    import pandas as pd
    a = np.array([1, 2, 3])
    print("import numpy as np → np.array([1,2,3]) =", a)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    print("import pandas as pd → DataFrame shape:", df.shape)


# -----------------------------------------------------------------------------
# PART 2: NUMPY — FOUNDATION FOR ML
# -----------------------------------------------------------------------------


def section_2_1_arrays():
    """2.1 NumPy arrays — shape, dtype, 2D, constructors."""
    print("\n" + "=" * 60)
    print("PART 2.1 — NumPy Arrays")
    print("=" * 60)

    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print("arr = np.array([1,2,3,4,5])")
    print("  shape:", arr.shape, "  dtype:", arr.dtype)

    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    print("\nmatrix (3x2):\n", matrix)
    print("  shape:", matrix.shape)

    print("\nConstructors:")
    print("  np.zeros((3,4)):\n", np.zeros((3, 4)))
    print("  np.ones((2,2)):\n", np.ones((2, 2)))
    print("  np.arange(0, 10, 2):", np.arange(0, 10, 2))
    print("  np.linspace(0, 1, 5):", np.linspace(0, 1, 5))


def section_2_2_indexing_slicing():
    """2.2 Indexing and slicing — rows, columns, boolean mask."""
    print("\n" + "=" * 60)
    print("PART 2.2 — Indexing and Slicing")
    print("=" * 60)

    import numpy as np
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("matrix:\n", matrix)
    print("  matrix[1, 0] =", matrix[1, 0])
    print("  matrix[:, 0] (first col) =", matrix[:, 0])
    print("  matrix[1:3, :] (rows 1–2) =\n", matrix[1:3, :])

    arr = np.array([1, 2, 3, 4, 5])
    print("\nBoolean indexing: arr[arr > 3] =", arr[arr > 3])
    print("  arr[arr % 2 == 0] =", arr[arr % 2 == 0])


def section_2_3_operations():
    """2.3 Operations — element-wise, aggregations, dot product."""
    print("\n" + "=" * 60)
    print("PART 2.3 — Operations")
    print("=" * 60)

    import numpy as np
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])
    print("a =", a, "  b =", b)
    print("  a + b =", a + b)
    print("  a * b =", a * b)
    print("  np.sqrt(a) =", np.sqrt(a))

    arr = np.array([1, 2, 3, 4, 5])
    print("\nAggregations: arr.sum() =", arr.sum(), " mean =", arr.mean(), " std =", round(arr.std(), 2))
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    print("  matrix.sum(axis=0) (down cols) =", matrix.sum(axis=0))
    print("  matrix.mean(axis=1) (across rows) =", matrix.mean(axis=1))

    # Dot product / matrix multiply
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    print("\nnp.dot(u, v) =", np.dot(u, v))
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print("A @ B (matrix multiply):\n", A @ B)


def section_2_4_broadcasting():
    """2.4 Broadcasting — shapes aligned automatically."""
    print("\n" + "=" * 60)
    print("PART 2.4 — Broadcasting")
    print("=" * 60)

    import numpy as np
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    row = np.array([10, 20])
    print("matrix:\n", matrix)
    print("row =", row)
    print("matrix + row (add 10 to col0, 20 to col1):\n", matrix + row)

    col = np.array([[100], [200], [300]])
    print("\ncol (3x1):\n", col)
    print("matrix + col:\n", matrix + col)


# -----------------------------------------------------------------------------
# PART 3: PANDAS BASICS
# -----------------------------------------------------------------------------


def section_3_pandas():
    """3.x Pandas — DataFrame, inspect, select, missing, convert to NumPy."""
    print("\n" + "=" * 60)
    print("PART 3 — Pandas Basics")
    print("=" * 60)

    import pandas as pd
    import numpy as np

    # Build a small DataFrame (no CSV needed)
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "age": [25, 30, None, 35, 28],
        "score": [85, 90, 78, None, 92],
        "passed": [True, True, False, True, True],
    })
    print("DataFrame (sample):")
    print(df)
    print("\ndf.shape:", df.shape)
    print("df.columns:", list(df.columns))
    print("\ndf.head(2):\n", df.head(2))
    print("df.describe():\n", df.describe())
    print("df.info():")
    df.info()

    print("\nSelect column: df['age'] =")
    print(df["age"])
    print("\nSelect rows where age > 28:")
    print(df[df["age"] > 28])
    print("\nMultiple columns: df[['name', 'score']]:\n", df[["name", "score"]])

    print("\n--- Missing values ---")
    print("df.isna().sum():\n", df.isna().sum())
    df_filled = df.copy()
    df_filled["age"] = df_filled["age"].fillna(df_filled["age"].median())
    df_filled["score"] = df_filled["score"].fillna(df_filled["score"].mean())
    print("After fillna (age=median, score=mean):\n", df_filled)

    print("\n--- DataFrame → NumPy (for scikit-learn) ---")
    X = df_filled[["age", "score"]].values
    y = df_filled["passed"].values
    print("X = df[['age','score']].values shape:", X.shape)
    print("y = df['passed'].values:", y)

    print("\n--- NumPy → DataFrame ---")
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    back = pd.DataFrame(arr, columns=["f1", "f2"])
    print("pd.DataFrame(arr, columns=['f1','f2']):\n", back)


# -----------------------------------------------------------------------------
# PART 4: GETTING READY FOR ML (mini workflow)
# -----------------------------------------------------------------------------


def section_4_ml_workflow():
    """4.x Mini ML workflow — load (simulated), explore, clean, split."""
    print("\n" + "=" * 60)
    print("PART 4 — Getting Ready for ML (Mini Workflow)")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 1. "Load" data (we create it)
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feature1": np.random.randn(n) * 10 + 5,
        "feature2": np.random.randn(n) * 2 + 1,
        "target": np.random.randn(n) * 3 + 10,
    })
    # Inject a few NaNs
    df.loc[3, "feature1"] = np.nan
    df.loc[7, "feature2"] = np.nan
    print("1. Data shape:", df.shape)
    print("   Missing:\n", df.isna().sum())

    # 2. Explore
    print("\n2. describe():\n", df.describe())

    # 3. Clean
    df_clean = df.fillna(df.median())
    print("\n3. After fillna(median): no missing values")

    # 4. Features and target
    X = df_clean[["feature1", "feature2"]].values
    y = df_clean["target"].values
    print("   X.shape:", X.shape, " y.shape:", y.shape)

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("\n5. train_test_split(0.2):")
    print("   X_train:", X_train.shape, " X_test:", X_test.shape)

    # 6. Scale (optional but good practice for many models)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("6. StandardScaler applied to features")

    print("\nFlow: Load → Explore → Clean → Features → Split → (Train model later)")


# -----------------------------------------------------------------------------
# RUN ALL SECTIONS
# -----------------------------------------------------------------------------


def run_all():
    """Run every section in order (as in the guide)."""
    section_1_variables_and_types()
    section_2_lists()
    section_3_dictionaries()
    section_4_control_flow()
    section_5_functions()
    section_6_imports()
    section_2_1_arrays()
    section_2_2_indexing_slicing()
    section_2_3_operations()
    section_2_4_broadcasting()
    section_3_pandas()
    section_4_ml_workflow()
    print("\n" + "=" * 60)
    print("All sections completed. See python_basics_for_ml_guide.txt for theory.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
