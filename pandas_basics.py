"""
Pandas Basics — Introduction for Junior and Senior Levels
=========================================================

Pandas is the standard library for tabular data in Python. This file introduces
concepts and runnable examples for both levels.

  JUNIOR (Part A)
  ---------------
  • Series and DataFrame — create and inspect
  • Read and inspect — read_csv, head, tail, describe, info, dtypes
  • Select columns and rows — single/multiple columns, filter by condition
  • Missing values — isna, dropna, fillna (median/mean)
  • DataFrame → NumPy — .values for use with sklearn
  • Extra: add/rename columns, value_counts

  SENIOR (Part B)
  ---------------
  • loc, iloc, set_index — label vs position indexing
  • Merge (join) — combine DataFrames on a key
  • groupby and agg — group by column(s), aggregate
  • apply — custom function per column or row
  • sort_values, drop_duplicates — order and dedupe
  • String and datetime — .str accessor, pd.to_datetime, .dt
  • Saving/loading — to_csv, read_csv; to_parquet for scale
  • Performance tips — vectorization, categorical, chunksize

Run: python pandas_basics.py
Or:  from pandas_basics import run_junior, run_senior, run_all
"""

import numpy as np
import pandas as pd


# =============================================================================
# PART A — JUNIOR LEVEL
# =============================================================================


def junior_series_dataframe():
    """Series and DataFrame basics."""
    print("\n" + "=" * 60)
    print("JUNIOR: Series and DataFrame")
    print("=" * 60)

    # Series: 1D labeled array
    s = pd.Series([10, 20, 30, 40], index=["a", "b", "c", "d"])
    print("Series s = pd.Series([10,20,30,40], index=['a','b','c','d'])")
    print(s)
    print("  s['b'] =", s["b"], "  s.values =", s.values)

    # DataFrame: table
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85, 90, 78],
    })
    print("\nDataFrame df:")
    print(df)
    print("  df.shape =", df.shape, "  df.columns =", list(df.columns))


def junior_add_rename_columns():
    """Add new columns and rename existing ones."""
    print("\n" + "=" * 60)
    print("JUNIOR: Add and rename columns")
    print("=" * 60)

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    print("df:")
    print(df)
    df["c"] = df["a"] + df["b"]
    print("\ndf['c'] = df['a'] + df['b']:")
    print(df)
    df = df.rename(columns={"a": "x", "b": "y"})
    print("df.rename(columns={'a':'x','b':'y'}):")
    print(df)


def junior_value_counts():
    """Count occurrences (useful for categorical columns)."""
    print("\n" + "=" * 60)
    print("JUNIOR: value_counts")
    print("=" * 60)

    df = pd.DataFrame({"dept": ["Eng", "Eng", "Sales", "Eng", "Sales"]})
    print("df['dept'].value_counts():")
    print(df["dept"].value_counts())
    print("\nNormalize (proportions): df['dept'].value_counts(normalize=True)")
    print(df["dept"].value_counts(normalize=True).round(2))


def junior_read_inspect():
    """Read CSV and inspect with head, describe, info."""
    print("\n" + "=" * 60)
    print("JUNIOR: Read and inspect")
    print("=" * 60)

    # Build in-memory CSV-style data (no file needed)
    from io import StringIO
    csv_text = "id,name,age,score\n1,Alice,25,85\n2,Bob,30,90\n3,Charlie,35,78\n4,Diana,28,92\n5,Eve,31,88"
    df = pd.read_csv(StringIO(csv_text))
    print("pd.read_csv(...) → DataFrame:")
    print(df)
    print("\ndf.head(2):\n", df.head(2))
    print("df.tail(2):\n", df.tail(2))
    print("df.describe():\n", df.describe())
    print("df.dtypes:\n", df.dtypes)
    print("df.info():")
    df.info()


def junior_select_columns_rows():
    """Select columns and filter rows."""
    print("\n" + "=" * 60)
    print("JUNIOR: Select columns and rows")
    print("=" * 60)

    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "score": [85, 90, 78, 92],
        "dept": ["Eng", "Eng", "Sales", "Sales"],
    })
    print("df:")
    print(df)

    print("\nSingle column: df['age']")
    print(df["age"])
    print("\nMultiple columns: df[['name', 'score']]")
    print(df[["name", "score"]])

    print("\nFilter rows: df[df['age'] > 28]")
    print(df[df["age"] > 28])
    print("\nFilter: df[df['dept'] == 'Eng']")
    print(df[df["dept"] == "Eng"])


def junior_missing_values():
    """Detect, drop, and fill missing values."""
    print("\n" + "=" * 60)
    print("JUNIOR: Missing values")
    print("=" * 60)

    df = pd.DataFrame({
        "a": [1, 2, None, 4, 5],
        "b": [10, None, 30, 40, 50],
        "c": [100, 200, 300, 400, 500],
    })
    print("df with NaNs:")
    print(df)
    print("\ndf.isna().sum():")
    print(df.isna().sum())

    print("\ndf.dropna() (drop rows with any NaN):")
    print(df.dropna())
    print("\ndf.fillna(0):")
    print(df.fillna(0))
    print("\ndf['a'].fillna(df['a'].median()):")
    df_filled = df.copy()
    df_filled["a"] = df_filled["a"].fillna(df_filled["a"].median())
    df_filled["b"] = df_filled["b"].fillna(df_filled["b"].mean())
    print(df_filled)


def junior_to_numpy():
    """Convert DataFrame to NumPy for sklearn etc."""
    print("\n" + "=" * 60)
    print("JUNIOR: DataFrame → NumPy")
    print("=" * 60)

    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "y": [0, 1, 0]})
    X = df[["x1", "x2"]].values
    y = df["y"].values
    print("df:")
    print(df)
    print("  X = df[['x1','x2']].values  shape:", X.shape)
    print("  y = df['y'].values ", y)


def run_junior():
    """Run all junior-level pandas sections."""
    junior_series_dataframe()
    junior_add_rename_columns()
    junior_value_counts()
    junior_read_inspect()
    junior_select_columns_rows()
    junior_missing_values()
    junior_to_numpy()
    print("\n--- End of Junior Pandas ---\n")


# =============================================================================
# PART B — SENIOR LEVEL
# =============================================================================


def senior_loc_iloc_index():
    """Label-based (loc) vs position-based (iloc) indexing."""
    print("\n" + "=" * 60)
    print("SENIOR: loc, iloc, and index")
    print("=" * 60)

    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 28],
        "score": [85, 90, 78, 92],
    }, index=["row0", "row1", "row2", "row3"])
    print("df with custom index:")
    print(df)

    print("\ndf.loc['row1'] (row by label):")
    print(df.loc["row1"])
    print("df.loc['row1':'row2', ['name', 'score']]:")
    print(df.loc["row1":"row2", ["name", "score"]])

    print("\ndf.iloc[0] (first row by position):")
    print(df.iloc[0])
    print("df.iloc[1:3, [0, 2]]:")
    print(df.iloc[1:3, [0, 2]])

    df_indexed = df.set_index("name")
    print("\ndf.set_index('name'):")
    print(df_indexed)
    print("  df_indexed.loc['Bob']:")
    print(df_indexed.loc["Bob"])


def senior_merge_join():
    """Combine DataFrames with merge (join)."""
    print("\n" + "=" * 60)
    print("SENIOR: Merge (join)")
    print("=" * 60)

    left = pd.DataFrame({"id": [1, 2, 3], "value": ["A", "B", "C"]})
    right = pd.DataFrame({"id": [2, 3, 4], "extra": [10, 20, 30]})
    print("left:\n", left)
    print("right:\n", right)
    merged = pd.merge(left, right, on="id", how="inner")
    print("\npd.merge(left, right, on='id', how='inner'):\n", merged)
    merged_left = pd.merge(left, right, on="id", how="left")
    print("how='left':\n", merged_left)


def senior_sort_drop_duplicates():
    """Sort by column(s) and remove duplicate rows."""
    print("\n" + "=" * 60)
    print("SENIOR: sort_values and drop_duplicates")
    print("=" * 60)

    df = pd.DataFrame({"name": ["Bob", "Alice", "Charlie", "Alice"], "score": [90, 85, 78, 85]})
    print("df:")
    print(df)
    print("\ndf.sort_values('score', ascending=False):")
    print(df.sort_values("score", ascending=False))
    print("\ndf.drop_duplicates(subset=['name'], keep='first'):")
    print(df.drop_duplicates(subset=["name"], keep="first"))


def senior_string_accessor():
    """Use .str for string methods on Series."""
    print("\n" + "=" * 60)
    print("SENIOR: String accessor (.str)")
    print("=" * 60)

    df = pd.DataFrame({"name": ["alice", "BOB", "Charlie"]})
    print("df:")
    print(df)
    print("\ndf['name'].str.upper():")
    print(df["name"].str.upper())
    print("df['name'].str.len():")
    print(df["name"].str.len())
    print("df['name'].str.contains('a'):")
    print(df["name"].str.contains("a"))


def senior_groupby_agg():
    """Group by column(s) and aggregate."""
    print("\n" + "=" * 60)
    print("SENIOR: groupby and agg")
    print("=" * 60)

    df = pd.DataFrame({
        "dept": ["Eng", "Eng", "Sales", "Sales", "Eng"],
        "name": ["A", "B", "C", "D", "E"],
        "salary": [70, 80, 60, 65, 75],
        "bonus": [5, 10, 8, 7, 9],
    })
    print("df:")
    print(df)
    by_dept = df.groupby("dept")
    print("\ndf.groupby('dept').agg({'salary': 'mean', 'bonus': 'sum'}):")
    print(by_dept.agg({"salary": "mean", "bonus": "sum"}))
    print("\ndf.groupby('dept')['salary'].mean():")
    print(by_dept["salary"].mean())


def senior_apply():
    """Apply custom function to columns or rows."""
    print("\n" + "=" * 60)
    print("SENIOR: apply")
    print("=" * 60)

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    print("df:")
    print(df)
    print("\ndf.apply(np.sum, axis=0) (sum each column):")
    print(df.apply(np.sum, axis=0))
    print("df.apply(lambda row: row['a'] + row['c'], axis=1):")
    print(df.apply(lambda row: row["a"] + row["c"], axis=1))


def senior_datetime():
    """Parse dates and use datetime accessor."""
    print("\n" + "=" * 60)
    print("SENIOR: Datetime")
    print("=" * 60)

    df = pd.DataFrame({
        "date_str": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "value": [100, 200, 150],
    })
    df["date"] = pd.to_datetime(df["date_str"])
    print("df with pd.to_datetime(df['date_str']):")
    print(df)
    print("\ndf['date'].dt.year:", df["date"].dt.year.tolist())
    print("df['date'].dt.month:", df["date"].dt.month.tolist())


def senior_saving_loading():
    """Save and load CSV, and mention other formats."""
    print("\n" + "=" * 60)
    print("SENIOR: Saving and loading")
    print("=" * 60)

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    path = "pandas_demo_output.csv"
    df.to_csv(path, index=False)
    print("df.to_csv('pandas_demo_output.csv', index=False)  → file written")
    back = pd.read_csv(path)
    print("pd.read_csv(path):\n", back)
    print("Also: to_excel, to_parquet, read_parquet for larger data.")
    print("Chunked read: pd.read_csv('big.csv', chunksize=10_000) for huge files.")


def senior_performance_tips():
    """Brief performance and style tips."""
    print("\n" + "=" * 60)
    print("SENIOR: Performance and style tips")
    print("=" * 60)

    print("""
  • Prefer vectorized operations (df['col'] * 2) over row-wise apply when possible.
  • Use .loc[] / .iloc[] for clarity; chained indexing (df['a'][0]) can trigger warnings.
  • For huge files: use chunksize in read_csv, or read_parquet.
  • Use categorical dtype for repeated string columns: df['col'] = df['col'].astype('category').
  • Copy only when needed: out = df.copy() before in-place-like changes if you want to keep df.
""")


def run_senior():
    """Run all senior-level pandas sections."""
    senior_loc_iloc_index()
    senior_merge_join()
    senior_sort_drop_duplicates()
    senior_string_accessor()
    senior_groupby_agg()
    senior_apply()
    senior_datetime()
    senior_saving_loading()
    senior_performance_tips()
    print("\n--- End of Senior Pandas ---\n")


# =============================================================================
# RUN
# =============================================================================


def run_all():
    """Run junior then senior sections."""
    print("\n" + "#" * 60)
    print("# Pandas Basics — Junior + Senior")
    print("#" * 60)
    run_junior()
    run_senior()


if __name__ == "__main__":
    run_all()
