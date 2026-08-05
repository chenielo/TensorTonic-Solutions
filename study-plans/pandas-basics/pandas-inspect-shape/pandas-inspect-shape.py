import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)
    print(df.dtypes)
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "columns": df.columns.to_list(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "total_values": df.size,
    }