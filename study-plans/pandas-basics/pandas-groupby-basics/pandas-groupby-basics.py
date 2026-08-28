import pandas as pd

def groupby_basics(data, group_col, value_col):
    """
    Returns: dict with 'sum', 'mean', 'count' (each a dict)
    """
    df = pd.DataFrame(data)
    df_agg = df.groupby(group_col)[value_col]
    return {
        "sum": df_agg.sum().to_dict(),
        "mean": df_agg.mean().to_dict(),
        "count": df_agg.count().to_dict()
    }