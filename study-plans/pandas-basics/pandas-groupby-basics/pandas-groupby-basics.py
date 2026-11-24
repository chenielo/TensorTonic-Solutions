import pandas as pd

def groupby_basics(data: dict, group_col: str, value_col: str) -> dict:
    """
    Returns sum, mean, and count dictionaries keyed by group label.
    """
    df = pd.DataFrame(data)
    grouped = df.groupby(group_col)[value_col]
    
    return {
        "sum": grouped.sum().to_dict(),
        "mean": grouped.mean().to_dict(),
        "count": grouped.count().to_dict()
    }