import pandas as pd

def merge_dataframes(left, right, on, how):
    """
    Returns: dict of column to value lists
    """
    df1=pd.DataFrame(left)
    df2=pd.DataFrame(right)
    out = pd.merge(df1, df2, on=on, how=how)
    return out.to_dict("list")