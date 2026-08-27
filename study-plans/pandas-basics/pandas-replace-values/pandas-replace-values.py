import pandas as pd

def replace_values(data, column, old_val, new_val):
    """
    Returns: dict with 'data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    cnt = (df[column] == old_val).sum()
    df[column] = df[column].apply(lambda x: new_val if x == old_val else x)
    return {
        "data":df.to_dict("list"),
        "count": cnt
    }