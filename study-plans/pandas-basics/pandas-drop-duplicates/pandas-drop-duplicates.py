import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)
    rb = len(df)
    df = df.drop_duplicates()
    ra = len(df)
    return [rb, ra, df.to_dict("list")]