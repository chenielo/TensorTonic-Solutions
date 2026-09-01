import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    out = pd.DataFrame()
    for df in dfs:
        df = pd.DataFrame(df)
        out = pd.concat([out, df], axis=0)
    return [out.shape, out.to_dict("list")]
        