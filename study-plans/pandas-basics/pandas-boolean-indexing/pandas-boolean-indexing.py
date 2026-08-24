import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    dff = df[df[column]>threshold]
    return {"filtered_data": dff.to_dict("list")
           ,"count": len(dff)
           }