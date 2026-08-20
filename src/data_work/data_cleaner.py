import pandas as pd


def clean_rating_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df = df[df["rating"] != -1]
    df.to_csv(output_path, index=False)
    return df
