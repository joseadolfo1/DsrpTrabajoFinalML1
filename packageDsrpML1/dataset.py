from pathlib import Path
import pandas as pd
from loguru import logger
from packageDsrpML1.config import CSV_FILE


def load_csv(path: Path = CSV_FILE) -> pd.DataFrame:
    logger.info(f"Cargando CSV desde: {path}")
   
    df = pd.read_csv(path, sep=";")
    logger.success(f"CSV cargado correctamente con shape {df.shape}")
    return df


if __name__ == "__main__":
    data = load_csv()
    print(data)

