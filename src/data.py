"""Módulo de ingestão e pré-processamento de dados para o Tech Challenge."""

from pathlib import Path
import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """Carrega o dataset bruto de um arquivo CSV utilizando pathlib.

    Args:
        file_path (str | Path): Caminho para o arquivo CSV bruto.

    Returns:
        pd.DataFrame: DataFrame contendo os dados brutos carregados.
    
    Raises:
        FileNotFoundError: Se o arquivo CSV não for encontrado no caminho especificado.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado no caminho: {path.resolve()}")

    df = pd.read_csv(path)
    return df