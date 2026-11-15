import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def preprocess_data():
    # 1. Carregar parâmetros do params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    test_size = params["data_split"]["test_size"]
    random_state = params["data_split"]["random_state"]

    # 2. Carregar o dataset bruto
    input_path = "data/raw/online_shoppers_intention.csv"
    df = pd.read_csv(input_path)

    # 3. Pré-processamento e Encoding
    # Converter colunas booleanas/alvo para numérico se necessário
    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].astype(int)
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)
        
    # Aplicar One-Hot Encoding nas colunas categóricas (Month e VisitorType)
    df = pd.get_dummies(df, columns=["Month", "VisitorType"], drop_first=True)

    # 4. Dividir em treino e teste usando os parâmetros do DVC
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["Revenue"] if "Revenue" in df.columns else None
    )

    # 5. Criar diretório de saída e salvar os arquivos processados
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print("Dados processados e salvos com sucesso em data/processed/")

if __name__ == "__main__":
    preprocess_data()