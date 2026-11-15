import os
import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Carregar parâmetros
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    random_state = params["data_split"]["random_state"]

    # 2. Carregar dados processados
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # Separar features (X) e target (y) - assumindo 'Revenue' como alvo
    target_col = "Revenue"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 3. Treinar o modelo
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    # 4. Avaliar modelo
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Acurácia do modelo no teste: {acc:.4f}")

    # 5. Salvar o modelo treinado
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    print("Modelo salvo com sucesso em models/model.joblib")

if __name__ == "__main__":
    train_model()