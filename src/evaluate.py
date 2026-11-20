import json
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model():
    # 1. Carregar modelo treinado e dados de teste
    model = joblib.load("models/model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")

    target_col = "Revenue"
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 2. Gerar predições
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # 3. Calcular métricas de desempenho
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs))
    }

    print("Métricas calculadas com sucesso:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    # 4. Salvar métricas em JSON para o DVC monitorar
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Métricas salvas em metrics/scores.json")

if __name__ == "__main__":
    evaluate_model()