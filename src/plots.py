import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

def generate_plots():
    os.makedirs("reports/figures", exist_ok=True)
    
    # 1. Carregar modelo e dados de teste
    model = joblib.load("models/model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    
    target_col = "Revenue"
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    # 2. Gerar e salvar a Matriz de Confusão
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, cmap="Blues", colorbar=False)
    plt.title("Matriz de Confusão - Intenção de Compra")
    plt.savefig("reports/figures/confusion_matrix.png", bbox_inches="tight")
    plt.close()
    
    # 3. Gerar e salvar a Curva ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title("Curva ROC - Modelo MLOps")
    plt.legend(loc="lower right")
    plt.savefig("reports/figures/roc_curve.png", bbox_inches="tight")
    plt.close()
    
    print("Gráficos salvos com sucesso em reports/figures/")

if __name__ == "__main__":
    generate_plots()