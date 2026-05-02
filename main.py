import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ===============================
# 1. Output folder
# ===============================
os.makedirs("outputs", exist_ok=True)

# ===============================
# 2. Load dataset
# ===============================
df = pd.read_csv("credit_card_fraud_10k.csv")

print("✅ Dataset shape:", df.shape)

# ===============================
# 3. Detect target column
# ===============================
possible_targets = ["Class", "class", "label", "Label", "target", "Target", "fraud", "is_fraud"]

target_column = None
for col in possible_targets:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("❌ Target column not found")

print("🎯 Target column:", target_column)

# ===============================
# 4. CLEAN + SAFE ENCODING (FIX ALL ERRORS)
# ===============================
df = df.dropna()

X = df.drop(target_column, axis=1)
y = df[target_column]

# 🔥 Convert ALL categorical columns safely
for col in X.columns:
    X[col] = pd.factorize(X[col])[0]

# Target encoding
y = pd.factorize(y)[0]

# ===============================
# 5. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 6. Scaling
# ===============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 7. Model
# ===============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# 8. Predictions
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ===============================
# 9. Metrics
# ===============================
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Report:\n", classification_report(y_test, y_pred))

# ===============================
# 10. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ===============================
# 11. FRAUD DISTRIBUTION
# ===============================
plt.figure()
sns.countplot(x=y)
plt.title("Fraud vs Normal")
plt.savefig("outputs/fraud_vs_normal.png")
plt.close()

# ===============================
# 12. PREDICTION DISTRIBUTION
# ===============================
plt.figure()
sns.histplot(y_pred, bins=2)
plt.title("Prediction Distribution")
plt.savefig("outputs/prediction_distribution.png")
plt.close()

# ===============================
# 13. CORRELATION HEATMAP (SAFE)
# ===============================
df_corr = X.copy()
df_corr["target"] = y

plt.figure()
sns.heatmap(df_corr.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# ===============================
# 14. ROC CURVE
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.legend()
plt.savefig("outputs/roc_curve.png")
plt.close()

# ===============================
# 15. Sample Prediction
# ===============================
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\n🔍 Sample Prediction:",
      "Fraud" if prediction[0] == 1 else "Normal")

print("\n✅ ALL OUTPUTS GENERATED SUCCESSFULLY")