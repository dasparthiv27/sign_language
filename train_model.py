import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load landmark data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")

model = RandomForestClassifier(
    n_estimators=500,          # more trees = better stability
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced",   # helps if dataset uneven
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)

model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(model, "sign_model.pkl")

print("\n✅ Model trained successfully")
print("🎯 Accuracy:", accuracy)
print("💾 Model saved as sign_model.pkl")
