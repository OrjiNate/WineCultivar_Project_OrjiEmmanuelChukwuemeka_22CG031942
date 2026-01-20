import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load the dataset
wine_data = load_wine()
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df['cultivar'] = wine_data.target

# 2. Feature Selection (Selecting 6 features + Target)
selected_features = ['alcohol', 'malic_acid', 'ash', 'magnesium', 'flavanoids', 'color_intensity']
X = df[selected_features]
y = df['cultivar']

# 3. Preprocessing & Model Pipeline
# Scaling is mandatory for chemical properties with different ranges
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(pipeline, 'model/wine_cultivar_model.pkl')
print("Model saved in model/wine_cultivar_model.pkl")