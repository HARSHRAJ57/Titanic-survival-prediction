import pandas as pd
import pickle
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


# =========================
# 1️⃣ TRAIN & SAVE MODEL
# =========================

train = pd.read_csv(r"C:\Users\Harsh\Downloads\train.csv")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']

# Numeric & categorical features
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

model_path = os.path.join(os.path.dirname(__file__), "titanic_full_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved successfully!")

# =========================
# 2️⃣ LOAD MODEL
# =========================

with open(model_path, "rb") as f:
    model = pickle.load(f)


# =========================
# 3️⃣ PYQT GUI
# =========================

# =========================
# 3️⃣ PYQT GUI (UPDATED THEME)
# =========================

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox, QFrame
)
from PyQt5.QtGui import QFont, QPalette, QBrush, QLinearGradient, QColor
from PyQt5.QtCore import Qt


class TitanicApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚢 Titanic Survival Predictor")
        self.setGeometry(300, 150, 500, 650)
        self.setStyleSheet(self.get_styles())
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # ===== Glass Card Frame =====
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout()

        # ===== Title =====
        title = QLabel("🚢 TITANIC SURVIVAL PREDICTOR")
        title.setFont(QFont("Times New Roman", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: gold; letter-spacing: 2px;")
        card_layout.addWidget(title)

        subtitle = QLabel("Will you survive the voyage?")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: black; font-size: 14px;")
        card_layout.addWidget(subtitle)

        # ===== Inputs =====
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age")

        self.fare_input = QLineEdit()
        self.fare_input.setPlaceholderText("Fare")

        self.sibsp_input = QLineEdit()
        self.sibsp_input.setPlaceholderText("Siblings/Spouses aboard")

        self.parch_input = QLineEdit()
        self.parch_input.setPlaceholderText("Parents/Children aboard")

        self.gender_input = QComboBox()
        self.gender_input.addItems(["male", "female"])

        self.embarked_input = QComboBox()
        self.embarked_input.addItems(["S", "C", "Q"])

        self.pclass_input = QComboBox()
        self.pclass_input.addItems(["1", "2", "3"])

        inputs = [
            self.age_input, self.fare_input,
            self.sibsp_input, self.parch_input,
            self.gender_input, self.embarked_input,
            self.pclass_input
        ]

        for widget in inputs:
            widget.setMinimumHeight(40)
            card_layout.addWidget(widget)

        # ===== Predict Button =====
        self.button = QPushButton("⚓ Predict Survival")
        self.button.setMinimumHeight(45)
        self.button.clicked.connect(self.predict_survival)
        card_layout.addWidget(self.button)

        card.setLayout(card_layout)
        main_layout.addWidget(card)
        self.setLayout(main_layout)

    def get_styles(self):
        return """
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #e0f7fa,
                stop:0.5 #b2ebf2,
                stop:1 #80deea
            );
        }

        QFrame#card {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 25px;
        }

        QLabel {
            color: black;
        }

        QLineEdit, QComboBox {
            background-color: Yellow;
            border: 1px solid #b0bec5;
            border-radius: 8px;
            padding: 8px;
            font-size: 14px;
            color: black;
        }

        QPushButton {
            background-color: #0277bd;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            color: white;
        }

        QPushButton:hover {
            background-color: #01579b;
        }

        QMessageBox {
            background-color: white;
        }
        """

    def predict_survival(self):
        try:
            data = pd.DataFrame([{
                'Pclass': int(self.pclass_input.currentText()),
                'Sex': self.gender_input.currentText(),
                'Age': float(self.age_input.text()),
                'SibSp': int(self.sibsp_input.text()),
                'Parch': int(self.parch_input.text()),
                'Fare': float(self.fare_input.text()),
                'Embarked': self.embarked_input.currentText()
            }])

            prediction = model.predict(data)
            probability = model.predict_proba(data)[0][1]

            if prediction[0] == 1:
                QMessageBox.information(
                    self,
                    "Prediction Result",
                    f"🟢 Likely to Survive\n\nSurvival Probability: {round(probability*100,2)}%"
                )
            else:
                QMessageBox.information(
                    self,
                    "Prediction Result",
                    f"🔴 Likely Did Not Survive\n\nSurvival Probability: {round(probability*100,2)}%"
                )

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")

# =========================
# 4️⃣ RUN APP
# =========================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TitanicApp()
    window.show()
    sys.exit(app.exec_())