import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QLineEdit, QPushButton, QVBoxLayout,
    QMessageBox, QFrame, QGraphicsDropShadowEffect,
    QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt


train = pd.read_csv(r"C:\Users\Harsh\Downloads\train.csv")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked', 'Pclass']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

model = pipeline


class TitanicApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚢 Titanic Survival Predictor")
        self.setGeometry(250, 40, 800, 950)
        self.setStyleSheet(self.get_styles())
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout()
        card_layout.setSpacing(18)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 120))
        card.setGraphicsEffect(shadow)

        title = QLabel("🚢 TITANIC SURVIVAL PREDICTOR")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("mainTitle")
        card_layout.addWidget(title)

        subtitle = QLabel("Predict Survival Using Machine Learning Model")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            color: #00008B;
            font-size: 15px;
            font-weight: bold;
        """)
        card_layout.addWidget(subtitle)

        acc_label = QLabel(f"Model Accuracy: {round(accuracy*100,2)}%")
        acc_label.setAlignment(Qt.AlignCenter)
        acc_label.setStyleSheet("""
            color: yellow;
            font-size: 15px;
            font-weight: bold;
        """)
        card_layout.addWidget(acc_label)

        self.age = QLineEdit()
        self.age.setPlaceholderText("Enter Age")

        self.fare = QLineEdit()
        self.fare.setPlaceholderText("Enter Fare")

        self.sibsp = QLineEdit()
        self.sibsp.setPlaceholderText("Siblings/Spouses")

        self.parch = QLineEdit()
        self.parch.setPlaceholderText("Parents/Children")

        self.gender = QComboBox()
        self.gender.addItems(["male", "female"])

        self.embarked = QComboBox()
        self.embarked.addItems(["S", "C", "Q"])

        self.pclass = QComboBox()
        self.pclass.addItems(["1", "2", "3"])

        for widget in [self.age, self.fare, self.sibsp,
                       self.parch, self.gender,
                       self.embarked, self.pclass]:
            widget.setMinimumHeight(42)
            card_layout.addWidget(widget)

        btn = QPushButton("⚓ Predict Survival")
        btn.setMinimumHeight(48)
        btn.clicked.connect(self.predict)
        card_layout.addWidget(btn)

        survived_counts = train['Survived'].value_counts()

        table1 = QTableWidget(2, 2)
        table1.setHorizontalHeaderLabels(["Status", "Count"])
        table1.setItem(0, 0, QTableWidgetItem("Not Survived"))
        table1.setItem(0, 1, QTableWidgetItem(str(survived_counts[0])))
        table1.setItem(1, 0, QTableWidgetItem("Survived"))
        table1.setItem(1, 1, QTableWidgetItem(str(survived_counts[1])))

        card_layout.addWidget(QLabel("Total Survival Count"))
        card_layout.addWidget(table1)

        gender_class = pd.crosstab(
            [train['Pclass'], train['Sex']],
            train['Survived']
        )

        table2 = QTableWidget(len(gender_class), 4)
        table2.setHorizontalHeaderLabels(
            ["Class", "Gender", "Not Survived", "Survived"]
        )

        for row_idx, ((pclass, sex), values) in enumerate(gender_class.iterrows()):
            table2.setItem(row_idx, 0, QTableWidgetItem(str(pclass)))
            table2.setItem(row_idx, 1, QTableWidgetItem(sex))
            table2.setItem(row_idx, 2, QTableWidgetItem(str(values[0])))
            table2.setItem(row_idx, 3, QTableWidgetItem(str(values[1])))

        card_layout.addWidget(QLabel("Male/Female Survival per Class"))
        card_layout.addWidget(table2)

        graph_btn = QPushButton("📊 Show Model Accuracy Graph")
        graph_btn.clicked.connect(self.show_accuracy_graph)
        card_layout.addWidget(graph_btn)

        class_btn = QPushButton("👥 Show Survival per Class Graph")
        class_btn.clicked.connect(self.show_class_graph)
        card_layout.addWidget(class_btn)

        card.setLayout(card_layout)
        main_layout.addWidget(card)
        self.setLayout(main_layout)

    def predict(self):
        try:
            data = pd.DataFrame([{
                'Pclass': int(self.pclass.currentText()),
                'Sex': self.gender.currentText(),
                'Age': float(self.age.text()),
                'SibSp': int(self.sibsp.text()),
                'Parch': int(self.parch.text()),
                'Fare': float(self.fare.text()),
                'Embarked': self.embarked.currentText()
            }])

            prediction = model.predict(data)
            probability = model.predict_proba(data)[0][1]

            if prediction[0] == 1:
                QMessageBox.information(
                    self,
                    "Result",
                    f"🟢 Likely to Survive\nProbability: {round(probability*100,2)}%"
                )
            else:
                QMessageBox.information(
                    self,
                    "Result",
                    f"🔴 Likely Did Not Survive\nProbability: {round(probability*100,2)}%"
                )

        except:
            QMessageBox.warning(self, "Error", "Enter valid numeric values.")

    def show_accuracy_graph(self):
        plt.figure()
        plt.bar(["Accuracy"], [accuracy])
        plt.ylim(0, 1)
        plt.title("Model Accuracy")
        plt.show()

    def show_class_graph(self):
        survival_counts = train.groupby(['Pclass','Survived']).size().unstack()
        survival_counts.plot(kind='bar')
        plt.title("Survival Count per Class")
        plt.ylabel("Number of People")
        plt.show()

    def get_styles(self):
        return """
        QWidget {
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 white,
                stop:1 lightyellow
            );
            font-family: Segoe UI;
        }

        QFrame#card {
            background-color: #6A7B9C;
            border-radius: 25px;
            padding: 35px;
        }

        QLabel#mainTitle {
            color: #00d4ff;
            font-size: 24px;
            font-weight: bold;
        }

        QLineEdit, QComboBox {
            background-color: #112240;
            color: white;
            border: 1px solid #233554;
            border-radius: 10px;
            padding: 10px;
            font-size: 14px;
        }

        QLineEdit:focus, QComboBox:focus {
            border: 2px solid #00d4ff;
        }

        QPushButton {
            background-color: #00b4d8;
            color: white;
            border-radius: 12px;
            font-size: 16px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #0096c7;
        }

        QPushButton:pressed {
            background-color: #e0f2ff;
        }

        QTableWidget {
            background-color: white;
            color: black;
            gridline-color: #cccccc;
        }
        """


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TitanicApp()
    window.show()
    sys.exit(app.exec_())
