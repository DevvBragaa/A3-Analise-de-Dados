import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
train_data = pd.read_csv('venv\\Include\\train.csv')
test_data = pd.read_csv('venv\\Include\\test.csv')

print(train_data.head())
print(train_data.info())
print(train_data.describe())

print(train_data.isnull().sum())


sns.countplot(x='Survived', data=train_data)
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.show()


# Imputar idade faltante com a mediana
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])

# Imputar valores faltantes em 'Embarked' e 'Fare'
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)


# Codificar sexo e embarque
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])

train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])


train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

train_data['IsAlone'] = (train_data['FamilySize'] == 0).astype(int)
test_data['IsAlone'] = (test_data['FamilySize'] == 0).astype(int)

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f'Melhores parâmetros: {grid_search.best_params_}')


best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)


y_test_pred = best_model.predict(X_test)


submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_test_pred
})
submission.to_csv('submission.csv', index=False)


