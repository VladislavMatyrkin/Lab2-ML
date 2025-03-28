import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_new.csv")

print(df.head())

df = df.drop(columns=['PassengerId', 'Name', 'Cabin'])
df = df.dropna()

# Преобразуем целевой признак в числовой формат
df['Transported_True'] = df['Transported_True'].astype(int)

# Разделение данных для классификации
X_class = df.drop(columns=['Transported_True'])#Матрица признаков (все столбцы, кроме Transported_True).
y_class = df['Transported_True']#Целевая переменная

# Разделение данных для регрессии(предсказание возраста)
X_reg = df.drop(columns=['Age'])
y_reg = df['Age']

# Делим данные на обучающие и тестовые выборки
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Классификация
logistic_model = LogisticRegression(max_iter=5000)
logistic_model.fit(X_class_train, y_class_train)#обучаем модель на обучающих данных
# Предсказания
y_class_pred = logistic_model.predict(X_class_test)#предсказание на тестовых данных
# Метрики
precision_logistic = precision_score(y_class_test, y_class_pred)
print(f'Precision (логистическая регрессия): {precision_logistic:.2f}')
print("Отчет классификации (логистическая регрессия):")
print(classification_report(y_class_test, y_class_pred))
# Матрица ошибок
cm_class = confusion_matrix(y_class_test, y_class_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_class, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок - Логистическая регрессия')
plt.show()

# Регрессия
linear_model = LinearRegression()
linear_model.fit(X_reg_train, y_reg_train)
# Предсказания
y_reg_pred = linear_model.predict(X_reg_test)
# Метрики
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f'Mean Squared Error: {mse:.2f}')#Среднеквадратичная ошибка (чем меньше, тем лучше).
print(f'R2 Score: {r2:.2f}')#Коэффициент детерминации (показывает, насколько хорошо модель объясняет данные).
# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_reg_pred, alpha=0.6)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], '--r', linewidth=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные значения')
plt.title('Регрессия - Истинные vs Предсказанные значения')
plt.show()


# Улучшение моделей

#Нормализацию данных (скалирование)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_class_scaled = scaler.fit_transform(X_class)
X_reg_scaled = scaler.fit_transform(X_reg)

# Переразделим данные с масштабированием
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class_scaled, y_class, test_size=0.2, random_state=42)
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg_scaled, y_reg, test_size=0.2, random_state=42)

# Повторим обучение моделей на масштабированных данных
logistic_model.fit(X_class_train, y_class_train)
linear_model.fit(X_reg_train, y_reg_train)

# Перепроверим метрики классификации
y_class_pred = logistic_model.predict(X_class_test)
precision_logistic = precision_score(y_class_test, y_class_pred)
print(f'Precision (логистическая регрессия с масштабированием): {precision_logistic:.2f}')

# Перепроверим метрики регрессии
y_reg_pred = linear_model.predict(X_reg_test)
mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)
print(f'Mean Squared Error (с масштабированием): {mse:.2f}')
print(f'R2 Score (с масштабированием): {r2:.2f}')

#использовать другие алгоритмы
#значение R^2 может быть маленькое из-за невысокой зависимости между признаками и целевой переменной
#маштабирование возможно слегка уменьшила значения,потому что изначально признаки были в схожих масштабах