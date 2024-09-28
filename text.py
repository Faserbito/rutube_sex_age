import joblib
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error

# Загрузка моделей
print('Я модели загружаю')
model_gender = joblib.load('model_sex.pkl')
model_age = joblib.load('model_age.pkl')
print('Я модели загрузил')

# Предсказания для пола
print('Я предсказываю пол')
y_gender_pred = model_gender.predict(X_test)

# Предсказания для возраста
print('Я предсказываю возраст')
y_age_pred = model_age.predict(X_test)

# Метрики для классификации пола
print('Я точность считаю')
accuracy_gender = accuracy_score(y_gender_test, y_gender_pred)
print(f'Accuracy for gender prediction: {accuracy_gender}')
print(classification_report(y_gender_test, y_gender_pred))

# Метрики для регрессии возраста
print('Я да')
mse_age = mean_squared_error(y_age_test, y_age_pred)
mae_age = mean_absolute_error(y_age_test, y_age_pred)
print(f'Mean Squared Error for age prediction: {mse_age}')
print(f'Mean Absolute Error for age prediction: {mae_age}')






# import pandas as pd

# # Предсказания для пола
# y_gender_pred = model_gender.predict(X_test)

# # Предсказания для возраста
# y_age_pred = model_age.predict(X_test)

# # Объединение фактических и предсказанных данных
# results = pd.DataFrame({
#     'Actual Gender': y_gender_test,  # Фактический пол
#     'Predicted Gender': y_gender_pred,  # Предсказанный пол
#     'Actual Age': y_age_test,  # Фактический возраст
#     'Predicted Age': y_age_pred  # Предсказанный возраст
# })

# # Вывод первых 10 строк результата
# print(results.head(10))