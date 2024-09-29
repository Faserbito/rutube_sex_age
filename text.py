# import joblib
# from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
# import pandas as pd
# import re

# # Определение классов возраста
# def age_classification(age):
#     if age <= 20:
#         return 0
#     elif age <= 30:
#         return 1
#     elif age <= 40:
#         return 2
#     elif age <= 60:
#         return 3
#     else:
#         return None  # или любое другое значение, чтобы обозначить возраст вне диапазона

# #----------------------------------------------------------------
# print('Я считываю данные...')
# df = pd.read_csv('sorted_output.csv')

# from sklearn.preprocessing import LabelEncoder

# # Преобразование категориальных данных
# le = LabelEncoder()
# categorical_columns = ['viewer_uid', 'region', 'ua_device_type', 'ua_client_type', 
#                        'ua_os', 'ua_client_name', 'category']

# for col in categorical_columns:
#     df[col] = le.fit_transform(df[col].astype(str))
    
# # Преобразование колонки 'event_timestamp' в datetime
# df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

# # Преобразование в Unix-время (количество секунд с начала эпохи)
# df['event_timestamp'] = df['event_timestamp'].astype('int64') // 10**9  # Переводим в секунды

# df['rutube_video_id'] = re.findall(r'\d+', str(df['rutube_video_id']))[0]
    
# X = df[['event_timestamp', 'viewer_uid', 'rutube_video_id', 
#          'region', 'ua_device_type', 'ua_client_type', 
#          'ua_os', 'ua_client_name', 'total_watchtime', 
#          'duration', 'author_id', 'category']]

# y_sex = df['sex']
# y_age = df['age']

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_sex_train, y_sex_test = train_test_split(X, y_sex, test_size=1, random_state=42)
# X_train, X_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=1, random_state=42)
# #----------------------------------------------------------------

# # Загрузка моделей
# print('Я загружаю модели')
# model_sex = joblib.load('model_sex.pkl')
# model_age = joblib.load('model_age.pkl')

# print('Я предсказываю пол')
# # Предсказания для пола
# y_sex_pred = model_sex.predict(X_test)

# print('Я предсказываю возраст')
# # Предсказания для возраста
# y_age_pred = model_age.predict(X_test)

# print('Я сортирую данные')
# viewer_uid_text = df.loc[X_test.index, 'viewer_uid']

# # Округление возраста
# y_age_pred_rounded = [round(age) for age in y_age_pred]

# # Применение классификации к предсказанному возрасту
# age_classes = [age_classification(age) for age in y_age_pred]

# # Создание итогового DataFrame
# results = pd.DataFrame({
#     'viewer_uid': viewer_uid_text,
#     'age': y_age_pred,
#     'sex': y_sex_pred,
#     'age_class': age_classes
# })

# grouped_results = results.groupby('viewer_uid').agg({
#     'age': 'mean',
#     'sex': lambda x: x.mode()[0],
#     'age_class': lambda x: x.mode()[0]
#     }).reset_index()

# # Округляем возраст после группировки
# grouped_results['age'] = grouped_results['age'].round().astype(int)

# # Сохранение в CSV файл
# grouped_results.to_csv('predictions_results.csv', index=False)

# print('Результаты сохранены в predictions_results.csv')

# # # Вывод первых 10 строк результата
# # print(results.head(10))

# from sklearn.metrics import f1_score, mean_squared_error

# # Оценка модели sex по F1 метрике
# f1_sex = f1_score(y_sex_test, y_sex_pred, average='weighted')  # 'weighted' для учета несбалансированных классов
# print(f'F1-score for sex prediction: {f1_sex}')

# # Оценка модели age по метрике среднеквадратичной ошибки (MSE)
# mse_age = mean_squared_error(y_age_test, y_age_pred)
# print(f'Mean Squared Error (MSE) for age prediction: {mse_age}')



import joblib
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error
import pandas as pd
import re
import itertools
import time
import threading

import seaborn as sns
import matplotlib.pyplot as plt

# Стиль seaborn для более красивых графиков
sns.set(style="whitegrid")

# Определение классов возраста
def age_classification(age):
    if 9 < age <= 20:
        return 0
    elif 20 < age <= 30:
        return 1
    elif 30 < age <= 40:
        return 2
    elif 40 < age <= 60:
        return 3
    else:
        return None

# Функция для отображения крутящегося индикатора
def spinning_cursor():
    for cursor in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        print(f'\r{task_message} {cursor}', end='')
        time.sleep(0.1)

#----------------------------------------------------------------
# Считывание данных
task_message = "Я считываю данные..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

df = pd.read_csv('sorted_output.csv')
done = True
t.join()
print(f'\r{task_message} Готово!')

# Преобразование категориальных данных
le = LabelEncoder()
categorical_columns = ['region', 'ua_device_type', 'ua_client_type', 
                       'ua_os', 'ua_client_name', 'category']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))
    
# Преобразование колонки 'event_timestamp' в datetime
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

# Преобразование в Unix-время (количество секунд с начала эпохи)
df['event_timestamp'] = df['event_timestamp'].astype('int64') // 10**9  # Переводим в секунды

df['rutube_video_id'] = re.findall(r'\d+', str(df['rutube_video_id']))[0]

X = df[['event_timestamp', 'viewer_uid', 'rutube_video_id', 
         'region', 'ua_device_type', 'ua_client_type', 
         'ua_os', 'ua_client_name', 'total_watchtime', 
         'duration', 'author_id', 'category']]

y_sex = df['sex']
y_age = df['age']

#----------------------------------------------------------------
# Загрузка моделей
task_message = "Я загружаю модели..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

model_sex = joblib.load('model_sex.pkl')
model_age = joblib.load('model_age.pkl')

done = True
t.join()
print(f'\r{task_message} Готово!')

# Предсказания для пола
task_message = "Я предсказываю пол..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

y_sex_pred = model_sex.predict(X)

done = True
t.join()
print(f'\r{task_message} Готово!')

# Предсказания для возраста
task_message = "Я предсказываю возраст..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

y_age_pred = model_age.predict(X)

done = True
t.join()
print(f'\r{task_message} Готово!')

# Сортировка данных
task_message = "Я сортирую данные..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

# Округление возраста
y_age_pred_rounded = [round(age) for age in y_age_pred]

# Создание итогового DataFrame
results = pd.DataFrame({
    'viewer_uid': df.loc[X.index, 'viewer_uid'],
    'age': y_age_pred,
    'sex': y_sex_pred
    # 'age_class': age_classes
})

grouped_results = results.groupby('viewer_uid').agg({
    'age': 'median',
    'sex': lambda x: x.mode()[0]
    # 'age_class': lambda x: x.mode()[0]
    }).reset_index()

# Округляем возраст после группировки
grouped_results['age'] = grouped_results['age'].round().astype(int)

grouped_results['age_class'] = grouped_results['age'].apply(age_classification)

# Сохранение в CSV файл
grouped_results.to_csv('predictions_results.csv', index=False)

done = True
t.join()
print(f'\r{task_message} Готово!')

# # Вывод первых 10 строк результата
# print(results.head(10))

# Оценка модели sex по F1 метрике
task_message = "Я оцениваю модель для пола..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

f1_sex = f1_score(y_sex, y_sex_pred, average='weighted')
done = True
t.join()
print(f'\rF1-score for sex prediction: {f1_sex}')

# Оценка модели age по метрике среднеквадратичной ошибки (MSE)
task_message = "Я оцениваю модель для возраста..."
done = False
t = threading.Thread(target=spinning_cursor)
t.start()

mse_age = mean_squared_error(y_age, y_age_pred)
done = True
t.join()
print(f'\rMean Squared Error (MSE) for age prediction: {mse_age}')

# Разница между фактическим и предсказанным возрастом
age_error = y_age - y_age_pred

plt.figure(figsize=(10, 6))
sns.histplot(age_error, kde=True, color="blue")
plt.title("Распределение ошибки предсказания возраста")
plt.xlabel("Ошибка предсказания возраста")
plt.ylabel("Количество")
# plt.show()
plt.savefig('age_prediction_error_distribution.jpg', format='jpg')
plt.close()

plt.figure(figsize=(8, 6))
sns.countplot(x=y_sex_pred, palette="pastel")
plt.title("Распределение предсказанных полов")
plt.xlabel("Пол (0 - женский, 1 - мужской)")
plt.ylabel("Количество")
# plt.show()
plt.savefig('predicted_sex_distribution.jpg', format='jpg')
plt.close()

# Создание DataFrame для визуализации ошибок
errors_df = pd.DataFrame({
    'Actual Age': y_age,
    'Predicted Age': y_age_pred,
    'Age Error': age_error
})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Age', y='Age Error', data=errors_df, color="green")
plt.title("Ошибка предсказания возраста в зависимости от фактического возраста")
plt.xlabel("Фактический возраст")
plt.ylabel("Ошибка предсказания")
# plt.show()
plt.savefig('age_error_vs_actual_age.jpg', format='jpg')
plt.close()