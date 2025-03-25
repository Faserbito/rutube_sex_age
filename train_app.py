import pandas as pd
import re

# Загрузка данных
df = pd.read_csv('sorted_output.csv')

from sklearn.preprocessing import LabelEncoder

# Преобразование категориальных данных
le = LabelEncoder()
categorical_columns = ['region', 'ua_device_type', 'ua_client_type', 
                       'ua_os', 'ua_client_name', 'category']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    
# Преобразование колонки 'event_timestamp' в datetime
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

# Преобразование в Unix-время (количество секунд с начала эпохи)
df['event_timestamp'] = df['event_timestamp'].astype('int64') // 10**9  # Переводим в секунды

df['rutube_video_id'] = re.findall(r'\d+', str(df['rutube_video_id']))[0]
    
X = df[['event_timestamp', 'viewer_uid', 'rutube_video_id', 
         'region', 'ua_device_type', 'ua_client_type', 
         'ua_os', 'ua_client_name', 'total_watchtime', 
         'duration', 'author_id', 'category']]

# Предположим, что 'gender' и 'age' - это ваши целевые переменные
y_gender = df['sex']
y_age = df['age']

from sklearn.model_selection import train_test_split
X_train_gender, X_test_gender, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)
X_train_age, X_test_age, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_squared_error

# Модель для предсказания пола
model_gender = RandomForestClassifier(n_estimators=300, random_state=42)
model_gender.fit(X_train_gender, y_gender_train)

y_gender_pred = model_gender.predict(X_test_gender)

f1_gender = f1_score(y_gender_test, y_gender_pred, average='weighted')  # 'weighted' для учета несбалансированных классов
print(f'F1-score for gender prediction: {f1_gender}')

# Оценка модели
# accuracy_gender = model_gender.score(X_test_gender, y_gender_test)
# print(f'Accuracy for gender prediction: {accuracy_gender}')

# Модель для предсказания возраста
model_age = RandomForestRegressor(n_estimators=300, random_state=42)
model_age.fit(X_train_age, y_age_train)

y_age_pred = model_age.predict(X_test_age)

# Оценка модели по метрике среднеквадратичной ошибки (MSE)
mse_age = mean_squared_error(y_age_test, y_age_pred)
print(f'Mean Squared Error (MSE) for age prediction: {mse_age}')

# Оценка модели
# accuracy_age = model_age.score(X_test_age, y_age_test)
# print(f'Accuracy for age prediction: {accuracy_age}')

import joblib

joblib.dump(model_gender, 'model_sex.pkl')
joblib.dump(model_age, 'model_age.pkl')











# import pandas as pd

# # Установка максимального количества выводимых строк (например, 50)
# pd.set_option('display.max_rows', None)

# # Установка вывода всех столбцов
# pd.set_option('display.max_columns', None)

# # Чтение двух CSV файлов
# df1 = pd.read_csv('train_targets.csv')
# df2 = pd.read_csv('train_events.csv') 
# df3 = pd.read_csv('video_info_v2.csv', encoding='utf-8')

# # Просмотр первых 5 строк отсортированного DataFrame
# # print(df3.head(50))

# # Объединение по ключу (например, по столбцу 'key_column')
# df_merged = pd.merge(df1, df2, on='viewer_uid')
# df_merged = pd.merge(df_merged, df3, on='rutube_video_id')

# # Сортировка по столбцу 'column_name'
# df_sorted = df_merged.sort_values(by='viewer_uid')

# #print(df_sorted.head(50))

# # Сохранение результата в новый CSV файл
# df_sorted.to_csv('sorted_output.csv', index=False,  encoding='utf-8-sig')

