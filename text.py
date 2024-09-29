import joblib
from sklearn.metrics import mean_squared_error
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

df = pd.read_csv('VOT.csv')
done = True
t.join()
print(f'\r{task_message} Готово!')

# Преобразование категориальных данных
le = LabelEncoder()
categorical_columns = ['region', 'ua_device_type', 'ua_client_type', 
                       'ua_os', 'ua_client_name']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col].astype(str))
    
# Преобразование колонки 'event_timestamp' в datetime
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

# Преобразование в Unix-время (количество секунд с начала эпохи)
df['event_timestamp'] = df['event_timestamp'].astype('int64') // 10**9  # Переводим в секунды

df['rutube_video_id'] = re.findall(r'\d+', str(df['rutube_video_id']))[0]

X = df[['event_timestamp', 'viewer_uid', 'rutube_video_id', 
         'region', 'ua_device_type', 'ua_client_type', 
         'ua_os', 'ua_client_name', 'total_watchtime']]

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
})

grouped_results = results.groupby('viewer_uid').agg({
    'age': 'median',
    'sex': lambda x: x.mode()[0]
    }).reset_index()

# Округляем возраст после группировки
grouped_results['age'] = grouped_results['age'].round().astype(int)

grouped_results['age_class'] = grouped_results['age'].apply(age_classification)

# Сохранение в CSV файл
grouped_results.to_csv('predictions_results.csv', index=False)

done = True
t.join()
print(f'\r{task_message} Готово!')

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



# Графики с показателями молелей

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Разница между фактическим и предсказанным возрастом
age_error = y_age - y_age_pred

f1_sex = f1_score(y_sex, y_sex_pred, average=None)
categories = ['Female', 'Male']

# Создание DataFrame для визуализации ошибок
errors_df = pd.DataFrame({
    'Actual Age': y_age,
    'Predicted Age': y_age_pred,
    'Age Error': age_error
})

# 1. Распределение ошибки предсказания возраста
plt.figure(figsize=(14, 12))

# Сетка 2x2 для графиков
plt.subplot(2, 2, 1)
sns.histplot(age_error, kde=True, color="blue")
plt.title("Распределение ошибки предсказания возраста")
plt.xlabel("Ошибка предсказания возраста")
plt.ylabel("Количество")

# 2. F1 Score для предсказания пола
plt.subplot(2, 2, 2)
sns.barplot(x=categories, y=f1_sex, palette='pastel')
plt.title('F1 Score для предсказания пола по категориям')
plt.xlabel('Категория')
plt.ylabel('F1 Score')

# 3. Ошибка предсказания возраста в зависимости от фактического возраста
plt.subplot(2, 2, 3)
sns.scatterplot(x='Actual Age', y='Age Error', data=errors_df, color="green")
plt.title("Ошибка предсказания возраста в зависимости от фактического возраста")
plt.xlabel("Фактический возраст")
plt.ylabel("Ошибка предсказания")

# Настройка плотного макета, чтобы графики не перекрывались
plt.tight_layout()

# Сохранение всех графиков в одном файле
plt.savefig('combined_model_plots.png', format='png')

# Отображение графиков в одном окне
plt.show()


# Графики со статистикой

# Задаем размер фигуры и создаем сетку 2x2
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. График со статистикой предсказанных полов
sns.countplot(x=y_sex_pred, palette="pastel", ax=axs[0, 0])
axs[0, 0].set_title("Распределение предсказанных полов")
axs[0, 0].set_xlabel("Пол (0 - женский, 1 - мужской)")
axs[0, 0].set_ylabel("Количество")

# 2. График распределения ошибок предсказания возраста
sns.histplot(age_error, kde=True, color="blue", ax=axs[0, 1])
axs[0, 1].set_title("Распределение ошибок предсказания возраста")
axs[0, 1].set_xlabel("Ошибка предсказания возраста")
axs[0, 1].set_ylabel("Количество")

# 3. График распределения пользователей по возрастным классам
sns.countplot(x=grouped_results['age_class'], palette='pastel', ax=axs[1, 0])
axs[1, 0].set_title('Распределение пользователей по возрастным классам')
axs[1, 0].set_xlabel('Возрастной класс')
axs[1, 0].set_ylabel('Количество')

# 4. График распределения пользователей по полу
sns.countplot(x=grouped_results['sex'], palette='pastel', ax=axs[1, 1])
axs[1, 1].set_title('Распределение пользователей по полу')
axs[1, 1].set_xlabel('Пол (0 - женский, 1 - мужской)')
axs[1, 1].set_ylabel('Количество')

# Настройка макета для предотвращения наложения графиков
plt.tight_layout()

# Сохранение графиков в один файл
plt.savefig('combined_plots.png', format='png')

# Показ всех графиков в одном окне
plt.show()