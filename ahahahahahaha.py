import pandas as pd
import re
import json

# �������� ������
df = pd.read_csv('sorted_output.csv')

from sklearn.preprocessing import LabelEncoder

# �������������� �������������� ������
le = LabelEncoder()
categorical_columns = ['region', 'ua_device_type', 'ua_client_type', 
                       'ua_os', 'ua_client_name', 'category']

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
    
# �������������� ������� 'event_timestamp' � datetime
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

# �������������� � Unix-����� (���������� ������ � ������ �����)
df['event_timestamp'] = df['event_timestamp'].astype('int64') // 10**9  # ��������� � �������

df['rutube_video_id'] = re.findall(r'\d+', str(df['rutube_video_id']))[0]
    
X = df[['event_timestamp', 'viewer_uid', 'rutube_video_id', 
         'region', 'ua_device_type', 'ua_client_type', 
         'ua_os', 'ua_client_name', 'total_watchtime', 
         'duration', 'author_id', 'category']]

# �����������, ��� 'gender' � 'age' - ��� ���� ������� ����������
y_gender = df['sex']
y_age = df['age']

from sklearn.model_selection import train_test_split

X_train, X_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, test_size=0.1, random_state=42)
X_train, X_test, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.1, random_state=42)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ������ ��� ������������ ����
model_gender = RandomForestClassifier(n_estimators=100, random_state=42)
model_gender.fit(X_train, y_gender_train)

# ������ ������
accuracy_gender = model_gender.score(X_test, y_gender_test)
print(f'Accuracy for gender prediction: {accuracy_gender}')

# ������ ��� ������������ ��������
model_age = RandomForestRegressor(n_estimators=100, random_state=42)
model_age.fit(X_train, y_age_train)

# ������ ������
accuracy_age = model_age.score(X_test, y_age_test)
print(f'Accuracy for age prediction: {accuracy_age}')











# import pandas as pd

# # ��������� ������������� ���������� ��������� ����� (��������, 50)
# pd.set_option('display.max_rows', None)

# # ��������� ������ ���� ��������
# pd.set_option('display.max_columns', None)

# # ������ ���� CSV ������
# df1 = pd.read_csv('train_targets.csv')
# df2 = pd.read_csv('train_events.csv')
# df3 = pd.read_csv('video_info_v2.csv', encoding='utf-8')

# # �������� ������ 5 ����� ���������������� DataFrame
# # print(df3.head(50))

# # ����������� �� ����� (��������, �� ������� 'key_column')
# df_merged = pd.merge(df1, df2, on='viewer_uid')
# df_merged = pd.merge(df_merged, df3, on='rutube_video_id')

# # ���������� �� ������� 'column_name'
# df_sorted = df_merged.sort_values(by='viewer_uid')

# #print(df_sorted.head(50))

# # ���������� ���������� � ����� CSV ����
# df_sorted.to_csv('sorted_output.csv', index=False,  encoding='utf-8-sig')

