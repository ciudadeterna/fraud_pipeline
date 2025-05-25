import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv('creditcard_sample.csv')


X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('diego')
model = RandomForestClassifier()
model.fit(X_train, y_train)
print('maradona')

try:
    joblib.dump(model, 'fraud_model.pkl')
    print('zizou')
except Exception as e:
    print('neymar')    
