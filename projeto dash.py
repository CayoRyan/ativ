import pandas as pd

df = pd.read_csv('train.csv')

# Ver as primeiras linhas do dataframe
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Contagem de valores nulos
print(df.isnull().sum())

# Distribuição dos preços
df['SalePrice'].hist()
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Preços de Casas"),
    dcc.Graph(
        id="price-dist",
        figure={
            'data': [
                {'x': df['SalePrice'], 'type': 'histogram', 'name': 'Distribuição de Preços'},
            ],
            'layout': {
                'title': 'Distribuição de Preços das Casas'
            }
        }
    ),
    dcc.Graph(
        id="feature-importance",
        figure={
            'data': [
                {'x': features, 'y': model.coef_, 'type': 'bar', 'name': 'Importância das Características'},
            ],
            'layout': {
                'title': 'Importância das Características no Modelo'
            }
        }
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True)

O nome dos integrantes da equipe.
-Augusto Dutra e Cayo Ryan 

O link do repositório escolhido do Kaggle.
-https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

O link do Github do projeto.
-https://github.com/CayoRyan/ativ/blob/main/projeto%20dash.py

