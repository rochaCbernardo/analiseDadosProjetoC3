# analiseDadosProjetoC3
Esse repositório contém uma análise do DataFrame house_prices([house_train.csv](house_train.csv)), disponível no Kaggle.

## Organização:
* [house_train](house_train.csv): Arquivo utilizado para a criação do DataFrame, df_house_train; 
* [analiseC3](analiseC3.ipynb): Aqui constam as análises realizadas e os modelos criados.
  - Em um primeiro momento, foi feita a análise exploratória dos dados para entender melhor a disposição dos dados e as características:
    - Verificando a quantidade de linhas e colunas do df;
      ```python
      df_house_train.shape
      ```
    - Utilizado para saber qual o nome das colunas;
      ```python
      df_house_train.columns
      ```
    - Mais informações do df, como a quantidade de valores não nulos, o tipo dos atributos e suas quantidades;
      ```python
      df_house_train.info()
      ```
    - Utilizado o comando para identificar a quantidade de valores nulos por atributo, para verificar quais seriam excluídos ou complementados;
      ```python
      df_house_train.isnull().sum()
      ```
    - Trecho criado para verificar os valores possíveis dos atributos object e quantidade de cada;
      ```python
      object_columns = df_house_train.select_dtypes(include='object').columns
      for col in object_columns:
      print(f"Análise da coluna: {col}")
      print(f"Valores assumidos:\n {df_house_train[col].value_counts()}")
      ```
  - Após verificar as características dos dados, foi feita a engenharia de características para selecionar as variáveis mais importantes:
      - Exclusão das colunas que possuíam mais de 70% dos valores nulos;
        ```python
        df_house_train = df_house_train.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']).copy()
        ```
    - Utilizando o MinMaxScaler para transformar os valores int e float em valores entre 0 e 1;
      ```python
      scaler_minMax = MinMaxScaler()
      numList = df_house_train.select_dtypes(include=[np.float64, np.int64]).columns

      df_house_train[numList] = pd.DataFrame(scaler_minMax.fit_transform(df_house_train[numList]))
      ```
    - Utilizando o LabelEnconder para transformar os atributos do tipo 'object' em números;
      ```python
      lb = LabelEncoder()
      objList = df_house_train.select_dtypes(include='object').columns
      
      for obj in objList:
        df_house_train[obj] = lb.fit_transform(df_house_train[obj].astype(str))
        ```
    - Preenchendo as demais colunas que possuem valores nulos com a média dos valores encontrados;
      ```python
      null_values = ['LotFrontage','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
      df_house_train.fillna(df_house_train[null_values].mean(), inplace=True)
      ```
    - Criação do atributo 'price_reg' para armazenar a previsão feita pelo modelo de Regressão Linear sobre o preço de venda das casas;
      ```python
      df_house_train['price_reg']=None
      ```
  - Aprendizagem supervisionada para prever o preço de venda de uma casa e se o valor será baixo ou alto:
    - Utilizando o LinearRegression, regressão linear, conseguimos prever o valor de venda das casas;
      ```python
      predictors = df_house_train[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',
       'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
       'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
       'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition', 'price_reg']]
      outcome = df_house_train['SalePrice']

      modelReg = LinearRegression()
      modelReg.fit(predictors, outcome)

      y_pred = modelReg.predict(predictors)

      df_house_train['price_reg'] = y_pred
      ```
    - Utilização do comando astype, para transformar a coluna 'price_binary' em uma saída binária de acordo com o valor predito, se df_house_train['price_reg']>0.2 a saída será = 1 e se não será = 0;
      ```python
      df_house_train['price_binary'] = (df_house_train['price_reg']>0.2).astype(int)
      ```
    - A partir da LogisticRegression, regressão logística, podemos prever o preço da venda da casa ser alto ou baixo, utilizando 20% do df para teste;
        - Precision de 96%, concluímos que dos dados utilizados para teste, 96% classificado como preço alto é realmente preço alto;
        - Recall de 95%, podemos concluir que o modelo está indentificando corretamente 95% dos exemplos positivos, valores altos, no exemplo de teste;
        - Obtivemos uma AUC de 96%, configurando a classificação correta de positivos e negativos 96% das vezes, sendo o modelo uma melhor escolha do que uma escolha aleatória;
          ```python
          x_logReg = df_house_train[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
           'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
           'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrArea', 'ExterQual',
           'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
           '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
           'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
           'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
           'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
           'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',
           'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition', 'price_reg']]
          y_logReg = df_house_train['price_binary']

          x_logReg_train, x_logReg_test, y_logReg_train, y_logReg_test = train_test_split(x_logReg, y_logReg, test_size=0.2, random_state = 42)

          modelLogReg = LogisticRegression(max_iter=1000)

          modelLogReg.fit(x_logReg_train, y_logReg_train)

          y_logReg_pred = modelLogReg.predict(x_logReg_test)

          precision_logReg = precision_score(y_logReg_test, y_logReg_pred)
          recall_logReg = recall_score(y_logReg_test, y_logReg_pred)
          auc_logReg = roc_auc_score(y_logReg_test, y_logReg_pred)

          print(f'Precision: {precision_logReg: .2f}')
          print(f'Recall: {recall_logReg: .2f}')
          print(f'AUC: {auc_logReg: .2f}')
          ```
      - Por fim, utilizamos a matriz confusão, nela podemos verificar que tivemos 158 verdadeiros negativos, ou seja, temos 158 valores baixos que foram classificados como baixo e 122 verdadeiros positivos, que foram altos e foram classificados como altos;
        ```python
        print(pd.crosstab(y_logReg_test, y_logReg_pred, rownames=['Real'], colnames=['Predito'], margins=True))
        ```
