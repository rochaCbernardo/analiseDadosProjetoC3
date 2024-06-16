# analiseDadosProjetoC3
Esse repositório contém uma análise do DataFrame house_prices([house_train.csv](house_train)), disponível no Kaggle.

## Organização:
* [house_train.csv](house_train): Arquivo utilizado para a criação do DataFrame, df_house_train; 
* Colocar nome do arquivo: Aqui constam as análises realizadas e os modelos criados.
  - Em um primeiro momento, foi feita a análise exploratória dos dados para entender melhor a disposição dos dados e as características;
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
  - Após verificar as características dos dados, foi feita a engenharia de características para selecionar as variáveis mais importantes;
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
