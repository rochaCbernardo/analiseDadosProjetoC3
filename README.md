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
  - Aprendizagem supervisionada:
    - Clusterização: vamos garantir que os valores ausentes sejam imputados corretamente e depois aplicar o K-Means;
      ```python
      from sklearn.preprocessing import StandardScaler, OneHotEncoder
      from sklearn.compose import ColumnTransformer
      from sklearn.pipeline import Pipeline
      from sklearn.impute import SimpleImputer

      print(df_house_train.head())
      print(df_house_train.info())

      numerical_features = df_house_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
      categorical_features = df_house_train.select_dtypes(include=['object']).columns.tolist()

      if 'SalePrice' in numerical_features:
        numerical_features.remove('SalePrice')
      if 'SalePrice' in categorical_features:
        categorical_features.remove('SalePrice')

      numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

      categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

      preprocessor = ColumnTransformer(
      transformers=[
        ('num', numerical_pipeline, numerical_features), ('cat', categorical_pipeline, categorical_features)])

      data_prepared = preprocessor.fit_transform(df_house_train)
      ```
    - Aplicação do K-Means:
      - Definimos o númeo de clusters;
      - Usamos K-Means para ajustar o modelo e prever os clusters;
      - Adicionamos a informação de cluster ao dataframe original;
      - Utilizamos PCA para reduzir dimensionalidade dos dados preparados;
        ```python
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.decomposition import PCA

        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        data_clusters = kmeans.fit_predict(data_prepared)

        df_house_train['Cluster'] = data_clusters

        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data_prepared.toarray() if hasattr(data_prepared, 'toarray') else data_prepared)

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], hue=data_clusters, palette='viridis')
        plt.title('Clusterização de Casas')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend(title='Cluster')
        plt.show()
      ```
    - Avaliação modo de clusterirazação (Precision, Recall e F1-Score);
      ```python
      np.random.seed(0)
      df_house_train['TrueLabels'] = np.random.choice(n_clusters, len(df_house_train))

      label_encoder = LabelEncoder()
      true_labels = label_encoder.fit_transform(df_house_train['TrueLabels'])

      cluster_to_true_label = {}

      for cluster in range(n_clusters):
        mask = df_house_train['Cluster'] == cluster
        most_common_label = df_house_train.loc[mask, 'TrueLabels'].mode()
        if not most_common_label.empty:
          cluster_to_true_label[cluster] = most_common_label.values[0]

      predicted_labels = df_house_train['Cluster'].map(cluster_to_true_label)

      precision = precision_score(true_labels, predicted_labels, average='macro')
      recall = recall_score(true_labels, predicted_labels, average='macro')
      f1 = f1_score(true_labels, predicted_labels, average='macro')


      print(f"Precision: {precision}")
      print(f"Recall: {recall}")
      print(f"F1-Score: {f1}")

      conf_matrix = confusion_matrix(true_labels, predicted_labels)
      print(f"Confusion Matrix:\n{conf_matrix}")
      ```
    - Redução de dimensionalidade:
      - Utilizamos 'simpleimputer' para tratar valores ausentes;
      - Aplicamos o transformador para obter dados preparados e verificarmos a sua forma;
        ```python
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        print(df_house_train.head())
        print(df_house_train.info())

        numerical_features = df_house_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df_house_train.select_dtypes(include=['object']).columns.tolist()

        if 'SalePrice' in numerical_features:
          numerical_features.remove('SalePrice')
        if 'SalePrice' in categorical_features:
          categorical_features.remove('SalePrice')

        numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

        categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))  # Codificar variáveis categóricas])

        preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_features),('cat', categorical_pipeline, categorical_features)])

        data_prepared = preprocessor.fit_transform(df_house_train)

        print(data_prepared.shape)
        ```
    - Utilizamos PCA para reduzir os dados para 2 componentes principais e plotamos os resultados em um gráfico de dispersão;
      ```python
      pca = PCA(n_components=2)
      data_pca = pca.fit_transform(data_prepared.toarray() if hasattr(data_prepared, 'toarray') else data_prepared)

      plt.figure(figsize=(10, 6))
      plt.scatter(data_pca[:, 0], data_pca[:, 1], alpha=0.5)
      plt.title('Redução de Dimensionalidade com PCA')
      plt.xlabel('Componente Principal 1')
      plt.ylabel('Componente Principal 2')
      plt.show()
      ```
    - Avaliação do modo de redução (Precision, Recall e F1-Score);
      ```python
      pca = PCA(n_components=2)
      data_pca = pca.fit_transform(data_prepared.toarray() if hasattr(data_prepared, 'toarray') else data_prepared)

      np.random.seed(0)
      df_house_train['TrueLabels'] = np.random.choice([0, 1], len(df_house_train))

      X_train, X_test, y_train, y_test = train_test_split(data_pca, df_house_train['TrueLabels'], test_size=0.3, random_state=42)

      classifier = LogisticRegression()
      classifier.fit(X_train, y_train)

      #Fazer previsões
      y_pred = classifier.predict(X_test)

      #Calcular métricas
      precision = precision_score(y_test, y_pred)
      recall = recall_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      
      #Plotar os dados reduzidos com rótulos verdadeiros para visualização
      plt.figure(figsize=(10, 6))
      plt.scatter(data_pca[:, 0], data_pca[:, 1], c=df_house_train['TrueLabels'], alpha=0.5, cmap='viridis')
      plt.title('Redução de Dimensionalidade com PCA')
      plt.xlabel('Componente Principal 1')
      plt.ylabel('Componente Principal 2')
      plt.colorbar(label='True Labels')
      plt.show()
      ```
    - Apriori:
      - Verificamos se as colunas categóricas foram identificadas corretamente;
      - Utilizamos SimpleImputer para tratar valores ausentes (valor mais frequente para categóricas);
      - Utilizamos OneHotEncoder para codificar as colunas categóricas em formato binário;
      - Criamos um dataframe categorical_df com os dados categóricos processados;
        ```python
        selected_categorical_features = ['MSZoning', 'Street', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Foundation', 'Heating', 'CentralAir']

        df_selected = df_house_train[selected_categorical_features]

        categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        categorical_data_prepared = categorical_pipeline.fit_transform(df_selected)

        categorical_df = pd.DataFrame(categorical_data_prepared, columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(selected_categorical_features))

        print(categorical_df.head())
        print(categorical_df.shape)
        ```
    - Utilizamos a função apriori da biblioteca mlxtend para identificar conjuntos frequentes de itens com um suporte mínimo de 5%;
    - Geramos regras de associação usando a função association_rules e filtramos as regras com um lift mínimo de 1.0;
    - Exibimos e ordenamos as regras de associação para visualizar as mais interessantes.
      ```python
      from mlxtend.frequent_patterns import apriori, association_rules

      frequent_itemsets = apriori(categorical_df, min_support=0.1, use_colnames=True)

      rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

      print(rules)

      rules = rules.sort_values(by='lift', ascending=False)
      print(rules.head())
      ```
    - Local Outlier Factor:
      - Verificamos se as colunas numéricas e categóricas foram identificadas corretamente;
      - Utilizamos SimpleImputer para tratar valores ausentes (média para variáveis numéricas e valor mais frequente para categóricas);
      - Utilizamos StandardScaler para escalar as colunas numéricas e OneHotEncoder para codificar as colunas categóricas;
      - Aplicamos o transformador para obter os dados preparados e verificamos sua forma;
        ```python
        numerical_features = df_house_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df_house_train.select_dtypes(include=['object']).columns.tolist()

        if 'SalePrice' in numerical_features:
          numerical_features.remove('SalePrice')
        if 'SalePrice' in categorical_features:
          categorical_features.remove('SalePrice')


        numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])

        categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

        preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_features),('cat', categorical_pipeline, categorical_features)])

        data_prepared = preprocessor.fit_transform(df_house_train)

        print(data_prepared.shape)
        ```
    - Usamos LocalOutlierFactor para detectar outliers no dataset. Definimos n_neighbors=20 e contamination=0.05 (assumindo que 5% dos dados são outliers);
    - Adicionamos a informação de outliers ao dataframe original (df_house_train);
    - Plotamos a distribuição dos scores de outliers;
    - Visualizamos os outliers identificados em um gráfico de dispersão comparando GrLivArea e SalePrice;
      ```python
      from sklearn.neighbors import LocalOutlierFactor

      lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
      outlier_scores = lof.fit_predict(data_prepared)

      df_house_train['Outlier'] = outlier_scores

      print("Número de outliers detectados:", np.sum(outlier_scores == -1))

      plt.figure(figsize=(10, 6))
      plt.hist(outlier_scores, bins=50, alpha=0.75, edgecolor='black')
      plt.title('Distribuição dos Scores de Outliers')
      plt.xlabel('Score de Outlier')
      plt.ylabel('Frequência')
      plt.show()

      plt.figure(figsize=(10, 6))
      outliers = df_house_train[outlier_scores == -1]
      non_outliers = df_house_train[outlier_scores != -1]
      plt.scatter(non_outliers['GrLivArea'], non_outliers['SalePrice'], c='blue', label='Non-Outliers', alpha=0.5)
      plt.scatter(outliers['GrLivArea'], outliers['SalePrice'], c='red', label='Outliers', alpha=0.5)
      plt.xlabel('GrLivArea')
      plt.ylabel('SalePrice')
      plt.title('Outliers Detectados pelo LOF')
      plt.legend()
      plt.show()
      ```
