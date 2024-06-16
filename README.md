# analiseDadosProjetoC3
Esse repositório contém uma análise do DataFrame house_prices((house_train.csv)[house_train]), disponível no Kaggle.

## Organização:
* Colocar nome do arquivo: Aqui constam as análises realizadas e os modelos criados.
  - Em um primeiro momento, foi feita a análise exploratória dos dados para entender melhor a disposição dos dados e as características;
    - Exemplo de análise exploratória feita:
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
  - Após verificar as características dos dados, foi feita a engenharia de características paara selecionar as variáveis mais importantes para o modelo;
      
