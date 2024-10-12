# **Engenharia de software - Inteligência artificial - 6° A**

 - Bruno Mello
 - Bryan Henrique
 - Caio Zampini
 - Carlos Herique
 - Lucas Lizot
 - Ronald Ivan
 - Vitor Ferreira
 - Victor Yuzo

 ---

# Análise de partidas de Age of Empires 2 utilizando SVM (Support Vector Machine)

## Objetivo(s) do projeto:
1. Bla
2. Bla
3. Bla 

---

## Análise do código:

- Importações necessários para rodar o projeto:

        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
        from sklearn.inspection import permutation_importance
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import LabelEncoder
        from sklearn.inspection import permutation_importance

- Carregando o dataset "[Age of Empires 2: DE Match Data](https://www.kaggle.com/datasets/nicoelbert/aoe-matchups?resource=download)" 

        file_path = 'https://raw.githubusercontent.com/FlamingoLindo/UMC-Age-of-Empires-2/main/aoe_data.csv'
        df = pd.read_csv(file_path)
        
O dataset foi baixado diretamento do Kaggle e ele está sendo hospedado diretamente neste repositório.

### De que o dataset é composto?

| match_id   | map    | map_size | duration | dataset            | difficulty | elo   | p1_civ   | p2_civ   | p1_xpos | p2_xpos | p1_ypos | p2_ypos | winner |
|------------|--------|----------|----------|--------------------|------------|--------|----------|----------|---------|---------|---------|---------|--------|
| 50453403   | Arabia | Tiny     | 3445     | Definitive Edition | Hardest    | 1104.0 | Vikings  | Mayans   | 92.0    | 24.0    | 37.0    | 78.0    | 0      |
| 118982970  | Arena  | Tiny     | 2932     | Definitive Edition | Hardest    | 884.5  | Britons  | Goths    | 70.0    | 82.0    | 16.0    | 98.0    | 0      |
| 57185801   | Arena  | Tiny     | 2573     | Definitive Edition | Hardest    | 905.5  | Chinese  | Malians  | 69.0    | 61.0    | 16.0    | 104.0   | 0      |
| 64335748   | Arabia | Tiny     | 851      | Definitive Edition | Hardest    | 1080.0 | Mayans   | Magyars  | 25.0    | 98.0    | 80.0    | 68.0    | 1      |
| 116883036  | Arabia | Tiny     | 4737     | Definitive Edition | Hardest    | 1050.0 | Berbers  | Slavs    | 83.0    | 30.0    | 28.0    | 85.0    | 1      |

**1. match_id**
Identificador único da partida.

**2. map:**
Nome do mapa jogado.

**3. map_size:**
Tamanho do mapa.

**4. duration:**
Duração da partida em segundos.

**5. dataset:**
Versão do jogo usada para a partida (e.g., Definitive Edition).

**6. difficulty:**
Nível de dificuldade da partida.

**7. elo:**
Classificação de habilidade (Elo) dos jogadores.

**8. p1_civ:**
Civilização escolhida pelo jogador 1.

**9. p2_civ:**
Civilização escolhida pelo jogador 2.

**10. p1_xpos:**
Posição X inicial do jogador 1 no mapa.

**11. p2_xpos:**
Posição X inicial do jogador 2 no mapa.

**12. p1_ypos:**
Posição Y inicial do jogador 1 no mapa.

**13. p2_ypos:**
Posição Y inicial do jogador 2 no mapa.

**14. winner:**
Indica se o jogador 1 venceu a partida (1 para vitória, 0 para derrota).

### Verificando se há valores unícos nas colunas:

    colunas = df.columns

    for nome_coluna in colunas:
        unique_values = df[nome_coluna].unique()
        print(f'Valores unícos em {nome_coluna}: {unique_values} \n')


### Valores Únicos por Coluna:

| Coluna        | Valores Únicos                                                                                                                                 |
|---------------|------------------------------------------------------------------------------------------------------------------------------------------------                                                                       |
| match_id      | [50453403, 118982970, 57185801, ..., 124942713, 141486159, 139998132]                                                                         |
| map           | ['Arabia', 'Arena', 'Four Lakes', 'Steppe', 'Golden Pit', 'African Clearing', 'Serengeti', 'Nomad', 'MegaRandom', 'Coastal Forest', ...]       |
| map_size      | ['Tiny']                                                                                                                                      |
| duration      | [3445, 2932, 2573, ..., 5999, 6287, 5860]                                                                                                     |
| dataset       | ['Definitive Edition']                                                                                                                        |
| difficulty    | ['Hardest']                                                                                                                                   |
| elo           | [1104.0, 884.5, 905.5, ..., 766.0, 2185.5, 2269.0]                                                                                            |
| p1_civ        | ['Vikings', 'Britons', 'Chinese', 'Mayans', 'Berbers', 'Khmer', 'Cumans', 'Huns', 'Malay', 'Ethiopians', 'Magyars', 'Franks', 'Tatars', ...]  |
| p2_civ        | ['Mayans', 'Goths', 'Malians', 'Magyars', 'Slavs', 'Teutons', 'Saracens', 'Koreans', 'Bulgarians', 'Lithuanians', 'Turks', 'Vietnamese', ...] |
| p1_xpos       | [92.0, 70.0, 69.0, ..., 90.0, 107.0, 91.0]                                                                                                    |
| p2_xpos       | [24.0, 82.0, 61.0, ..., 40.0, 47.0, 37.0]                                                                                                     |
| p1_ypos       | [37.0, 16.0, 25.0, ..., 20.0, 72.0, 39.0]                                                                                                     |
| p2_ypos       | [78.0, 98.0, 104.0, ..., 100.0, 87.0, 55.0]                                                                                                   |
| winner        | [0, 1]                                                                                                                                        |

### Colunas removidas:

    df.drop(['Unnamed: 0', 'match_id', 'map_size', 'dataset', 'difficulty'], axis=1, inplace=True)

 | map    | duration | elo   | p1_civ  | p2_civ  | p1_xpos | p2_xpos | p1_ypos | p2_ypos | winner |
|--------|----------|-------|---------|---------|---------|---------|---------|---------|--------|
| Arabia | 3445     | 1104.0| Vikings | Mayans  | 92.0    | 24.0    | 37.0    | 78.0    | 0      |
| Arena  | 2932     | 884.5 | Britons | Goths   | 70.0    | 82.0    | 16.0    | 98.0    | 0      |
| Arena  | 2573     | 905.5 | Chinese | Malians | 69.0    | 61.0    | 16.0    | 104.0   | 0      |
| Arabia | 851      | 1080.0| Mayans  | Magyars | 25.0    | 98.0    | 80.0    | 68.0    | 1      |
| Arabia | 4737     | 1050.0| Berbers | Slavs   | 83.0    | 30.0    | 28.0    | 85.0    | 1      |

Decidimos remover as colunas a cima, pois elas não farão diferença durante o trinamento da IA, pois algumas telas tem o mesmo valor para todas as linhas.

### Verificação de valores faltantes no dataset:
    df.isna().sum()

coluna | Quantidade
|------|-----------|
map      |        0
duration |       0
elo      |        0
p1_civ   |        0
p2_civ   |       0
p1_xpos  |   10596
p2_xpos  |   10596
p1_ypos  |   10596
p2_ypos  |   10596
winner   |       0

### O que fazer com esse valores faltantes?

Pensamos em duas possibilidades para esse problema:

1. Substituir os valores faltantes pela média de cada coluna.

        # Média p1_xpos
        med_p1x = df['p1_xpos'].mean()
        df['p1_xpos'].fillna(med_p1x, inplace=True)

        # Média p2_xpos
        med_p2x = df['p2_xpos'].mean()
        df['p2_xpos'].fillna(med_p2x, inplace=True)

        # Média p1_ypos
        med_p1y = df['p1_ypos'].mean()
        df['p1_ypos'].fillna(med_p1y, inplace=True)


        # Média p2_ypos
        med_p2y = df['p2_ypos'].mean()
        df['p2_ypos'].fillna(med_p2y, inplace=True) 
ou 

2. Remover linhas com valores nulos.

        df.dropna(inplace=True)
        df.isna().sum()

### Remoção de linhas do dataset:

    df.drop(range(primeira_linha, ultima_linha), inplace=True, errors='ignore')

Fazemos essa etapa para que o modelo possa ser treinado mais rapidamente.

### Tipos de encoding:

**Label encoding:**

    le_map = LabelEncoder()
    df['map_encoded'] = le_map.fit_transform(df['map'])
    le_civ = LabelEncoder()

    df['p1_civ_encoded'] = le_civ.fit_transform(df['p1_civ'])
    le_civ2 = LabelEncoder()

    df['p2_civ_encoded'] = le_civ2.fit_transform(df['p2_civ'])

**OneHot enconding**

### Remoção de colunas categoricas:

Após passar as colunas categoricas ('map', 'p1_civ', 'p2_civ') pelo encoding, removemos elas para que atrapalhem durante o treinamento.

    df.drop(['map', 'p1_civ', 'p2_civ'], axis=1, inplace=True)

### Separação dos dados de teste e treino

    X = df.drop('winner', axis=1)
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

### Classificação de Vetor de Suporte:

    model = SVC(kernel='linear')
    model.fit(X_train_scaled, y_train)

### Primeira matriz de confusão:

    y_pred = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusão - SVM Linear')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()
    print(classification_report(y_test, y_pred))

![primeira matriz de confusão](https://github.com/FlamingoLindo/UMC-Age-of-Empires-2/blob/Branch-VitorFerreira/assets/first_matrix.png)

|              | precision | recall | f1-score | support |
|--------------|------------|--------|----------|---------|
| **0**        | 0.52       | 0.42   | 0.46     | 337     |
| **1**        | 0.50       | 0.60   | 0.55     | 329     |
|     ⠀        |      ⠀     |   ⠀    |   ⠀      |    ⠀    |
| **accuracy** |            |        | 0.51     | 666     |
| **macro avg**| 0.51       | 0.51   | 0.50     | 666     |
| **weighted avg** | 0.51   | 0.51   | 0.50     | 666     |

*Essa matriz e sua precisão são apenas um exemplo. 

### Testanto diferentes valores para C, Gamma e Kernals utilizando Grid Search:

    svm = SVC(probability=True)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    # Instanciar o modelo SVM
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

**Melhores parâmetros e estimadores:**

    print('Melhores parâmetros: ', grid_search.best_params_)
    print('Melhor estimador: ', grid_search.best_estimator_)

    Melhores parâmetros:  {'C': 10, 'gamma': 1, 'kernel': 'rbf'}
    Melhor estimador:  SVC(C=10, gamma=1, probability=True)

**Relátorio de classificação com os melhores parâmetros e estimadores:**

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

**Acurácia:** 0.51

**Relatório de Classificação:**

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **0**        | 0.52      | 0.50   | 0.51     | 337     |
| **1**        | 0.51      | 0.52   | 0.51     | 329     |
|              |           |        |          |         |
| **accuracy** |           |        | 0.51     | 666     |
| **macro avg**| 0.51      | 0.51   | 0.51     | 666     |
| **weighted avg** | 0.51   | 0.51   | 0.51     | 666     |

![segunda matrix](https://github.com/FlamingoLindo/UMC-Age-of-Empires-2/blob/Branch-VitorFerreira/assets/second_matrix.png)

### Permutação:

    result = permutation_importance(best_svm, X_test_scaled, y_test, n_repeats=10, random_state=42)
    sorted_idx = result.importances_mean.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
    plt.xlabel('Importância das features')
    plt.title('Importância das Features via Permutação no dataset')
    plt.show()

![permutação](https://github.com/FlamingoLindo/UMC-Age-of-Empires-2/blob/Branch-VitorFerreira/assets/permutation.png)

### Análise das features e suas classificações:

    fp_indices = np.where((y_pred == 1) & (y_test != 1))[0]
    fn_indices = np.where((y_pred != 1) & (y_test == 1))[0]
    fp_samples = X_test.iloc[fp_indices]
    fn_samples = X_test.iloc[fn_indices]
    correctly_classified_samples = X_test[(y_pred == y_test)]
    fp_mean = fp_samples.mean()
    fn_mean = fn_samples.mean()
    correctly_classified_mean = correctly_classified_samples.mean()
    comparison_df = pd.DataFrame({
        'Falsos positivos': fp_mean,
        'Falsos Negativos': fn_mean,
        'Classificados corretamente': correctly_classified_mean
    })
    print('Comparação das Features (Médias):')
    print(comparison_df)

**Comparação das Features (Médias):**

| Feature           | Falsos Positivos | Falsos Negativos | Classificados Corretamente |
|-------------------|------------------|------------------|----------------------------|
| **duration**      | 2487.38          | 2255.95          | 2332.40                    |
| **elo**           | 1113.19          | 1135.77          | 1131.78                    |
| **p1_xpos**       | 60.73            | 58.54            | 58.99                      |
| **p2_xpos**       | 60.80            | 58.83            | 61.15                      |
| **p1_ypos**       | 59.17            | 61.19            | 61.25                      |
| **p2_ypos**       | 61.28            | 57.55            | 58.61                      |
| **map_encoded**   | 5.81             | 6.47             | 5.82                       |
| **p1_civ_encoded**| 18.83            | 18.55            | 18.35                      |
| **p2_civ_encoded**| 18.22            | 15.64            | 18.20                      |
