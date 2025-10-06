import pandas as pd
import random, math
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from IPython.display import display
import matplotlib.pyplot as plt
import sqlite3

def criaTabelaAleatoriamente():
  'Crie um script para simular um dataset de vendas com pelo menos 50 registros, contendo as colunas'
  'ID, Data, Produto, Categoria, Quantidade, Preço'
  'O período dos dados deve ser de 01/01/2023 a 31/12/2023'
  D_Produto = {0: ['Protetor Solar Facial', 'Protetor Solar Corporal', 'Protetor Solar Antioleosidade'], 1: ['Hidratante Facial', 'Hidratante Corporal', 'Hidratante Antioleosidade'],
   2: ['Gel de Limpeza Facial', 'Gel de Limpeza Corporal', 'Gel de Limpeza Antioleosidade']}
  D_Categoria = ['Proteção', 'Hidratação', 'Limpeza']
  D_Preco = {0: [39.30, 61.29, 79.90], 1: [35.50, 59.90, 82.00], 2: [32.50, 63.50, 79.50]}
  dados = {'ID':[], 'Data':[], 'Produto': [], 'Categoria': [], 'Quantidade': [], 'Preço Unitario ($)': []}

  tabela = pd.DataFrame(dados)
  preenchimento = []
  opcoes = [0, 1, 2]
  pesos = [50, 20, 30]

  for i in range(1,252):
    resultado = random.choices(opcoes, weights=pesos, k=1)
    random_array = [random.randint(0, 2), resultado[0]]
    preenchimento.append(i)
    dd = f'{random.randint(1,28):02d}'
    mm = f'{(i%12)+1:02d}'
    data = dd+'/'+mm+'/2023'
    preenchimento.append( datetime.strptime(data, '%d/%m/%Y') )
    preenchimento.append(D_Produto[random_array[0]] [random_array[1]])
    preenchimento.append(D_Categoria[random_array[0]])
    preenchimento.append( determinaQuanidade(random_array, data) ) # Quantidade
    preenchimento.append(D_Preco[random_array[0]][random_array[1]])
    tabela.loc[len(tabela)] = preenchimento

    preenchimento.clear()

  return tabela

def determinaQuanidade(vetor, data):
# Facial -> Maior quantidade + Comum => Vezes na Tabela
 # Meses (12; 1; 2) Muito Protetor Solar Facial qtd*3; Corporal+3; Antioleosidade *2 + 3
 # Meses (12; 1; 2) _2 Todos * 1,5
 # Corporal -> Menor quantidade memos comum
 # AntiOleosidade  -> Menos quantidade e Comum
  quantidade = random.randint(1, 7)
  mes = data.split('/')[1]
  if mes == '12' or mes == '01' or mes == '02':
    if vetor[0] == 0 and vetor[1] == 0:
      quantidade *= 3 + random.randint(0, 1)
    elif vetor[0] == 0 and vetor[1] == 1:
      quantidade += 3
    elif vetor[0] == 0 and vetor[1] == 2:
      quantidade = (quantidade *2)+3
    elif vetor[1]== 2:
      quantidade = math.ceil(quantidade*1.5)

  elif vetor[1] == 0:
    quantidade += 4

  elif vetor[1] == 2:
    quantidade = max(1, quantidade-2)

  return quantidade

def geraDadosFaltantes(tabela):
  historico = []
  for i in range(201):
    temp = [random.randint(1, 4), random.randint(0, 96)]
    _ = True
    while _:
      if temp in historico:
        temp = [random.randint(1, 4), random.randint(0, 95)]
      else:
        tabela.loc[temp[1], tabela.columns[ temp[0] ] ] = None
        _ = False
        historico.append(temp)
  return tabela

def transformaStringNumerico(tabela):
  #Limpeza    -> 1
  #Hidratação -> 2
  #Proteção   -> 3
  #Gel de Limpeza Facial          -> 1.1
  #Gel de Limpeza Corporal        -> 1.2
  #Gel de Limpeza Antioleosidade  -> 1.3
  #Hidratante Facial              -> 2.3
  #Hidratante Corporal            -> 2.2
  #Hidratante Antioleosidade      -> 2.3
  #Protetor Solar Facial          -> 3.1
  #Protetor Solar Corporal        -> 3.2
  #Protetor Solar Antioleosidade  -> 3.3


  tabela['Categoria'] = tabela['Categoria'].map({'Limpeza': 1, 'Hidratação': 2, 'Proteção': 3})
  tabela['Produto'] = tabela['Produto'].map({'Gel de Limpeza Facial':11, 'Gel de Limpeza Corporal':12, 'Gel de Limpeza Antioleosidade': 13,
                                             'Hidratante Facial':21, 'Hidratante Corporal':22, 'Hidratante Antioleosidade': 23,
                                             'Protetor Solar Facial':31, 'Protetor Solar Corporal':32, 'Protetor Solar Antioleosidade': 33})
  tabela['Data'] = pd.to_datetime(tabela['Data']).dt.month
  tabela['Data'] = tabela['Data'].astype('Int64')
  tabela['Quantidade'] = tabela['Quantidade'].astype('Int64')
  tabela['Categoria'] = tabela['Categoria'].astype('Int64')
  tabela['Produto'] = tabela['Produto'].astype('Int64')
  return tabela

# Função auxiliar para preencher colunas discretas com KNN (tratando as outras colunas corretamente)
def preencher_coluna_discreta(df, col, colunas_discretas):
    df_treino = df.dropna(subset=[col])
    df_pred = df[df[col].isna()]

    if df_pred.empty:
        return df

    X_train = df_treino.drop(columns=[col])
    y_train = df_treino[col]
    X_pred = df_pred.drop(columns=[col])

    # tratamos como categóricas todas as colunas discretas (exceto a que é alvo)
    categorical_cols = [c for c in X_train.columns if c in colunas_discretas]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    # pipelines
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    model = Pipeline([
        ('preproc', preprocessor),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)

    df.loc[df[col].isna(), col] = y_pred
    return df

def preencher_quantidade(df, coluna_continua, colunas_discretas):
    col = coluna_continua
    df_treino = df.dropna(subset=[col])
    df_pred = df[df[col].isna()]

    if df_pred.empty:
        return df

    X_train = df_treino.drop(columns=[col])
    y_train = df_treino[col]
    X_pred = df_pred.drop(columns=[col])

    categorical_cols = [c for c in X_train.columns if c in colunas_discretas]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    model = Pipeline([
        ('preproc', preprocessor),
        ('reg', KNeighborsRegressor(n_neighbors=5))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)

    # arredonda se quer manter inteiros
    df.loc[df[col].isna(), col] = y_pred.round().astype(int)
    return df

def descobreVendaTotal(tabela):
  vendaTotalProduto = {}
  dicionario = {  11: 'Limpeza Facial', 12: 'Limpeza Corporal', 13: 'Limpeza Antioleosidade',
                  21: 'Hidratante Facial', 22: 'Hidratante Corporal', 23: 'Hidratante Antioleosidade',
                  31: 'Protetor Solar Facial', 32: 'Protetor Solar Corporal', 33: 'Protetor Solar Antioleosidade'}

  for produtoUnico in tabela['Produto'].unique():
      produtoAtual = tabela[tabela['Produto'] == produtoUnico]
      total_vendas = ( produtoAtual['Quantidade'] * produtoAtual['Preço Unitario ($)'] ).sum()
      vendaTotalProduto[dicionario[produtoUnico]] = total_vendas

  print(f'Total de vendas por produto:\n')
  vendaTotalProduto = pd.DataFrame(vendaTotalProduto.items(), columns=['Produto', 'Total de Vendas'])
  display(vendaTotalProduto.sort_values(by='Total de Vendas', ascending=False))
  return vendaTotalProduto

def imprimeHistorico(tabela):
  historico_Produtos = []
  indIce = []
  d_Grafico = {  11: 'Limpeza Facial', 12: 'Limpeza Corporal', 13: 'Limpeza Antioleosidade',
                  21: 'Hidratante Facial', 22: 'Hidratante Corporal', 23: 'Hidratante Antioleosidade',
                  31: 'Protetor Solar Facial', 32: 'Protetor Solar Corporal', 33: 'Protetor Solar Antioleosidade'}
  d_Cor= {11: '#b98aeb', 12: '#8b26f4', 13: '#c2a4e1', 21: '#80efe5', 22: '#15afa1', 23: '#a6e8e2', 31: '#f8a091', 32: '#db472d', 33: '#e6a69a'}
  d_Linha ={11: '-.', 12: '-', 13: '--', 21: '-.', 22: '-', 23: '--', 31: '-.', 32: '-', 33: '--'}

  for produtoUnico in tabela['Produto'].unique():
      produtoAtual = tabela[tabela['Produto'] == produtoUnico]
      historico_Individual = produtoAtual.groupby('Data')['Quantidade'].sum().reset_index()
      historico_Produtos.append(historico_Individual)
      indIce.append(produtoUnico)

  plt.figure(figsize=(12, 8))
  for i in range(len(historico_Produtos)):
      if indIce[i]==13 or indIce[i] == 23 or indIce[i] == 33:
        plt.plot(historico_Produtos[i]['Data'], historico_Produtos[i]['Quantidade'], label=d_Grafico[indIce[i]], color=d_Cor[indIce[i]], linestyle=d_Linha[indIce[i]], linewidth=3)
      else:
        plt.plot(historico_Produtos[i]['Data'], historico_Produtos[i]['Quantidade'], label=d_Grafico[indIce[i]], color=d_Cor[indIce[i]], linestyle=d_Linha[indIce[i]])
  plt.xlabel('Mês')
  plt.ylabel('Quantidade Vendida')
  plt.title('Tendência de Vendas ao Longo do Tempo')
  plt.legend()
  plt.grid(True)
  plt.show()

def agrupaPorCategoria(tabelaSegmentada):
  mapeamento_area_corpo = {
      11: 'Facial', 21: 'Facial', 31: 'Facial',
      12: 'Corporal', 22: 'Corporal', 32: 'Corporal',
      13: 'Antioleosidade', 23: 'Antioleosidade', 33: 'Antioleosidade'
  }

  # Adicionar coluna de Área do Corpo ao DataFrame
  tabelaSegmentada['Area do Corpo'] = tabelaSegmentada['Produto'].map(mapeamento_area_corpo)

  # Calcular vendas totais e quantidade por Área do Corpo usando groupby
  vendaSegmentada_C = tabelaSegmentada.groupby('Area do Corpo').agg(
      Total_de_Vendas=('Preço Unitario ($)', lambda x: (x * tabelaSegmentada.loc[x.index, 'Quantidade']).sum()),
      Quantidade=('Quantidade', 'sum')
  ).reset_index()
  # Calcular a rentabilidade
  vendaSegmentada_C['Rentabilidade'] = vendaSegmentada_C['Total_de_Vendas'] / vendaSegmentada_C['Quantidade']

  return tabelaSegmentada, vendaSegmentada_C

def agrupaPorEtapa(tabelaSegmentada):
  mapeamento_TipoProduto = {
      11: 'Limpeza', 21: 'Hidratante', 31: 'Protetor',
      12: 'Limpeza', 22: 'Hidratante', 32: 'Protetor',
      13: 'Limpeza', 23: 'Hidratante', 33: 'Protetor'
  }
  tabelaSegmentada['Tipo Produto'] = tabelaSegmentada['Produto'].map(mapeamento_TipoProduto)

  vendaSegmentada_TP = tabelaSegmentada.groupby('Tipo Produto').agg(
      Total_de_Vendas=('Preço Unitario ($)', lambda x: (x * tabelaSegmentada.loc[x.index, 'Quantidade']).sum()),
      Quantidade=('Quantidade', 'sum')
  ).reset_index()

  vendaSegmentada_TP['Rentabilidade'] = vendaSegmentada_TP['Total_de_Vendas'] / vendaSegmentada_TP['Quantidade']

  return tabelaSegmentada, vendaSegmentada_TP

def historicoCaracteristica(tabelaSegmentada, visao):
  indice = {0: 'Quantidade', 1: 'Faturamento'}
  legenda = {0: 'Vendas em Unidade Agrupado por Caracteristica', 1:'Faturamento Agrupado por Caracteristica'}
  unidade = {0: ' (Und)', 1: ' ($)'}
  historico_ParteCorpo = []
  i_Corpo = []
  d_CorCorpo= {'Facial (Und)': '#8b26f4', 'Corporal (Und)': '#15afa1', 'Antioleosidade (Und)': '#db472d',
                'Facial ($)': '#8b26f4', 'Corporal ($)': '#15afa1', 'Antioleosidade ($)': '#db472d'}

  for produtoUnico in tabelaSegmentada['Area do Corpo'].unique():
      produtoAtual = tabelaSegmentada[tabelaSegmentada['Area do Corpo'] == produtoUnico].copy() # Use copy to avoid SettingWithCopyWarning
      produtoAtual['Faturamento'] = produtoAtual['Quantidade'] * produtoAtual['Preço Unitario ($)']
      h_Individual_C = produtoAtual.groupby('Data').agg(
          Quantidade=('Quantidade', 'sum'),
          Faturamento=('Faturamento', 'sum')
      ).reset_index()
      historico_ParteCorpo.append(h_Individual_C)
      i_Corpo.append(produtoUnico)
  #display(historico_ParteCorpo[0])

  plt.figure(figsize=(12, 8))
  for i in range(len(historico_ParteCorpo)):
      qtd = i_Corpo[i] + unidade[visao]
      money = i_Corpo[i] + unidade[visao]
      plt.plot(historico_ParteCorpo[i]['Data'], historico_ParteCorpo[i][indice[visao]], label=qtd, color=d_CorCorpo[qtd])
  plt.xlabel('Mês')
  plt.ylabel(indice[visao])
  plt.title(legenda[visao])
  plt.legend()
  plt.grid(True)
  plt.show()

def historicoEtapa(tabelaSegmentada, visao):
  indice = {0: 'Quantidade', 1: 'Faturamento'}
  legenda = {0: 'Vendas em Unidade Agrupado por Etapa', 1:'Faturamento Agrupado por Etapa'}
  unidade = {0: ' (Und)', 1: ' ($)'}

  historico_CaracteristicaP = []
  i_CaraP = []
  d_CorCorpo= {'Limpeza (Und)': '#8b26f4', 'Hidratante (Und)': '#15afa1', 'Protetor (Und)': '#db472d',
                'Limpeza ($)': '#8b26f4', 'Hidratante ($)': '#15afa1', 'Protetor ($)': '#db472d'}

  for produtoUnico in tabelaSegmentada['Tipo Produto'].unique():
      produtoAtual = tabelaSegmentada[tabelaSegmentada['Tipo Produto'] == produtoUnico].copy()
      produtoAtual['Faturamento'] = produtoAtual['Quantidade'] * produtoAtual['Preço Unitario ($)']
      h_Individual_CP = produtoAtual.groupby('Data').agg(
          Quantidade=('Quantidade', 'sum'),
          Faturamento=('Faturamento', 'sum')
      ).reset_index()
      historico_CaracteristicaP.append(h_Individual_CP)
      i_CaraP.append(produtoUnico)

  plt.figure(figsize=(12, 8))
  for i in range(len(historico_CaracteristicaP)):
      qtd = i_CaraP[i] + unidade[visao]
      money = i_CaraP[i] + unidade[visao]
      plt.plot(historico_CaracteristicaP[i]['Data'], historico_CaracteristicaP[i][indice[visao]], label=qtd, color=d_CorCorpo[qtd])
  plt.xlabel('Mês')
  plt.ylabel(indice[visao])
  plt.title(legenda[visao])
  plt.legend()
  plt.grid(True)
  plt.show()

tabela = criaTabelaAleatoriamente()

t_2 = geraDadosFaltantes(tabela.copy())
preProcessada = transformaStringNumerico(t_2.copy())

colunas_discretas = ['Produto', 'Categoria', 'Data']
coluna_continua = 'Quantidade'
df = preProcessada.copy()

for c in colunas_discretas:
    df = preencher_coluna_discreta(df, c, colunas_discretas)

# Aplicar para quantidade (contínua)
clean = preencher_quantidade(df, coluna_continua, colunas_discretas)

descobreVendaTotal(clean.copy())

imprimeHistorico(clean.copy())

dfExpandido, agrupado_Categoria = agrupaPorCategoria(clean.copy())
print(f'Total de vendas por produto:\n')
display(agrupado_Categoria.sort_values(by='Total_de_Vendas', ascending=False))

print(f'Total de vendas por Tipo de Produto:\n')
dfExpandido, agrupado_Caracteristica = agrupaPorEtapa(dfExpandido.copy())
display(agrupado_Caracteristica.sort_values(by='Total_de_Vendas', ascending=False))

historicoCaracteristica(dfExpandido.copy(),0)
historicoCaracteristica(dfExpandido.copy(),1)

historicoEtapa(dfExpandido.copy(),0)
historicoEtapa(dfExpandido.copy(),1)

conexao = sqlite3.connect("meu_banco.db")

# Criar conexão com um banco temporário em memória
con = sqlite3.connect(':memory:')

# Transformar DataFrame em tabela SQL chamada 'clientes'
dfExpandido.to_sql('vendas', con, index=False, if_exists='replace')