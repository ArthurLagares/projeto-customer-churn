# 📊 Projeto de Classificação de Churn de Clientes

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 📌 Objetivo do Projeto
Desenvolver um **modelo preditivo de classificação** capaz de **identificar a probabilidade de um cliente encerrar o contrato (churn)** com uma empresa de telecomunicações. O projeto tem como foco apoiar **decisões estratégicas de retenção de clientes**, utilizando técnicas de **Machine Learning supervisionado**.

---

## 🧠 Contexto do Projeto
- **Natureza**: Profissional / Acadêmico
- **Origem dos dados**: Kaggle
- **Setor**: Telecomunicações

A base de dados representa uma empresa de telecomunicações que comercializa planos de internet. O conjunto contém variáveis contratuais dos clientes, incluindo:
- Tempo de permanência (em meses) como cliente
- Valor da mensalidade
- Valor total gasto
- Tipo de plano contratado
- Serviços adicionais
- Indicador de churn (variável alvo)

---

## 👥 Público-alvo
Profissionais, estudantes e entusiastas das áreas de **Ciência de Dados**, **Análise de Dados**, **Machine Learning** e **Business Analytics**, bem como gestores interessados em estratégias de retenção de clientes.

---

## 🛠️ Tecnologias Utilizadas
- **Python**
- Bibliotecas principais:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - imbalanced-learn

---

## 📂 Estrutura do Projeto
```
├──📂arquivos
├── notebook.ipynb        # Notebook principal com análise e modelagem
├── funcoes_auxiliares.py       
├── README.md             # Documentação do projeto
└── data/                 # Base de dados (opcional)
```

---

## 🔍 Metodologia
O projeto foi estruturado seguindo um fluxo padrão de projetos profissionais de Machine Learning:

1. Importação das bibliotecas
2. Carregamento e tratamento dos dados
3. Análise Exploratória dos Dados (EDA)
4. Preparação dos dados para modelagem
5. Treinamento e avaliação de modelos de classificação
   - Modelo sem balanceamento
   - Balanceamento por *undersampling*
   - Balanceamento por *oversampling*
6. Seleção e ajuste do modelo final
7. Avaliação de desempenho e conclusões

---

## 🤖 Modelagem e Avaliação
Devido ao **desbalanceamento da variável alvo**, foram avaliadas diferentes estratégias de balanceamento para melhorar a capacidade preditiva dos modelos.

O **modelo final selecionado foi o Random Forest**, por apresentar melhor desempenho global e maior robustez nas métricas analisadas.

### 📈 Métricas de Desempenho
- **Precisão (Precision)**: superior a 70%
- **Recall**: superior a 70%
- Melhor desempenho na classe **"Não Churn"**, resultado esperado devido à maior prevalência dessa classe no conjunto de dados

Os resultados indicam que o modelo apresenta **desempenho satisfatório e aplicável em cenários reais de negócio**.

---

## ✅ Resultados Alcançados
- Modelo final com **boa capacidade de generalização**
- Métricas de desempenho consistentes
- Solução adequada para apoiar ações de retenção de clientes

---

## 📝 Conclusão
Este projeto demonstra uma aplicação prática de **Machine Learning para previsão de churn**, um problema crítico em empresas orientadas a serviços. A abordagem adotada, aliada aos resultados obtidos, evidencia o potencial do modelo como **ferramenta de apoio à tomada de decisão estratégica**.

---

## 📎 Considerações Finais
O projeto pode ser facilmente expandido para:
- Inclusão de novos atributos
- Teste de outros algoritmos
- Uso de validação cruzada
- Deploy do modelo em ambiente produtivo

Contribuições e sugestões são bem-vindas.

