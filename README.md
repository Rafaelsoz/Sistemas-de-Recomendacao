# Recomendação de Músicas com Multi-Armed Bandits (Epsilon-Greedy & UCB)

Este repositório contém um projeto didátic, voltado para o Ensino Médio, a fim de explicar **como sistemas de recomendação aprendem** usando o problema de **Multi-Armed Bandits**.  
A ideia é simular ou coletar interativamente o feedback do usuário em recomendações de **gêneros musicais**, comparando:

- **Epsilon-Greedy**
- **UCB (Upper Confidence Bound)**
- **Aleatório** como baseline

O app foi feito com **Streamlit** e inclui visualizações em **Plotly**.

---

## ✨ O que você encontra aqui

### ✅ Modo 1 — Simulado
- O ambiente tem **probabilidades verdadeiras** de “like” por gênero feito com dados sintéticos.
- O algoritmo recomenda e recebe recompensas automaticamente.
- O app plota curvas de aprendizado:
  - **Likes acumulados (recompensa acumulada)**
  - **% de escolhas do melhor gênero**
  - **Proporção de recomendações por gênero**
  - **Média estimada de likes por gênero**

### ✅ Modo 2 — Ao vivo com a turma
- A turma vira o ambiente: cada recomendação recebe um feedback:
  - 👍 gostei = 1
  - 👎 não gostei = 0
- Após cada feedback, o algoritmo **já gera a próxima recomendação automaticamente**.
- Para tornar a dinâmica mais realista o app exibe um **nome de música aleatório** coerente com o gênero .

---

## 📄 Conceitos 

### Exploração vs Explotação
- **Explorar**: testar opções menos conhecidas para aprender.
- **Explotar**: repetir a opção que parece melhor.

### Epsilon-Greedy
- Com probabilidade **ε**, escolhe um gênero aleatório (explora).
- Com probabilidade **1-ε**, escolhe o gênero com melhor média de likes (explota).

### UCB
- Escolhe o gênero com maior:
  - **média estimada** + **bônus de incerteza**
- Gêneros pouco testados ganham bônus maior → exploração “inteligente”.

---

## 🚀 Como rodar localmente

### 1) Clonar o repositório
```bash
git clone git@github.com:Rafaelsoz/Sistemas-de-Recomendacao.git
cd SEU_REPO
```

### 2) Criar um ambiente virtual
```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```` 

### 3) Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4) Rodar Aplicativo
```bash
streamlit run app.py
```

----
