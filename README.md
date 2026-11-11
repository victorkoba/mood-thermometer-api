# 😊 Termômetro de Humor (Mood Thermometer API)

Aplicação para detecção de rostos e análise de expressões faciais em tempo real ou por foto, utilizando **FastAPI**, **OpenCV**, **Hugging Face Transformers (ViT)** e **Streamlit**.

## 🚀 Como rodar o projeto

### 1. Criar e ativar o ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Iniciar a API (FastAPI)

Em um terminal (com a `.venv` ativa), execute:

```bash
uvicorn api:app --reload
```

* A API ficará disponível em: `http://127.0.0.1:8000`
* Documentação interativa (Swagger UI): `http://127.0.0.1:8000/docs`

### 4. Iniciar a interface (Streamlit)

Com a API rodando, abra um **segundo terminal** (com a `.venv` ativa) e execute:

```bash
python -m streamlit run app.py
```

* O painel abrirá no navegador em: `http://localhost:8501`

---

## 📌 Principais Rotas da API

| Método | Rota | Descrição |
| :--- | :--- | :--- |
| `GET` | `/` | Verifica se a API está no ar |
| `POST` | `/analyze` | Recebe uma imagem (`image`) e retorna os rostos e emoções detectadas |
| `POST` | `/assessments` | Registra manualmente uma avaliação de humor |
| `GET` | `/assessments` | Lista as avaliações registradas (filtro opcional por `polarity`) |
| `GET` | `/assessments/{id}` | Busca uma avaliação pelo ID |
| `GET` | `/resume` | Retorna o resumo consolidado por polaridade e emoção |
| `DELETE` | `/assessments` | Limpa todas as avaliações registradas |
