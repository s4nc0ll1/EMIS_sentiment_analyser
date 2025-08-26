# EMIS Sentiment Analyser

Herramienta interactiva desarrollada con Streamlit para visualizar y analizar sentimientos sobre publicaciones económicas usando APIs internas de EMIS/CEIC.

---

## 🚀 Requisitos

- Python >= 3.12
- [`uv`](https://github.com/astral-sh/uv) (gestor de entornos + dependencias ultra-rápido)

Instálalo con:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
````

---

## Instalación con `uv`

1. Clona el repositorio y entra al proyecto:

```bash
git clone git@github.com:s4nc0ll1/EMIS_sentiment_analyser.git
cd EMIS_sentiment_analyser
```

2. Crea el entorno virtual y sincroniza las dependencias (con acceso al índice privado CEIC):

```bash
uv venv
uv sync
```
---

## Ejecutar la aplicación

Para iniciar el servidor de Streamlit:

```bash
uv run streamlit run app.py
```

Luego abre en tu navegador:

```
http://localhost:8501
```

---

## Estructura del proyecto

```
.
├── app.py
├── config.py
├── core
│   ├── emis_downloader
│   │   ├── cli.py
│   │   ├── fetcher.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── resolver.py
│   │   ├── sentiment.py
│   │   └── storage.py
│   └── __init__.py
├── data
│   ├── countries.json
│   └── industries.json
├── emis_api.py
├── __init__.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── series.py
├── static
│   ├── ceic-logo.png
│   ├── images.png
│   ├── logo2.png
│   ├── logo.svg
│   └── styles.css
├── templates
│   └── index.html
└── uv.lock

5 directories, 25 files
```

---

## Autor

Desarrollado por ISI Markets, Departamento de tecnología, Bogotá
