# 🎯 NEW USERS START HERE

**Welcome to Entity Resolution!** This guide is specifically designed for newcomers to named entity resolution and this package. Follow these steps to get up and running quickly.

## 🚀 **Your First 30 Minutes**

### **Step 1: Quick Setup (5 minutes)**
```bash
# Navigate to the package directory
cd entity_resolution_demo_package

# Create + activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Install tutorial dependencies
pip install -e ".[tutorial]"

# Set up Jupyter kernel (creates "Entity Resolution Demo" kernel)
python setup_kernel.py

# Verify everything is set up correctly
python setup_environment.py
```

If you haven’t already, skim README.md for a high-level overview of the repo layout and datasets.

### **Step 2: Configure Your Environment (10 minutes)**
```bash
# Create your .env file with credentials
cp env.template .env

# Edit the .env file with your actual credentials
nano .env  # or use your preferred editor
```

**You'll need:**
- **Elasticsearch Cloud account** (free tier available)
- **OpenAI API key** (for AI-powered matching)
- **Internet connection** (for Wikipedia API)

**Environment variables needed:**
- `ELASTIC_ENDPOINT` - Your Elasticsearch URL/endpoint (preferred if you have it)
- `ELASTIC_CLOUD_ID` - Your Elasticsearch Cloud ID (alternative to `ELASTIC_ENDPOINT`)
- `ELASTIC_API_KEY` - Your Elasticsearch API key
- `OPENAI_API_KEY` - Your OpenAI API key
- `LITELLM_PROXY_URL` - (Optional) LiteLLM proxy URL for using proxy instead of direct OpenAI API

### **Step 3: Install the NER Model (5 minutes)**
**⚠️ CRITICAL**: The NER model must be manually deployed to your Elasticsearch Cloud instance using Docker.

**Installation Steps:**
```bash
# 1. Install Docker if you haven't already
# 2. Run this command (replace with your actual credentials):
docker run -it --rm docker.elastic.co/eland/eland \
  eland_import_hub_model \
  --cloud-id $ELASTIC_CLOUD_ID \
  --es-api-key $ELASTIC_API_KEY \
  --hub-model-id facebookai/xlm-roberta-large-finetuned-conll03-english \
  --task-type ner \
  --start
```

**Important Notes:**
- The model will appear as `facebookai__xlm-roberta-large-finetuned-conll03-english` (with double underscores) in Elasticsearch
- Use the Hugging Face format (single slash) in the Docker command: `facebookai/xlm-roberta-large-finetuned-conll03-english`
- This process may take 5-10 minutes depending on your internet connection
- The `--start` flag automatically deploys the model after import

**Why this step is needed:**
- The e5 embedding model is built into Elasticsearch Cloud
- The NER model must be manually installed and deployed
- Without this model, entity extraction will fail
- The Kibana UI import option is not available in Elastic Cloud

**Need help?** For detailed instructions and troubleshooting, see the [Elastic NER deployment guide](https://www.elastic.co/blog/how-to-deploy-nlp-named-entity-recognition-ner-example).

### **Step 4: Jupyter Kernel Setup (2 minutes)**
**⚠️ IMPORTANT**: The notebooks require a specific Jupyter kernel to work properly.

**What is a Jupyter kernel?**
- A Jupyter kernel is the Python environment that runs your notebook code
- The "Entity Resolution Demo" kernel has all the required packages installed
- Without the correct kernel, you'll get import errors

**The setup script already created the kernel, but here's how to verify:**
```bash
# Check that the kernel is installed
jupyter kernelspec list

# You should see "entity-resolution" in the list
```

**In Jupyter, you'll need to select the correct kernel:**
1. Open a notebook in Jupyter
2. Look for the kernel name in the top-right corner (might show "Python 3" or similar)
3. Click on it and select "Entity Resolution Demo"
4. Now your notebook will have access to all the required packages

**Why this matters:**
- ✅ **No Import Errors**: `import entity_resolution_demo` will work
- ✅ **All Dependencies**: elasticsearch, openai, pandas, etc. are available
- ✅ **Consistent Environment**: Same setup for all users

### **Step 5: Start Your First Notebook (10 minutes)**
```bash
# Start Jupyter from the repo root (recommended)
cd ..
jupyter lab

# Open your first notebook
notebooks/01_entity_preparation_v4.ipynb
```

If Notebook 01 runs end-to-end without errors, your environment is correctly set up.

## 📊 **Datasets (Minimal vs Comprehensive)**

This demo repo includes two dataset modes:

- **Minimal datasets (default in Notebook 01)**:
  - `notebooks/minimal_entities.json`
  - These are designed to run quickly and get you to “Notebook 01 runs” fast.

- **Comprehensive evaluation datasets (all tiers 0–5)**:
  - `comprehensive_evaluation/data/`
  - See `comprehensive_evaluation/README.md` for tier structure and provenance.

**Tier‑5 note (explicit):** the v4 capstone notebook uses the canonical Tier‑5 files:
- `comprehensive_evaluation/data/tier5_test_articles_v2.json`
- `comprehensive_evaluation/data/tier5_watch_list_cleaned.json`

## 🔧 **Troubleshooting (top 5)**

1) **Notebook 01 can’t import the package**
- Do: confirm you ran `pip install -e .` inside `entity_resolution_demo_package/`.
- Expect: `import entity_resolution_demo` works without `ModuleNotFoundError`.

2) **Elasticsearch connection fails**
- Do: verify `ELASTIC_CLOUD_ID` and `ELASTIC_API_KEY` in `.env`, then rerun `python setup_environment.py`.
- Expect: the setup script reports Elasticsearch as reachable.

3) **`.env` variables aren’t being picked up**
- Do: ensure `.env` is located at `entity_resolution_demo_package/.env` and you ran `python setup_environment.py` from that directory.
- Expect: the setup script reports required variables as configured.

4) **Jupyter is opened from the wrong directory**
- Do: launch Jupyter from the **repo root** (Step 4 uses `cd ..` for a reason).
- Expect: paths like `notebooks/01_entity_preparation_v4.ipynb` resolve cleanly.

5) **Kernel/import issues inside Jupyter**
- Do: run `python setup_kernel.py`, then select the “Entity Resolution Demo” kernel in Jupyter.
- Expect: notebooks use the same environment you installed with `pip install -e .`.

---

**Need help?** Revisit `README.md` and `comprehensive_evaluation/README.md`, then rerun `python setup_environment.py`.