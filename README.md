# 🫁 Pneumonia Detection from Chest X-Rays (AI/ML Major Project)

An end-to-end deep learning web app that detects **Pneumonia** from chest X-ray images,
using **Transfer Learning (EfficientNetB0)** + **Grad-CAM** explainability, deployed as a
Flask web application.

---

## 📁 Project Structure

```
pneumonia-detector/
├── notebooks/
│   └── train_model.ipynb        # Train on Google Colab (free GPU)
├── model/
│   └── train.py                 # Same training code as .py script
├── app/
│   ├── app.py                   # Flask backend (prediction + Grad-CAM)
│   ├── templates/
│   │   └── index.html           # Frontend UI
│   └── static/
│       └── uploads/             # Uploaded images stored here temporarily
├── requirements.txt
├── Procfile                     # For Render/Railway deployment
└── README.md
```

---

## 🚀 Step-by-Step: How to Build & Deploy This

### Step 1 — Get the Dataset
Use the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

It has ~5,800 X-ray images labeled `NORMAL` / `PNEUMONIA`.

### Step 2 — Train the Model (use Google Colab — free GPU)
1. Open `notebooks/train_model.ipynb` in Google Colab (upload it there)
2. Upload the Kaggle dataset (or use `kaggle.json` API key to download directly in Colab)
3. Run all cells — it will:
   - Load EfficientNetB0 pretrained on ImageNet
   - Fine-tune on chest X-ray data
   - Save the trained model as `pneumonia_model.h5`
4. Download `pneumonia_model.h5` and place it inside the `model/` folder

Training takes ~20-30 mins on Colab's free GPU for 10 epochs.

### Step 3 — Run Locally to Test
```bash
cd pneumonia-detector
pip install -r requirements.txt
python app/app.py
```
Open `http://localhost:5000` in your browser, upload an X-ray image, and see the prediction
with a Grad-CAM heatmap showing which lung regions influenced the decision.

### Step 4 — Deploy (Free Options)
**Option A: Render.com (easiest)**
1. Push this project to a GitHub repo
2. Go to https://render.com → New Web Service → connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app.app:app`
5. Deploy — you'll get a live public URL

**Option B: Hugging Face Spaces**
1. Create a new Space (SDK: Gradio or Docker)
2. Push your code — HF Spaces gives free hosting with GPU options too

**Option C: Railway.app** — similar to Render, very beginner-friendly

---

## 🧠 What Makes This Project Strong for Final Year Evaluation
- **Transfer Learning**: Shows you understand modern CNN architectures, not training from scratch
- **Grad-CAM Explainability**: Visually shows *why* the model made a decision — examiners love this, and it's a great talking point in viva
- **Full deployment**: Working live demo > just a Jupyter notebook
- **Real-world relevance**: Healthcare AI has strong social impact angle for your report

## 📊 Suggested Report/PPT Sections
1. Problem Statement & Motivation
2. Literature Survey (cite 4-5 papers on pneumonia detection via deep learning)
3. Dataset Description & Preprocessing
4. Model Architecture (EfficientNetB0 + custom classification head)
5. Training methodology, hyperparameters, augmentation
6. Results: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
7. Grad-CAM visualizations (sample outputs)
8. Deployment architecture diagram
9. Limitations & Future Work (e.g., multi-disease classification, mobile app)

## ⚠️ Important Disclaimer (include in your report/app)
This tool is for **educational/research purposes only** and is not a substitute for
professional medical diagnosis.

---

## 🔧 Possible Extensions (if you want to go further)
- Add multi-class classification (Normal / Bacterial Pneumonia / Viral Pneumonia)
- Add a second disease (e.g., COVID-19 chest X-ray dataset) for multi-disease support
- Build a mobile app frontend using Flutter calling the same Flask API
- Add user authentication + prediction history dashboard
