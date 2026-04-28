# 🧠 NLP Sentiment Lab (TF-IDF vs BERT)

An interactive text classification project that compares a **traditional NLP baseline** with a **transformer-based model** for sentiment analysis.

---

## 🌐 Live Demo

👉 [demo](https://impk-nlp-sentiment-lab-mjzsfnpfzkoofj3g8qkf3w.streamlit.app)

---

## 🧠 What this project does

This project classifies opinionated text and compares two approaches:

* 📦 **TF-IDF + Logistic Regression** — a strong classical baseline
* 🤖 **BERT** — a transformer-based sentiment model

It allows you to:

- Paste a review, tweet, or short paragraph
- Upload `.txt` or `.csv` files
- Compare both models side by side
- View accuracy and F1 score benchmarks
- Download a JSON comparison report

---

## 🎯 Why this matters

Text classification is one of the most common real-world NLP tasks.

This project shows the difference between:

- lightweight classical NLP pipelines
- modern transformer-based language models

👉 It demonstrates **model comparison, evaluation, and practical deployment** in one clean interface.

---

## 🚀 Features

* ✍️ Text input and file upload
* 🧹 Text cleaning pipeline
* 🧠 TF-IDF + Logistic Regression
* 🤖 BERT-based sentiment analysis
* 📊 Accuracy and F1 comparison
* 🔁 Side-by-side model comparison
* 💾 Downloadable JSON report
* 🌐 Interactive Streamlit app

---

## 🛠 Tech Stack

* Python
* Streamlit
* scikit-learn
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* pandas
* numpy

---

## ⚙️ How it works

1. **Input**  
   The user pastes text or uploads a file.

2. **Clean**  
   The text is normalized and prepared for inference.

3. **Predict**  
   Both models predict sentiment.

4. **Compare**  
   The app displays predictions and benchmark metrics.

---

## ▶️ Run locally

```bash
git clone https://github.com/dealmeidaferreiraAlexandra/impk-nlp-sentiment-lab.git
cd impk-nlp-sentiment-lab

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python train_traditional.py
python evaluate_models.py

streamlit run app.py

🧪 Notes
The traditional baseline is trained locally on IMDb data.
The BERT model uses the pretrained textattack/bert-base-uncased-imdb checkpoint.
First BERT run may take longer because the model is downloaded.
Evaluation metrics are computed on a representative IMDb test subset for reproducibility and speed.

👩‍💻 Author

Developed by Alexandra de Almeida Ferreira
GitHub: https://github.com/dealmeidaferreiraAlexandra
LinkedIn: https://www.linkedin.com/in/dealmeidaferreira

📄 License

This project is licensed under the MIT License.

