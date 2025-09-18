# PrimeNews - Fake News Predictor

PrimeNews is a Streamlit-based web application that predicts whether a news article is **Fake** or **True** using machine learning models trained on real-world news datasets. The app provides a simple, interactive interface for users to paste news text and get instant predictions from two different models.

## 🚀 Features
- **Fast, interactive web app** built with Streamlit
- **Dual-model prediction**: Logistic Regression & Decision Tree
- **Pre-trained models** for instant results (no retraining required)
- **Custom text input** for any news article
- **Modern UI** with background image and left-aligned title
- **No external API required** (all predictions are local)

## 🏗️ Project Structure
```
PrimeNews/
├── main.py                  # Streamlit app (run this file)
├── train_and_save_models.py # Script to train and save models/vectorizer
├── check_models.py          # Script to inspect saved models/vectorizer
├── Models/                  # Folder for saved models/vectorizer (not in Git)
│   ├── vectorizer.joblib
│   ├── lr_model.joblib
│   └── dtc_model.joblib
├── Dataset/                 # Folder for datasets (not in Git)
│   ├── True.csv
│   └── Fake.csv
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

## 🖥️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SanjeevKambale/PrimeNews.git
cd PrimeNews
```

### 2. Install dependencies
Make sure you have Python 3.8+ installed. Then run:
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install streamlit scikit-learn pandas joblib
```

### 3. Prepare models and data
- Place your `True.csv` and `Fake.csv` files in the `Dataset/` folder.
- Run `train_and_save_models.py` to train and save the models/vectorizer:
```bash
python train_and_save_models.py
```
- This will create the `Models/` folder with the required `.joblib` files.

### 4. Run the app
```bash
streamlit run main.py
```

## 📝 Usage
1. Open the app in your browser (Streamlit will provide a local URL).
2. Paste any news article text into the input box.
3. Click **Predict** to see if the news is likely **Fake** or **True**.
4. Use **Refresh Input** to clear the text area.

## ⚠️ Disclaimer
- This tool is for educational/demo purposes only. Model predictions are not 100% accurate.
- Always verify news from trusted sources.

## 📁 Files to Include in Git
- `main.py`, `train_and_save_models.py`, `check_models.py`, `.gitignore`, `README.md`
- **Do NOT include**: `Models/`, `Dataset/`, or any large data/model files.

## 🙏 Credits
- Built with [Streamlit](https://streamlit.io/), [scikit-learn](https://scikit-learn.org/), and open news datasets.
- Developed by [PrimeDev](https://github.com/SanjeevKambale).

---
© 2025 PrimeNews • Built with ❤️ by PrimeDev
