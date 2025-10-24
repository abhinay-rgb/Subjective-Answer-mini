# 📘 Subjective Answer Evaluator

A smart and intuitive **Streamlit web application** for evaluating subjective answers using AI-powered techniques such as semantic similarity, grammar checks, and keyword matching.

This tool simplifies academic grading by assisting teachers with automated evaluations and giving students meaningful feedback.

## 🚀 Features

- 🔐 User roles: Admin, Teacher, Student
- 🧠 Semantic similarity scoring with **Sentence Transformers**
- ✍️ Grammar correction using **TextBlob**
- 🔑 Keyword matching with synonym support (via NLTK WordNet)
- 📂 Bulk assignment creation using CSV uploads
- 📊 Real-time evaluation and feedback
- 📉 Performance analytics dashboard
- ⚠️ Plagiarism detection among student submissions
- 🗃️ Local database storage using SQLite

---

## 📦 Required Dependencies

Before running the app, ensure you have the following Python libraries installed:

```bash
pip install streamlit pandas sentence-transformers nltk bcrypt textblob
```

Additionally, download necessary NLTK corpora:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🛠️ How to Run

Make sure you're in the project directory, then start the app with:

```bash
streamlit run code.py
```

This will launch the app in your default web browser at [http://localhost:8501](http://localhost:8501).

---

## 🧾 Bulk Upload CSV Format

To create multiple assignments at once, upload a CSV with these columns:

- `subject`
- `question`
- `model_answer`
- `keywords`
- `deadline` (optional, format: `DD/MM/YYYY` or `DD-MM-YYYY`)

---

## 📁 Project Files

```
subjective-evaluator/
├── ak.py                  # Main application code
├── evaluator.db           # Local SQLite database (auto-created)
├── csv file for bulk.csv  # Sample CSV for testing bulk upload
└── README.md              # Project documentation
```

---

## 👨‍💻 Author

**Abhinay Kumar**  
GitHub: [@Abhinay](https://github.com/abhinay-rgb)


---

## 🪪 License

This project is open-source and available under the MIT License. 