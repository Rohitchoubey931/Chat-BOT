# Simple RAG Chatbot with Voice Output

A beginner-friendly chatbot that answers questions from your own text file and can read the answers out loud.

It uses the main idea behind **RAG (Retrieval-Augmented Generation)**: first *retrieve* the most relevant information from a knowledge base, then use it to *answer*. The code is short, heavily commented, and runs fully offline. No API key is needed.

---

## Features

- Answers questions using your own `knowledge_base.txt`
- Shows which chunk each answer came from, with a match score
- Says "I don't know" when nothing relevant is found, instead of making things up
- Optional voice output (type `voice on` to enable)
- Only about 130 lines of beginner-level Python

---

## Project Structure

```
simple-rag-chatbot/
├── chat.py          # the chatbot code
├── knowledge_base.txt  # the text the chatbot learns from
├── requirements.txt    # Python libraries needed
├── .gitignore          # files Git should not upload
├── LICENSE             # MIT license
└── README.md           # this file
```

---

## How It Works

```
 knowledge_base.txt
        │
        ▼
 [1] LOAD      split the text into chunks (one paragraph = one chunk)
        │
        ▼
 [2] INDEX     convert every chunk into numbers (TF-IDF vectors)
        │
 You ask a question
        │
        ▼
 [3] RETRIEVE  convert the question into numbers, compare with all
               chunks (cosine similarity), pick the top matches
        │
        ▼
 [4] ANSWER    reply with the best chunk(s) + source and score
        │
        ▼
 [5] SPEAK     (optional) read the answer out loud with pyttsx3
```

### Step by step

**1. Load and chunk.**
The bot reads `knowledge_base.txt` and splits it wherever there is a blank line. Each paragraph becomes one *chunk*. Small chunks let the bot find the exact part that answers your question instead of returning a whole document.

**2. Index (TF-IDF).**
A computer cannot compare sentences directly, so we convert each chunk into a list of numbers. This project uses **TF-IDF**, which gives a word a high score when it appears often in one chunk but rarely in the others. For example, the word "pyttsx3" is rare, so it strongly points to the chunk about voice.

**3. Retrieve (cosine similarity).**
Your question is converted into numbers in the same way. The bot then measures how close the question is to each chunk using **cosine similarity** (0 means nothing in common, 1 means very similar). The top 2 chunks above the minimum score are selected.

**4. Answer.**
The selected chunks are shown as the reply along with the chunk number and match score. If no chunk scores above `MIN_SCORE`, the bot says it could not find the answer.

**5. Speak (optional).**
If voice is on, the answer text is passed to **pyttsx3**, an offline text-to-speech library, which reads it aloud.

> Note: this beginner version does retrieval and shows the retrieved text as the answer. A full RAG system adds a language model (LLM) that rewrites the retrieved text into a natural answer. See "Next Steps" below.

---

## Setup

### 1. Requirements
- Python 3.9 or newer
- pip

### 2. Clone the repo
```bash
git clone https://github.com/<your-username>/simple-rag-chatbot.git
cd simple-rag-chatbot
```

### 3. Create a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install the libraries
```bash
python3 -m pip install -r requirements.txt
```

### 5. Run the chatbot
```bash
python3 chat.py
```

---

## Usage

| What you type | What happens |
|---|---|
| any question | the bot searches the knowledge base and replies |
| `voice on` | answers are also read out loud |
| `voice off` | turns voice off |
| `quit` | exits the chatbot |

### Example conversation

```
You: what is RAG?

Bot: RAG stands for Retrieval-Augmented Generation. It is a technique where
a chatbot first searches a set of documents ... 
   [source: chunk 1, match 0.35]

You: voice on
Bot: Voice is ON.

You: what is cosine similarity
Bot: Cosine similarity is a way to measure how close two vectors are ...
   [source: chunk 5, match 0.27]
(the answer is also spoken)

You: who won the world cup
Bot: Sorry, I could not find that in my knowledge base. Try asking in a different way.
```

---

## Customize It

### Use your own knowledge
1. Open `knowledge_base.txt`.
2. Replace the content with your own text (notes, FAQ, syllabus, resume details, anything).
3. Put **one blank line** between paragraphs. Each paragraph becomes one chunk.
4. Save and run `python3 chat.py` again.

Tips: keep each paragraph about one topic, and write 2-5 sentences per paragraph.

### Change the settings
At the top of `chat.py`:

| Setting | Meaning | Try |
|---|---|---|
| `TOP_K` | how many chunks to show | 1 for short answers, 3 for more detail |
| `MIN_SCORE` | minimum match needed to answer | raise it (0.2) for stricter answers, lower it (0.05) to answer more often |
| `rate` in `speak()` | speaking speed | 150 (slower) to 200 (faster) |

---

## Troubleshooting

**"Voice is not available" message**
- Make sure you ran `pip install -r requirements.txt`.
- **macOS:** pyttsx3 needs `pyobjc-framework-Cocoa`, which is already in `requirements.txt`. If it still fails, run `pip install pyobjc-framework-Cocoa`.
- **Linux:** install the speech engine with `sudo apt install espeak`.
- **Windows:** it works with the built-in voices, so no extra setup is usually needed.

**"No module named 'sklearn'"**
Python cannot see the libraries. Install them with the same Python that runs the bot:
```bash
python3 -m pip install -r requirements.txt
```
If your terminal line starts with `(venv)` or `(.venv)`, install while the environment is active.

**"Could not find knowledge_base.txt"**
Run the command from inside the project folder, or check that the file name is spelled correctly.

**The bot gives a wrong or odd answer**
TF-IDF matches *words*, not *meaning*. If you ask using different words than the text uses, it can miss. Try rephrasing, or add more detail to your knowledge base. Upgrading to embeddings (below) solves this properly.

---

## Limitations

- Matches keywords, not meaning, so synonyms may not match.
- Returns the stored text as-is, not a newly written answer.
- No memory of earlier messages in the chat.
- Voice output only; you type your questions.

---

## Next Steps (upgrade ideas)

1. **Better retrieval:** replace TF-IDF with `sentence-transformers` embeddings so it understands meaning.
2. **Vector database:** store embeddings in ChromaDB or FAISS for larger knowledge bases.
3. **Add an LLM:** send the question and the retrieved chunks to an LLM API and ask it to write the final answer using only that context. This makes it a complete RAG system.
4. **Read PDFs:** load documents with PyPDF instead of a plain text file.
5. **Voice input:** add speech-to-text with Whisper so you can talk to the bot.
6. **Web interface:** wrap it in FastAPI or Streamlit with a text box and audio player.
7. **Chat memory:** keep the last few messages so follow-up questions work.

---

## Tech Used

- Python
- scikit-learn (TF-IDF and cosine similarity)
- pyttsx3 (offline text-to-speech)

---

## License

This project is open source under the MIT License (see the `LICENSE` file). Feel free to use and modify it.
