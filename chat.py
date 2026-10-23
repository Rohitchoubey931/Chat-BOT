"""
Simple RAG Chatbot (beginner level)
-----------------------------------
This chatbot answers your questions using a text file (knowledge_base.txt).

How it works (RAG = Retrieval + Answer):
1. LOAD     -> read the text file and split it into small chunks (paragraphs)
2. INDEX    -> convert every chunk into numbers using TF-IDF
3. RETRIEVE -> convert your question into numbers and find the most similar chunks
4. ANSWER   -> reply with the best matching chunk(s)
5. SPEAK    -> (optional) read the answer out loud

Run it with:  python3 chat.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------- Settings ------------------------------------
KNOWLEDGE_FILE = "knowledge_base.txt"  # the file the bot learns from
TOP_K = 2        # how many matching chunks to use in the answer
MIN_SCORE = 0.10  # below this similarity, we say "I don't know"


# ----------------------------- Step 1: LOAD --------------------------------
def load_chunks(path):
    """Read the file and split it into chunks. A blank line separates chunks."""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = []
    for part in text.split("\n\n"):
        part = part.strip()
        if part:  # ignore empty parts
            chunks.append(part)
    return chunks


# ----------------------------- Step 2: INDEX -------------------------------
def build_index(chunks):
    """Turn each chunk into a vector (list of numbers) using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks)
    return vectorizer, matrix


# ----------------------------- Step 3: RETRIEVE ----------------------------
def retrieve(question, chunks, vectorizer, matrix, top_k=TOP_K):
    """Find the chunks most similar to the question."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).flatten()

    # indexes of chunks sorted from best to worst score
    best_indexes = scores.argsort()[::-1][:top_k]

    results = []
    for i in best_indexes:
        if scores[i] >= MIN_SCORE:
            results.append((i + 1, chunks[i], scores[i]))  # (chunk number, text, score)
    return results


# ----------------------------- Step 4: ANSWER ------------------------------
def make_answer(results):
    """Build the reply from the retrieved chunks."""
    if not results:
        return "Sorry, I could not find that in my knowledge base. Try asking in a different way."

    answer = ""
    for chunk_number, text, score in results:
        answer += f"{text}\n   [source: chunk {chunk_number}, match {score:.2f}]\n\n"
    return answer.strip()


# ----------------------------- Step 5: SPEAK -------------------------------
def speak(text):
    """Read the text out loud. If voice is not set up, print a small message."""
    try:
        import pyttsx3  # imported here so the bot still works without it

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)  # speaking speed
        engine.say(text)
        engine.runAndWait()
    except Exception:
        print("(Voice is not available. Check the README troubleshooting section.)")


def clean_for_voice(results):
    """For voice we read only the text, not the '[source...]' tags."""
    return " ".join(text for _, text, _ in results)


# ----------------------------- Main chat loop ------------------------------
def main():
    try:
        chunks = load_chunks(KNOWLEDGE_FILE)
    except FileNotFoundError:
        print(f"Could not find '{KNOWLEDGE_FILE}'. Put it in the same folder as chat.py.")
        return

    vectorizer, matrix = build_index(chunks)
    voice_on = False

    print("=" * 55)
    print("  Simple RAG Chatbot")
    print(f"  Loaded {len(chunks)} chunks from {KNOWLEDGE_FILE}")
    print("  Commands: 'voice on', 'voice off', 'quit'")
    print("=" * 55)

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye!")
            break
        if question.lower() == "voice on":
            voice_on = True
            print("Bot: Voice is ON.")
            continue
        if question.lower() == "voice off":
            voice_on = False
            print("Bot: Voice is OFF.")
            continue

        results = retrieve(question, chunks, vectorizer, matrix)
        answer = make_answer(results)
        print(f"\nBot: {answer}")

        if voice_on:
            if results:
                speak(clean_for_voice(results))
            else:
                speak(answer)


if __name__ == "__main__":
    main()