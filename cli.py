"""Command-line interface.

Usage:
    python cli.py ingest              # index everything in data/docs
    python cli.py chat                # interactive chat in the terminal
    python cli.py chat --voice        # also save each answer as an mp3 in data/audio
"""
import argparse

from app.llm import LLMError
from app.rag import RAGPipeline


def run_chat(voice: bool) -> None:
    pipeline = RAGPipeline()
    history = []
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if len(question) < 3:
            continue
        try:
            result = pipeline.ask(question, history=history)
        except LLMError as exc:
            print(f"Error: {exc}\n")
            continue

        print(f"\nBot: {result['answer']}")
        for s in result["sources"]:
            page = f", p. {s['page']}" if s["page"] else ""
            print(f"  - {s['source']}{page} (score {s['score']})")
        if voice and result["sources"]:
            from app.tts import text_to_speech

            try:
                print(f"  Audio saved to: {text_to_speech(result['answer'])}")
            except Exception as exc:
                print(f"  Text-to-speech failed: {exc}")
        print()

        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Index all documents in the docs folder")
    chat = sub.add_parser("chat", help="Start an interactive chat")
    chat.add_argument("--voice", action="store_true", help="Save spoken answers as mp3 files")
    args = parser.parse_args()

    if args.command == "ingest":
        print(RAGPipeline().ingest_folder())
    else:
        run_chat(args.voice)


if __name__ == "__main__":
    main()
