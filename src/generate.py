"""Answer questions with Gemma 2B (Ollama) using chunks from hybrid search."""
import ollama

from retrieve import Retriever

LLM = "gemma2:2b"
ANSWER_OPTIONS = {"temperature": 0, "num_predict": 400, "num_ctx": 4096}
TOP_K_CONTEXT = 4



ANSWER_PROMPT = """You are a technical assistant for Unreal Engine documentation.

Use ONLY the information explicitly supported by the documentation excerpts
provided below.

Rules:
- Do not answer from your general knowledge or training data.
- First determine whether the excerpts contain enough information to answer
  the question.
- If the excerpts do not contain the answer, respond exactly:
  "I could not find that in the documentation."
- Do not fill gaps with assumptions or information you already know.
- Explain the documented information clearly in your own words.
- Preserve exact names of nodes, pins, checkboxes, and features.
- Every factual claim must be supported by the excerpts.
- Cite supporting excerpts using their IDs, such as [1] or [2].
- Never invent excerpt IDs.

Documentation excerpts:
{context}

Question: {question}

Answer:"""



QUESTIONS = [
    "How does the FlipFlop node alternate its outputs?",
    "What does the Loop checkbox do on a Timeline?",
    "How do I create a new math expression blueprint to put complex formulas in?",
    "What's the purpose of the random pin in the multi gate, what does it do",
    "If I press the F7 button on a blueprint, what will happen",
    "how to compile a blueprint",
    "what's the data type of the output of get play rate",
    "how to iterate through all the actors of a certain class in the level blueprint",
    "what is the keyboard shortcut to search blueprints",
    "list me all the searchable tags within the nodes filter",
    "How can one Blueprint notify another Blueprint when something happens?",
    "How can I reuse a sequence of Blueprint nodes across multiple Blueprints without having to recreate it each time?"
]



import re


def build_context(hits: list) -> str:
    excerpts = []

    for i, hit in enumerate(hits, start=1):
        excerpts.append(
            f'[Excerpt {i} | pages {hit["pages"]}]\n'
            f'{hit["return_text"]}'
        )

    return "\n\n".join(excerpts)


def answer(
    question: str,
    retriever: Retriever,
    mode: str = "hybrid",
    k: int = TOP_K_CONTEXT,
) -> dict:
    hits = retriever.search(question, mode, top_k=k)

    prompt = ANSWER_PROMPT.format(
        context=build_context(hits),
        question=question,
    )

    response = ollama.chat(
        model=LLM,
        messages=[{"role": "user", "content": prompt}],
        options=ANSWER_OPTIONS,
    )

    raw_answer = response["message"]["content"].strip()

    # Map each valid excerpt ID to its original page metadata.
    excerpt_pages = {
        i: hit["pages"]
        for i, hit in enumerate(hits, start=1)
    }

    used_ids = []
    for match in re.finditer(r"\[(\d+)\]", raw_answer):
        excerpt_id = int(match.group(1))
        if excerpt_id in excerpt_pages and excerpt_id not in used_ids:
            used_ids.append(excerpt_id)

    # Convert valid excerpt citations into page citations.
    def replace_citation(match):
        excerpt_id = int(match.group(1))

        if excerpt_id not in excerpt_pages:
            return ""

        pages = excerpt_pages[excerpt_id]
        if isinstance(pages, (list, tuple)):
            page_text = ", ".join(str(page) for page in pages)
        else:
            page_text = str(pages)

        return f"(p. {page_text})"

    final_answer = re.sub(
        r"\[(\d+)\]",
        replace_citation,
        raw_answer,
    )

    # Show only pages from excerpts the model actually cited.
    cited_pages = []
    for excerpt_id in used_ids:
        pages = excerpt_pages[excerpt_id]
        if not isinstance(pages, (list, tuple)):
            pages = [pages]

        for page in pages:
            if page not in cited_pages:
                cited_pages.append(page)

    return {
        "answer": final_answer,
        "sources": cited_pages,
        "prompt_tokens": response["prompt_eval_count"],
    }




def main() -> None:
    retriever = Retriever()

    print("Documentation Q&A (Gemma 2B)")
    print("Enter a question, or type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            result = answer(question, retriever)

            print(f"\nAnswer:\n{result['answer']}")
            print(f"Sources: {result['sources']}")
            print(
                f"Prompt tokens: {result['prompt_tokens']} / "
                f"{ANSWER_OPTIONS['num_ctx']}"
            )

        except Exception as e:
            print(f"Error while answering: {e}")


if __name__ == "__main__":
    main()