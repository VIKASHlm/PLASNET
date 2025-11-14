import subprocess
import json

def query_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Send a prompt to Ollama and get response.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error querying Ollama: {e}"


def mix_with_llm(mix_result: dict, model: str = "llama3") -> str:
    """
    Take plastiphalt mix dictionary and ask LLM to provide recommendations.
    """
    prompt = f"""
    You are an expert in plastic-infused road construction.
    Given the following mix design result:

    {json.dumps(mix_result, indent=2)}

    Explain:
    - What plastics are used and why
    - Expected durability
    - Recommendations for road engineers
    - Warnings (if any)

    Provide a professional but easy-to-understand summary.
    """

    return query_ollama(prompt, model=model)
