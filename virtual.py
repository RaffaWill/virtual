import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = {
    "max_new_tokens": 512,     # Maximum length of the response
    "return_full_text": False,      # Only return the generated text
}


def query(messages):
    """Sends a conversation history to the AI assistant and returns the answer.

    Args:
      messages (list): A list of dictionaries, each with "role" and "content" keys.

    Returns:
      str: The answer from the AI assistant.
    """

    output = pipe(messages, **generation_args)
    return output[0]['generated_text']


def chat():
    """Enables interactive chat sessions with the AI assistant."""

    conversation_history = [
        {"role": "system",
         "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user."}
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = query(conversation_history)
            print("Assistant: ", response)

            conversation_history.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"An error occurred: {e}, please try again.")
chat()