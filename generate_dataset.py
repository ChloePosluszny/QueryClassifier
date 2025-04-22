from openai import OpenAI

max_tokens = 8000

def load_prompt(prompt_file_path):
    """
    Load the prompt from a file.
    """
    with open(prompt_file_path, "r") as prompt_file:
        prompt = prompt_file.read()
    
    return prompt

def load_api(key_file_path):
    """
    Initialize API key and client.
    """
    with open(key_file_path, "r") as api_key_file:
        key = api_key_file.readline()
    
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
    
    return client

def generate_queries(client, prompt, num_queries=1000, temp = 0.5):
    """
    Generate queries for the model.
    """
    if (num_queries > 1000):
        raise ValueError("num_queries must be less than or equal to 1000.")
    
    # Append num_queries to the end of the prompt
    prompt += f"\n\nLet n be {num_queries} queries."
    
    response = client.chat.completions.create(
        model= "deepseek/deepseek-chat-v3-0324:free",
        input= prompt,
        max_tokens = max_tokens,
        temperature = temp, # adjust temperature for randomness/creativity (higher = more random/creative)
    )
    
    # print the generated dataset to confirm queries
    print(response.choices[0].message.content)
    
    correct = input("\nDid the dataset generate correctly? (y/n): ")
    
    if (correct == "y"):
        # Save the dataset to a file
        with open("data/generated_dataset.csv", "w") as dataset_file:
            dataset_file.write(response.choices[0].message.content)
    else:
        print("Datset creation failed.")
    
def main():
    # Load the API key
    client = load_api("secrets/api_key")
    
    prompt = load_prompt("prompts/generate_dataset.txt")
    
    # Generate queries
    generate_queries(client, prompt, num_queries = 1000, temp = 0.7)

if __name__ == "__main__":
    main()