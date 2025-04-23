from openai import OpenAI
import requests

max_tokens = 8000

def load_prompt(prompt_file_path):
    """
    Load the prompt from a file.
    """
    with open(prompt_file_path, "r") as prompt_file:
        prompt = prompt_file.read()
    
    return prompt

def load_client(key_file_path):
    """
    Initialize API key and client.
    """
    with open(key_file_path, "r") as api_key_file:
        key = api_key_file.readline()
    
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key, # key is for openrouter, limits to 50 queries a day
        )
    
    return client

def generate_queries(client, prompt, num_queries=500, temp = 0.5):
    """
    Generate queries for the model.
    """
    if (num_queries > 500 or num_queries < 1):
        raise ValueError("num_queries must be less than or equal to 1000 and above 0.")
    
    # Append num_queries to the end of the prompt
    prompt += f"\n\nLet n be {num_queries} queries."
    
    print("Generating dataset...")
    
    i = 1
    response = None
    while (i <= 15 and (response is None or response.choices is None or response.choices[0].message.content == "" or response.choices[0].message.content == "Error")):
        # Wait for api to return a message
        print(f"\nAttempt #{i}")
        
        response = client.chat.completions.create(
            model= "deepseek/deepseek-chat-v3-0324:free", # Deepseek has no rate limits and times out after 30 minutes
            messages = 
            [{
                "role": "system",
                "content": "You are a helpful assistant that generates queries for a dataset. You follow directions exactly and you respond in the exact format requested with no additional formatting or quotes."
            },
            {
                "role": "user",
                "content": prompt,
            },],
            max_tokens = max_tokens,
            temperature = temp, # adjust temperature for randomness/creativity (higher = more random/creative)
            response_format= 
            {
                'type': 'json_object'
            },
        )
        
        i += 1
    
    # print the generated dataset to confirm queries
    print("\n" + response.choices[0].message.content)
    
    correct = input("\nDid the dataset generate correctly? (y/n): ")
    
    if (correct == "y"):
        # Save the dataset to a file
        with open("data/generated_dataset.json", "w") as dataset_file:
            dataset_file.write(response.choices[0].message.content)
    else:
        print("Datset creation failed.")
        
        print("\nResponse Metadata:")
        print(f"Model: {response.model}")
        print(f"Token Usage: {response.usage}")
        print(f"Finish Reason: {response.choices[0].finish_reason}")
    
def main():
    # Load the API key
    client = load_client("secrets/api_key_2")
    
    prompt = load_prompt("prompts/generate_dataset.txt")
    
    # Generate queries
    generate_queries(client, prompt, num_queries = 250, temp = 0.7) # num_queries does not always work

if __name__ == "__main__":
    main()