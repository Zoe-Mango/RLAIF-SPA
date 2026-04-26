import pandas as pd
import json
from openai import OpenAI

# Initialize the custom OpenAI client with API key and base URL
client = OpenAI(api_key="your_key", base_url="https://www.jcapikey.com/v1")

def generate_labels_for_text(text, text_id, output_file_path):
    sentences = text.split('. ')  # Split text based on sentence-ending punctuation
    all_labels = []
    
    # Prepare prompt template for sentence-by-sentence reasoning without explanation
    prompt_template = """
    Please choose the most appropriate label from the four categories for the sentence below:

    1. **Structure Label**: Choose from the following options:
        - Introduction
        - Background
        - Argument
        - Conclusion
        - Transition
        - Example
        - Problem Statement
        - Methodology
        - Data Analysis
        - Summary
        - Greeting/Introduction
        - Request/Proposal
        - Statement
        - Question
        - Exclamation/Reaction
        - Advice/Recommendation
        - Clarification
        - Farewell
        - Complaint
        - Apology

    2. **Emotion Label**: Choose from the following options:
        - Positive
        - Negative
        - Neutral
        - Excited
        - Worried
        - Desperate
        - Angry
        - Expectant
        - Confused
        - Optimistic
    
    3. **Speech Speed Label**: Choose from the following levels:
        - Extremely Slow
        - Slow
        - Medium Slow
        - Medium Fast
        - Extremely Fast

    4. **Tone Label**: Choose from the following options:
        - Declarative
        - Interrogative
        - Imperative
        - Exclamation
        - Rhetorical Question
        - Command

    The sentence is:
    "{sentence}"

    Please output only the labels as follows:
    1. Structure
    2. Emotion
    3. Speech Speed
    4. Tone
    """

    with open(output_file_path, 'a') as f:
        for i, sentence in enumerate(sentences):
            prompt = prompt_template.format(text=text, sentence=sentence.strip(), text_id=text_id)
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="gpt-4o-2024-11-20", 
            )
            
            labels = response.choices[0].message.content.strip()
            
            print(f"Sentence {i+1}: {sentence.strip()}")
            print(f"Labels: {labels}")
            
            sentence_labels = {
                "sentence_number": i + 1,
                "sentence": sentence.strip(),
                "labels": labels
            }
            
            json.dump({"id": text_id, "sentence_labels": sentence_labels}, f, indent=4)
            f.write("\n") 
            
            all_labels.append(sentence_labels)

    return all_labels


# Load the CSV file containing the text data
csv_file_path = '/datasets/librispeech_clean_test-clean_texts.csv'
df = pd.read_csv(csv_file_path)

# Check the first few rows to see how the data is structured
print(df.head())

# Assuming the CSV has columns 'id' and 'text'
output_results = []

# Specify output file path
output_file_path = '/datasets/librispeech_clean_test-clean_processed_labels.json'

# Iterate over each row in the DataFrame, process the text, and store results
for idx, row in df.iterrows():
    text = row['text']  # Text column from the CSV
    text_id = row['id']  # Use the 'id' column from the CSV as the ID
    print(f"Processing text {text_id}")
    
    # Generate the labels for this text
    result = generate_labels_for_text(text, text_id, output_file_path)
    
    # Store the result
    output_results.append(result)

print(f"Processing complete. Results saved to {output_file_path}")
