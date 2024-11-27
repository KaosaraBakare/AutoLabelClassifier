import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def generate_summary(model, tokenizer, prompt, device):
    """
    Generate a summary of the report.
    """
    # Ensure input is truncated if too long
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
    
    print(f"Input IDs: {input_ids}")  # Debugging line to check tokenized input
    
    gen_ids = model.generate(input_ids=input_ids, max_new_tokens=100, repetition_penalty=1.15)
    output = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return output.strip()


@torch.no_grad()
def generate_probability_present(model, tokenizer, device, YES_ID, NO_ID, eval_prompt):
    """
    Generate the probability for the presence of the condition.
    """
    # Tokenize the prompt, ensuring truncation to model max length
    input_ids = tokenizer(eval_prompt, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
    
    # Run the model to get logits for the last token
    model_out = model(input_ids)
    final_output_token_logits = model_out.logits[0, -1]
    
    # Extract the logits for YES and NO tokens
    yes_score = final_output_token_logits[YES_ID].cpu().item()
    no_score = final_output_token_logits[NO_ID].cpu().item()
    
    # Perform softmax to calculate probabilities
    p_present = torch.nn.functional.softmax(torch.tensor([yes_score, no_score]), dim=0)[0].item()
    return p_present


def format_prompt(messages):
    """
    Format chat messages for GPT-2 as a plain string.
    """
    formatted = ""
    for message in messages:
        formatted += f"{message['role'].upper()}:\n{message['content']}\n\n"
    return formatted


def main(
    condition: str,
    definition: str,
    data: str,
    output: str,
    model_name: str,
    transformers_cache: str,
    threshold: float = 0.5,
    device="cuda:0",
) -> None:
    """
    Main function for labeling the reports.
    """
    # Set cache location
    import os
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define tokens
    YES_TOKEN = "yes"
    NO_TOKEN = "no"
    YES_ID = tokenizer.convert_tokens_to_ids(YES_TOKEN)
    NO_ID = tokenizer.convert_tokens_to_ids(NO_TOKEN)

    # Load data
    df = pd.read_csv(data, low_memory=False, index_col=0)

    outputs = []
    for index, sample in tqdm(df.iterrows(), total=len(df)):
        report = sample.report
        identifier = index

        # Create prompt messages
        messages = [
            {"role": "system", "content": f"You are diagnosing {condition} using the following report."},
            {"role": "user", "content": f"Report:\n{report}"},
            {"role": "user", "content": f"Write a summary focusing on findings related to {condition}. Definition: {definition}"},
        ]
        eval_prompt = format_prompt(messages)

        # Generate the summary
        summary = generate_summary(model, tokenizer, eval_prompt, device)

        # Prepare the follow-up prompt
        follow_up_prompt = (
            f"{eval_prompt}\n\nASSISTANT:\n{summary}\n\nUSER:\nDoes the patient have {condition}? "
            f"Answer 'yes' or 'no'. Only output one token after 'ANSWER: '.\n\nASSISTANT:\nANSWER: "
        )
        
        # Generate probability of the condition being present
        p_present = generate_probability_present(model, tokenizer, device, YES_ID, NO_ID, follow_up_prompt)

        # Determine the prediction based on the probability
        prediction = "yes" if p_present > threshold else "no"

        # Store the results
        outputs.append(
            {
                "id": identifier,
                "report": report,
                "summary": summary,
                "probability": p_present,
                "prediction": prediction,
            }
        )

    # Save the output to a CSV file
    df_out = pd.DataFrame(outputs)
    df_out.to_csv(output, index=False)
    print(f"Results saved to {output}")
