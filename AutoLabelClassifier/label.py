import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def generate_summary(model, tokenizer, prompt, device, ASSISTANT_HEADER):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    gen_ids = model.generate(
        input_ids=input_ids, max_new_tokens=1024, repetition_penalty=1.15
    )[0]
    output = tokenizer.decode(gen_ids, skip_special_tokens=False)
    answer = output.split(ASSISTANT_HEADER)[-1].replace("<|eot_id|>", "")
    return answer.strip()


@torch.no_grad()
def generate_probability_present(model, tokenizer, device, YES_ID, NO_ID, eval_prompt):
    input_ids = tokenizer(eval_prompt, return_tensors="pt")["input_ids"].to(device)
    model_out = model(input_ids)
    final_output_token_logits = model_out.logits[0][-1]
    yes_score = final_output_token_logits[YES_ID].cpu().item()
    no_score = final_output_token_logits[NO_ID].cpu().item()
    # Perform softmax over "yes" or "no"
    p_present = torch.nn.functional.softmax(
        final_output_token_logits[[YES_ID, NO_ID]], dim=0
    ).cpu()[0].item()
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
    Main function for labelling the reports.
    """
    # Print arguments
    print(f"Running Labelling for {condition} with the following args:")
    print(f"Data: {data}")
    print(f"Model: {model_name}")
    print(f"Definition: {definition}")
    print(f"Transformers Cache: {transformers_cache}")

    # Set cache location
    import os

    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    print("Using cache at:", os.getenv("TRANSFORMERS_CACHE"))

    # Load model and tokenizer
    print(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.half()
    model.to(device)
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        trust_remote_code=True,
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=1024,
    )

    YES_TOKEN = "yes"
    NO_TOKEN = "no"
    YES_ID = tokenizer(YES_TOKEN, add_special_tokens=False).input_ids[0]
    NO_ID = tokenizer(NO_TOKEN, add_special_tokens=False).input_ids[0]

    ASSISTANT_HEADER = "ASSISTANT:"

    prompt1 = f"You are a radiologist. Your job is to diagnose {condition} using a medical report. Tell the truth and answer as precisely as possible."

    # Load data
    df = pd.read_csv(data, low_memory=False, index_col=0)
    print(f"Loaded in data with shape: {df.shape}")

    print("Making prompts...")
    outputs = []
    for index, sample in tqdm(df.iterrows(), total=len(df)):
        example = sample.report
        identifier = index

        # Create the prompt messages
        messages = [
            {"role": "system", "content": f"{prompt1}\nReport: {example}"},
            {
                "role": "user",
                "content": f"Write a summary for the above report, focusing on findings related to {condition}, according to this definition: {definition}",
            },
        ]
        eval_prompt = format_prompt(messages) + f"\n{ASSISTANT_HEADER}\n"
        gen_summary = generate_summary(
            model, tokenizer, eval_prompt, device, ASSISTANT_HEADER
        )

        # Add additional query for probability generation
        messages += [
            {"role": "assistant", "content": f"{gen_summary}"},
            {
                "role": "user",
                "content": f"Based on your summary, does the patient have {condition}? Answer 'yes' for yes, 'no' for no. Only output one token after 'ANSWER: '",
            },
        ]
        eval_prompt = format_prompt(messages) + f"\n{ASSISTANT_HEADER}\nANSWER: "

        # Calculate probability and prediction
        p_present = generate_probability_present(
            model, tokenizer, device, YES_ID, NO_ID, eval_prompt
        )
        prediction = "yes" if p_present > threshold else "no"

        print("Report:", example)
        print("Summary:", gen_summary)
        print("Probability:", p_present)
        print("Disease Present Prediction:", prediction)

        outputs.append(
            {
                "id": identifier,
                "report": example,
                "summary": gen_summary,
                "probability": p_present,
                "prediction": prediction,
            }
        )

    # Save results
    df_out = pd.DataFrame(outputs)
    df_out.to_csv(output, index=False)
    print(f"Results saved to {output}")
