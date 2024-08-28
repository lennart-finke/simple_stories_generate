import json
from datetime import datetime
import tiktoken
import os
from generate_stories import create_simple_story_prompt, get_random_params

def get_batch_dataset(num_completions, model):
    lines = []
    total_tokens = 0
    enc = tiktoken.encoding_for_model(model)

    for i in range(num_completions):
        params = get_random_params()
        prompt = create_simple_story_prompt(params)
        message_tokens = len(enc.encode(prompt))
        total_tokens += message_tokens
        messages=[{"role": "user", "content": prompt}]

        lines.append(json.dumps(
            {"custom_id": str(i)+json.dumps(params),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model, "messages": messages}}
            ))
    
    print(f"Total input tokens: {total_tokens}")

    return lines


def main(num_completions: int, model = "gpt-4o-mini"):
    if not os.path.exists("data"):
        os.makedirs("data")
    lines = get_batch_dataset(num_completions, model)

    with open(f"data/{model}_batch_completion_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.jsonl", "w") as fp:
        fp.write("\n".join(lines))

if __name__ == '__main__':
    main(20)
