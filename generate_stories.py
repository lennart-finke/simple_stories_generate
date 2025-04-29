import random
import itertools
import json
import os
from openai import OpenAI
import hashlib
import time
import anthropic
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import textwrap

from text_data import themes, topics, styles, features, word_types, grammars, personas, letter_frequencies, LANGUAGE, END_STRING

MAX_STORIES_PER_COMPLETION = 30

class RateLimitException(Exception):
    pass


def get_random_params():
    letters = list(letter_frequencies.keys())
    weights = list(letter_frequencies.values())
    random_letter = random.choices(letters, weights=weights, k=1)[0]
    start_word_type = random.choice(word_types)

    grammar = random.choice(grammars)
    persona = random.choice(personas)
    if random.random() < 0.5:
        grammar = ""
    if random.random() < 0.5:
        persona = ""
    return {
        "theme": random.choice(themes),
        "topic": random.choice(topics),
        "style": random.choice(styles),
        "feature": random.choice(features),
        "initial_letter": random_letter,
        "initial_word_type": start_word_type,
        "grammar": grammar,
        "persona": persona,
        "num_paragraphs": random.randint(1, 9),
    }

def iterate_params(seed=42):
    # A slightly hacky way to yield all combinations of parameters, but having empty "grammar" and "persona" values half and a third of the time.
    random.seed(seed)
    
    # Generate a letter pool with frequencies proportional to the prime number
    letter_pool = [
        letter
        for letter, frequency in letter_frequencies.items()
        for _ in range(int(frequency * 997 / 100))
    ]
    random.shuffle(letter_pool)

    # Generate all combinations of the non-letter parameters and shuffle
    combinations = list(itertools.product(themes, styles, features, word_types))
    print(f"Using {len(combinations)} combinations...")
    random.shuffle(combinations)
    
    assert len(grammars) % 2 != 0, "Number of grammars should be odd"
    assert len(personas) % 3 != 0, "Number of personas should not be divisible by 3"
    assert len(topics) % 2 != 0 and len(topics) % 3 != 0

    k = 0
    while True:
        theme, style, feature, word_type = combinations[k % len(combinations)]
        random_letter = letter_pool[k % len(letter_pool)]  # Cycle through shuffled letter pool
        
        # Grammar and persona are set to empty strings half, and a third of the time respectively
        grammar = grammars[k % len(grammars)] if k % 2 == 0 else ""
        persona = personas[k % len(personas)] if k % 3 == 0 else ""
        topic = topics[k % len(topics)]

        yield {
            "theme": theme,
            "topic": topic,
            "style": style,
            "feature": feature,
            "grammar": grammar,
            "persona": persona,
            "initial_letter": random_letter,
            "initial_word_type": word_type,
            "num_paragraphs": 1 + (k % 9),
        }
        k += 1

def create_simple_story_prompt(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])
    singular = params['num_paragraphs'] == 1

    if LANGUAGE == "en":
        template_singular = textwrap.dedent(f"""\
            Write a short story ({params['num_paragraphs']} paragraphs) using very basic words.
            The story """)
        template_plural = textwrap.dedent(f"""\
            Write {num_stories_per_completion} short stories ({params['num_paragraphs']} 
            paragraph{'' if singular else 's'} each) using very basic words. Do not number
            each story or write a headline. Make the stories diverse by fully exploring the
            theme, but each story should be self-contained. 
            Separate the stories by putting {END_STRING} in between. Make the stories as 
            qualitatively distinct to each other as possible. In particular, never start two
            stories the same way!
            Each story """)
        template = textwrap.dedent("""\
            should be about {theme}, include {topic}, be {style} in its writing style and 
            ideally feature {feature}.{grammar}{persona} If you need to use proper names, 
            make them from space-separated common words. Either don't give characters a name, 
            or select from Mia, Alex, Jean, Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, 
            Emmanuel, Anne, Peter, Maria or Luis. Complex story structure is great, but 
            please remember to only use very simple words! If you can, start the story
            with {initial_word_type} that begins with the letter {initial_letter}.""")
        if singular:
            template = template_singular + template
        else:
            template = template_plural + template
        
        params = params.copy()
        if params['grammar']:
            params['grammar'] = textwrap.dedent(f"""\
                The most important thing is to write an engaging easy story, but where it 
                makes sense, demonstrate the use of {params['grammar']}.""")
        if params['persona']:
            params['persona'] = f" Write from the perspective of {params['persona']}."
    elif LANGUAGE == "ja":
        num_chars = params["num_paragraphs"] * 100
        singular = num_stories_per_completion == 1
        template_singular = f"非常に簡単な単語を使用して、子供でも分かるような約{num_chars}文字の物語を書いてください。"
        template_plural = f"非常に簡単な単語を使用して、子供でも理解できるような約{num_chars}文字の物語を{num_stories_per_completion}個書いてください。各物語に番号やタイトルをつけないでください。テーマを十分に探求し、物語に多様性を持たせつつ、各物語は独立したものであるべきです。物語それぞれは{END_STRING}という言葉で区切ってください。"
        template = "「{theme}」が話題となっていて、{topic}について書けば良いです。{style}文体で、なるべく{feature}を利用してください。固有名詞を使う場合は、スペースで区切られた一般的な単語で組み立てるが良い。登場人物に名前を付けないか、以下の名前から選んでください。恭子、まりや、そうすけ、あかり、はやと、れいな、佐藤、なつほ、えみり、智也、太郎、智史、夏目、石川、坂口、田中。難しい漢字を使わず、簡単な言葉のみを使用してください。できれば{initial_word_type}で始め、その言葉は「{initial_letter}」と読む文字から始まるようにしてください。"
        if singular:
            template = template_singular + template
        else:
            template = template_plural + template
        
        params = params.copy()
        if params['grammar']:
            params['grammar'] = f"わかりやすくて魅力的な物語を書くことが最優先ですが、できれば自然と{params['grammar']}という文型も織り込んで下さい。"
        if params['persona']:
            params['persona'] = f"{params['persona']}の視点で書いてください。"

    prompt = template.format(**params)
    return prompt, num_stories_per_completion


def generate_content(gen_model, prompt):
    assert "gpt" in gen_model or "claude" in gen_model, "Invalid model name"
    if "gpt" in gen_model:  # OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY_SIMPLESTORIES"])
        completion = client.chat.completions.create(
            model=gen_model,
            top_p=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        completion = completion.choices[0].message.content
    elif "claude" in gen_model:  # Anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY_SIMPLESTORIES"])
        completion = client.messages.create(
            model=gen_model,
            max_tokens=min(1024*MAX_STORIES_PER_COMPLETION, 8192),
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        completion = completion.content[0].text
    
    return completion

def process_completion(gen_model, completion, params, expected_num_stories=None):
    id = hashlib.md5(completion.encode()).hexdigest()
    stories = [x.strip() for x in completion.split(END_STRING) if len(x.strip()) > 1]
    table = str.maketrans({
                "\u201d": "\"",
                "\u201c": "\"",
                "\u2019": "'",
                "\u2018": "'",
                "\u2014": "-",
                "\u2026": "..."
    })
    stories = [x.translate(table) for x in stories]
    if (len(stories) != expected_num_stories and expected_num_stories):
        print(f"Completion did not include expected number of stories, actual={len(stories)} != expected={expected_num_stories}\nend of completion: {completion[-100:]}")
    return [{
        'generation_id': id + "-" + str(k),
        'story': story,
        'model': gen_model,
        'num_stories_in_completion': len(stories),
        "expected_num_stories_in_completion": expected_num_stories,
        **params
    } for k, story in enumerate(stories)]

def generate_simple_story(gen_model, params: dict):
    prompt, expected_num_stories = create_simple_story_prompt(params.copy())
    
    try:
        completion = generate_content(gen_model, prompt)
        return process_completion(gen_model, completion, params, expected_num_stories)
    except Exception as e:
        # TODO Implement Rate Limit Logic for different APIs
        raise RateLimitException(e)

def generate_and_log_simple_stories(gen_model: str, params: dict, formatted_time: str):
    json_struct = generate_simple_story(gen_model, params)
    lines = [json.dumps(item, ensure_ascii=False) for item in json_struct if 'story' in item]
    
    filename = f'data/stories-{gen_model}-{formatted_time}.jsonl'
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def worker_thread(gen_model: str, params: dict, formatted_time: str):
    while True:
        try:
            return generate_and_log_simple_stories(gen_model, params, formatted_time)
        except RateLimitException as e:
            print(f"Rate limit hit: {e}, backing off for 5 seconds...")
            time.sleep(5)
            continue

def main(num_completions: int, num_threads: int = 20, model = "gpt-4o-mini"):
    print(f"Number of Parameter Combinations: {len(themes)*len(topics)*len(styles)*len(features)}")

    if not os.path.exists("data"):
        os.makedirs("data")
    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_story = {
            executor.submit(worker_thread, model, get_random_params(), formatted_time): i for i in range(num_completions)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_story), total=num_completions, desc="Generating stories"):
            try:
                data = future.result()
            except Exception as e:
                print(f"Story generation failed with exception: {e}")


# Reference models: ["gpt-4o", "gpt-4o-mini", "claude-sonnet-3.5-20241022"]
if __name__ == '__main__':
    i = iterate_params()
    print(next(i))
    print(next(i))
    NUM_COMPLETIONS = 2
    #main(NUM_COMPLETIONS, num_threads=40, model="gpt-4o-mini")