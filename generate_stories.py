import random
import json
import os
from openai import OpenAI
import hashlib
import time
import anthropic
import concurrent.futures
from tqdm import tqdm
from datetime import datetime

MAX_STORIES_PER_COMPLETION = 40
END_STRING = "THE END."

class RateLimitException(Exception):
    pass

themes = {"en": ["Friendship","Courage","Coming of age", "Kindness","Adventure","Imagination","Family","Perseverance","Curiosity","Honesty","Romance","Teamwork","Responsibility","Strategy","Magic","Discovery","Betrayal","Deception","Generosity","Creativity","Self-Acceptance","Helping Others","Hardship","Agency","Power","Revenge","Independence","Problem-Solving","Resourcefulness","Long-Term Thinking","Optimism","Humor","Love","The Five Senses","Tradition","Innovation","Hope","Dreams","Belonging","Travel","Overcoming","Trust","Morality","Happiness","Consciousness","Failure","Conflict","Cooperation","Growth","Loss","Celebration","Transformation","Scheming","Challenge","Planning","Wonder","Surprises","Conscience","Intelligence","Logic","Resilience"]}["en"]
topics = {"en": ["Talking animals", "Fantasy worlds", "Time travel", "Space exploration", "Mystical creatures", "Underwater adventures", "Dinosaurs", "Pirates", "Superheroes", "Fairy tales", "Outer space", "Hidden treasures", "Magical lands", "Enchanted forests", "Secret societies", "Robots and technology", "Sports", "School life", "Holidays", "Cultural traditions", "Magical objects", "Lost civilizations", "Subterranean Worlds", "Bygone Eras", "Invisibility", "Giant creatures", "Miniature worlds", "Alien encounters", "Haunted places", "Shape-shifting", "Island adventures", "Unusual vehicles", "Undercover missions", "Dream worlds", "Virtual worlds", "Riddles", "Sibling rivalry", "Treasure hunts", "Snowy adventures", "Seasonal changes", "Mysterious maps", "Royal kingdoms", "Living objects", "Gardens", "Lost cities", "The arts", "The sky"]}["en"]
styles = {"en": ["Whimsical","Playful","Epic","Fairy tale-like","Folk tale-like","Modern","Classic","Lyric","Mythological","Lighthearted","Adventurous","Heartwarming","Humorous","Mystical","Action-packed","Fable-like","Surreal","Philosophical","Melancholic","Noir","Romantic","Tragic","Minimalist","Suspenseful"]}["en"]
features = {"en": ["dialogue", "a moral lesson", "a twist ending", "foreshadowing", "irony", "inner monologue", "symbolism", "a MacGuffin", "a non-linear timeline", "a flashback", "a nested structure", "a story within a story", "multiple perspectives", "Checkhov's gun", "the fourth wall", "a cliffhanger", "an anti-hero", "juxtaposition", "climactic structure"]}["en"]
grammars = {"en": ["present tense","past tense","future tense","progressive aspect","perfect aspect","passive voice","conditional mood","imperative mood","indicative mood","relative clauses","prepositional phrases","indirect speech","exclamative sentences","comparative forms","superlative forms","subordinate clauses","ellipsis","anaphora","cataphora","wh-questions","yes-no questions","gerunds","participle phrases","inverted sentences","non-finite clauses","determiners","quantifiers","adjective order","parallel structure","discourse markers","appositive phrases","conjunctions",]}["en"]

def get_random_params():
    grammar = random.choice(grammars)
    if random.random() < 0.5:
        grammar = ""
    return {
        "theme": random.choice(themes),
        "topic": random.choice(topics).lower(),
        "style": random.choice(styles).lower(),
        "feature": random.choice(features),
        "grammar": grammar,
        "num_paragraphs": random.randint(1, 10),
    }

def create_simple_story_prompt(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])

    singular = params['num_paragraphs'] == 1
    template_singular = f"Write a short story ({params['num_paragraphs']} paragraphs) using very basic words that a young child could understand. \nThe story "
    template_plural = f"Write {num_stories_per_completion} short stories ({params['num_paragraphs']} paragraph{'' if singular else 's'} each) using very basic words that a young child could understand. Do not number each story or write a headline. Make the stories diverse by fully exploring the theme, but each story should be self-contained. Separate the stories by putting {END_STRING} in between.\nEach story "
    template = "should be about {theme}, include {topic}, be {style} in its writing style and ideally feature {feature}.{grammar} If you need to use proper names, make them from common words. Either don't give characters a name, or select from Mia, Alex, Jean, Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, Emmanuel, Anne, Peter, Maria, Luis and derivations of these. Complex story structure is great, but please always use the simplest words possible."
    if singular:
        template = template_singular + template
    else:
        template = template_plural + template
    
    if params['grammar']:
        params['grammar'] = f" The most important thing is to write an engaging easy story, but where it makes sense, demonstrate the use of {params['grammar']}."

    prompt = template.format(**params)
    return prompt, num_stories_per_completion

def generate_content(gen_model, prompt):
    assert "gpt" in gen_model or "claude" in gen_model, "Invalid model name"
    if "gpt" in gen_model:  # OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY_SIMPLESTORIES"])
        completion = client.chat.completions.create(
            model=gen_model,
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
            top_p=0.9,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        completion = completion.content[0].text
    
    return completion

def generate_simple_story(gen_model, params: dict):
    prompt, expected_num_stories = create_simple_story_prompt(params.copy())
    id = hashlib.md5(prompt.encode()).hexdigest()
    
    try:
        completion = generate_content(gen_model, prompt)
        stories = [x.strip() for x in completion.split(END_STRING) if len(x.strip()) > 1]
        if (len(stories) != expected_num_stories):
            print(f"Completion did not include expected number of stories, actual={len(stories)} != expected={expected_num_stories}\nend of completion: {completion[-100:]}")
        return [{
            'generation_id': id + "-" + str(k),
            'story': story,
            'model': gen_model,
            'num_stories_in_completion': len(stories),
            "expected_num_stories_in_completion": expected_num_stories,
            **params
        } for k, story in enumerate(stories)]
    except Exception as e:
        # TODO Implement Rate Limit Logic for different APIs
        raise RateLimitException(e)

def generate_and_log_simple_stories(gen_model: str, params: dict, formatted_time: str):
    json_struct = generate_simple_story(gen_model, params)
    lines = [json.dumps(item) for item in json_struct if 'story' in item]
    
    filename = f'data/stories-{gen_model}-{formatted_time}.jsonl'
    with open(filename, "a") as f:
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

# Reference models: ["gpt-4o", "gpt-4o-mini", "claude-sonnet-3.5-20240620"]
if __name__ == '__main__':
    NUM_COMPLETIONS = 200

    main(NUM_COMPLETIONS, num_threads=50, model="claude-3-5-sonnet-20240620")