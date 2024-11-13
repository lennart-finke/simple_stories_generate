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

MAX_STORIES_PER_COMPLETION = 40
END_STRING = "THE END."

class RateLimitException(Exception):
    pass


themes = {"en": ["Friendship", "Courage", "Contradiction", "Coming of age", "Kindness", "Amnesia", "Adventure", "Imagination", "Family", "Perseverance", "Curiosity", "Honesty", "Romance", "Teamwork", "Responsibility", "Strategy", "Magic", "Discovery", "Betrayal", "Deception", "Generosity", "Creativity", "Self-Acceptance", "Helping Others", "Hardship", "Agency", "Power", "Revenge", "Independence", "Problem-Solving", "Resourcefulness", "Long-Term Thinking", "Optimism", "Humor", "Love", "The Five Senses", "Tradition", "Innovation", "Hope", "Dreams", "Belonging", "Travel", "Overcoming", "Trust", "Morality", "Happiness", "Consciousness", "Failure", "Conflict", "Cooperation", "Growth", "Loss", "Celebration", "Transformation", "Scheming", "Challenge", "Planning", "Wonder", "Surprises", "Conscience", "Intelligence", "Logic", "Resilience"]}["en"]
topics = {"en": ["talking animals", "fantasy worlds", "time travel", "a deadline or time limit", "space exploration", "mystical creatures", "underwater adventures", "dinosaurs", "pirates", "superheroes", "fairy tales", "outer space", "hidden treasures", "magical lands", "enchanted forests", "secret societies", "robots and technology", "sports", "school life", "holidays", "cultural traditions", "magical objects", "lost civilizations", "subterranean worlds", "bygone eras", "invisibility", "giant creatures", "miniature worlds", "alien encounters", "haunted places", "shape-shifting", "island adventures", "unusual vehicles", "undercover missions", "dream worlds", "virtual worlds", "riddles", "sibling rivalry", "treasure hunts", "snowy adventures", "seasonal changes", "mysterious maps", "royal kingdoms", "living objects", "gardens", "lost cities", "the arts", "the sky"]}["en"]
styles = {"en": ["whimsical", "playful", "epic", "fairy tale-like", "modern", "classic", "lyric", "mythological", "lighthearted", "adventurous", "heartwarming", "humorous", "mystical", "action-packed", "fable-like", "surreal", "philosophical", "melancholic", "noir", "romantic", "tragic", "minimalist", "suspenseful"]}["en"]
features = {"en": ["dialogue", "in medias res", "a moral lesson", "absence indicating a presence", "a story told through letters", "a twist ending", "an unrealiable narrater", "foreshadowing", "irony", "inner monologue", "symbolism", "a MacGuffin", "a non-linear timeline", "a reverse timeline", "circular narrative structure", "a flashback", "a nested structure", "a story within a story", "a Red Herring", "multiple perspectives", "Checkhov's gun", "the fourth wall", "a cliffhanger", "an anti-hero", "juxtaposition", "climactic structure"]}["en"]
grammars = {"en": ["present tense", "past tense", "future tense", "progressive aspect", "perfect aspect", "passive voice", "conditional mood", "imperative mood", "indicative mood", "relative clauses", "prepositional phrases", "indirect speech", "exclamative sentences", "comparative forms", "superlative forms", "subordinate clauses", "ellipsis", "anaphora", "cataphora", "wh-questions", "yes-no questions", "gerunds", "participle phrases", "inverted sentences", "non-finite clauses", "determiners", "quantifiers", "adjective order", "parallel structure", "discourse markers", "appositive phrases"]}["en"]
personas = {"en": ["an explorer archetype", "a rebellious author", "a powerful leader", "a wise, old person who wants to teach the young", "an innocent author", "a moralistic teacher", "a hopeless romantic", "a hurt, ill-intentioned person", "an academic", "a jester archetype", "a poet", "a philosopher", "a mother", "a father", "someone curious", "someone evil", "someone who wants to prove a point", "a child", "a pedant", "the everyman", "the oppressed", "a cruel person", "someone who loves order and structure"]}["en"]
word_types = {"en": ["a noun", "an adjective", "an adverb", "a preposition"]}["en"]

# Below is copied from https://en.wikipedia.org/wiki/Letter_frequency and cross-checked with http://norvig.com/mayzner.html. 
letter_frequencies = {"en": {
        'A': 11.7, 'B': 4.4, 'C': 5.2, 'D': 3.2, 'E': 2.8,
        'F': 4.0, 'G': 1.6, 'H': 4.2, 'I': 7.3, 'J': 0.51,
        'K': 0.86, 'L': 2.4, 'M': 3.8, 'N': 2.3, 'O': 7.6,
        'P': 4.3, 'Q': 0.22, 'R': 2.8, 'S': 6.7, 'T': 16.0,
        'U': 1.2, 'V': 0.82, 'W': 5.5, 'X': 0.045, 'Y': 0.76,
        'Z': 0.045
}}["en"]

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
    combinations = list(itertools.product(themes, topics, styles, features, word_types))
    print(f"Using {len(combinations)} combinations...")
    random.shuffle(combinations)
    
    print(len(grammars), len(personas))
    assert(len(grammars) % 2 != 0), "Number of grammars should be odd"
    assert(len(personas) % 3 != 0), "Number of personas should not be divisible by 3"

    for k, combination in enumerate(combinations):
        theme, topic, style, feature, word_type = combination
        random_letter = letter_pool[k % len(letter_pool)]  # Cycle through shuffled letter pool
        
        # Grammar and persona are set to empty strings half, and a third of the time respectively
        grammar = grammars[k % len(grammars)] if k % 2 == 0 else ""
        persona = personas[k % len(personas)] if k % 3 == 0 else ""
        
        # Yield each combination with the desired parameters
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

def create_simple_story_prompt(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])

    singular = params['num_paragraphs'] == 1
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

    prompt = template.format(**params)
    prompt.replace("\n", "")
    return prompt, num_stories_per_completion

def generate_content(gen_model, prompt):
    assert "gpt" in gen_model or "claude" in gen_model, "Invalid model name"
    if "gpt" in gen_model:  # OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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


# Reference models: ["gpt-4o", "gpt-4o-mini", "claude-sonnet-3.5-20241022"]
if __name__ == '__main__':
    i = iterate_params()
    print(next(i))
    print(next(i))
    #NUM_COMPLETIONS = 200
    #main(NUM_COMPLETIONS, num_threads=40, model="gpt-4o-mini")