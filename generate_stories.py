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

LANGUAGE = "ja"

MAX_STORIES_PER_COMPLETION = 40
END_STRING = {"en": "THE END.", "ja": "終わり。"}["ja"]

class RateLimitException(Exception):
    pass


themes = {"ja": ["自尊心", "立身出世", "わびさび", "悲しみ", "怒り", "ワクワク", "おしゃれ", "企画", "画策", "怖さ", "友情", "勇気", "親切", "幸せ", "想像力", "家族", "嘘", "好奇心", "恋愛", "チームワーク", "責任", "戦略たて", "魔法", "発見", "裏切り", "欺瞞", "寛大さ", "創造性", "困難", "自立", "権力", "復讐", "独立", "問題解決", "長期的思考", "楽観主義", "ユーモア", "愛", "五感", "伝統", "革新", "希望", "夢", "帰属意識", "旅行", "克服", "信頼", "道徳", "意識", "失敗", "対立", "協力", "成長", "喪失", "祝福", "変容", "計略", "挑戦", "計画", "驚き", "良心", "賢さ", "推理"], "en": ["Friendship", "Courage", "Coming of age", "Kindness", "Adventure", "Imagination", "Family", "Perseverance", "Curiosity", "Honesty", "Romance", "Teamwork", "Responsibility", "Strategy", "Magic", "Discovery", "Betrayal", "Deception", "Generosity", "Creativity", "Self-Acceptance", "Helping Others", "Hardship", "Agency", "Power", "Revenge", "Independence", "Problem-Solving", "Resourcefulness", "Long-Term Thinking", "Optimism", "Humor", "Love", "The Five Senses", "Tradition", "Innovation", "Hope", "Dreams", "Belonging", "Travel", "Overcoming", "Trust", "Morality", "Happiness", "Consciousness", "Failure", "Conflict", "Cooperation", "Growth", "Loss", "Celebration", "Transformation", "Scheming", "Challenge", "Planning", "Wonder", "Surprises", "Conscience", "Intelligence", "Logic", "Resilience"]}[LANGUAGE]
topics = {"ja": ["平安時代での冒険", "温泉旅行", "江戸の生活", "明治維新", "忍びと侍", "夏の祭り", "上京", "怪談話", "日常生活", "お化け話", "縄文時代", "バブル時代", "話す動物たち", "異世界", "時間旅行", "宇宙探検", "神秘的な生き物たち", "水中の冒険", "恐竜の時代", "海賊の物語", "ヒーローの伝説", "昔話", "未知の宇宙", "隠された財宝", "魔法の国", "不思議な森", "秘密の組織", "ロボットと未来技術", "体育と競技", "学校生活のドラマ", "休日の冒険", "魔法の道具", "滅んでいた文明", "地下の世界", "過去の時代への旅", "巨大な生物", "小さな世界", "宇宙人との出会い", "孤島の冒険", "奇妙な乗り物", "夢の世界", "仮想空間", "探偵と謎解き", "兄弟の競争", "宝探し", "雪国への旅", "季節の移り変わり", "謎の地図", "王国の物語", "庭園の秘密", "芸術", "大空"], "en": ['talking animals',  'fantasy worlds',  'time travel',  'space exploration',  'mystical creatures',  'underwater adventures',  'dinosaurs',  'pirates',  'superheroes',  'fairy tales',  'outer space',  'hidden treasures',  'magical lands',  'enchanted forests',  'secret societies',  'robots and technology',  'sports',  'school life',  'holidays',  'cultural traditions',  'magical objects',  'lost civilizations',  'subterranean worlds',  'bygone eras',  'invisibility',  'giant creatures',  'miniature worlds',  'alien encounters',  'haunted places',  'shape-shifting',  'island adventures',  'unusual vehicles',  'undercover missions',  'dream worlds',  'virtual worlds',  'riddles',  'sibling rivalry',  'treasure hunts',  'snowy adventures',  'seasonal changes',  'mysterious maps',  'royal kingdoms',  'living objects',  'gardens',  'lost cities',  'the arts',  'the sky']}[LANGUAGE]
styles = {"ja": ['気まぐれな', '遊び心のある', '壮大な', 'おとぎ話風のような', '現代的な', '古典的な', '叙情的な', '神話的な', '軽快な', '冒険的な', '心温まる', '神秘的な', 'アクション満載の', '寓話風の', '超現実的な', '哲学的な', '哀愁漂う', 'ロマンチックな', '悲劇的な', 'ミニマリストな', 'サスペンスフルな'], "en": ['whimsical', 'playful', 'epic', 'fairy tale-like', 'modern', 'classic', 'lyric', 'mythological', 'lighthearted', 'adventurous', 'heartwarming', 'humorous', 'mystical', 'action-packed', 'fable-like', 'surreal', 'philosophical', 'melancholic', 'noir', 'romantic', 'tragic', 'minimalist', 'suspenseful']}[LANGUAGE]
features = {"ja": ["対話", "道徳的教訓", "どんでん返しの結末", "伏線", "皮肉", "内面の独白", "象徴主義", "マクガフィン", "非線形のタイムライン", "フラッシュバック", "入れ子構造", "物語の中の物語", "複数の視点", "チェーホフの銃", "第四の壁", "クリフハンガー", "アンチヒーロー", "対比", "クライマックス構造"], "en": ["dialogue", "a moral lesson", "a twist ending", "foreshadowing", "irony", "inner monologue", "symbolism", "a MacGuffin", "a non-linear timeline", "a flashback", "a nested structure", "a story within a story", "multiple perspectives", "Checkhov's gun", "the fourth wall", "a cliffhanger", "an anti-hero", "juxtaposition", "climactic structure"]}[LANGUAGE]
grammars = {"ja": ["現在形", "過去形", "進行形", "完了形", "受身形", "使役形", "可能形", "意向形", "仮定形", "命令形", "テ形", "タ形", "ナイ形", "条件形", "尊敬語", "謙譲語", "丁寧語", "授受表現", "は・がの使い分け", "を・にの使い分け", "助詞", "副詞", "形容詞の活用", "形容動詞", "名詞修飾", "接続詞", "補助動詞", "数量詞の使い方", "疑問詞", "間接疑問", "重文", "複文", "話し言葉の省略"], "en": ["present tense","past tense","future tense","progressive aspect","perfect aspect","passive voice","conditional mood","imperative mood","indicative mood","relative clauses","prepositional phrases","indirect speech","exclamative sentences","comparative forms","superlative forms","subordinate clauses","ellipsis","anaphora","cataphora","wh-questions","yes-no questions","gerunds","participle phrases","inverted sentences","non-finite clauses","determiners","quantifiers","adjective order","parallel structure","discourse markers","appositive phrases"]}[LANGUAGE]

def get_random_params():
    grammar = random.choice(grammars)
    if random.random() < 0.5:
        grammar = ""
    return {
        "theme": random.choice(themes),
        "topic": random.choice(topics),
        "style": random.choice(styles),
        "feature": random.choice(features),
        "grammar": grammar,
        "num_paragraphs": random.randint(1, 9),
    }

def iterate_params(seed=42):
    # A slightly hacky way to yield all combinations of parameters, but having empty "grammar" value half of the time.
    # Assumes the lengths of (themes * topics * styles * features), grammars, the range of num_paragraphs and 2 are coprime.
    random.seed(seed)

    # This stores all combinations in memory at the moment, inelegant but not a big problem at the moment. Can be easily refactored if all parameter list lengths are coprime.
    combinations = list(itertools.product(themes, topics, styles, features))
    random.shuffle(combinations)
    for k, combination in enumerate(combinations):
        theme, topic, style, feature = combination
        grammar = grammars[k % len(grammars)]
        if k % 2 == 0:
            grammar = ""
        yield {
            "theme": theme,
            "topic": topic,
            "style": style,
            "feature": feature,
            "grammar": grammar,
            "num_paragraphs": 1+(k%9),
        }
            
def create_simple_story_prompt_en(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])

    singular = params['num_paragraphs'] == 1
    template_singular = f"Write a short story ({params['num_paragraphs']} paragraphs) using very basic words that a preschool child could understand. \nThe story "
    template_plural = f"Write {num_stories_per_completion} short stories ({params['num_paragraphs']} paragraph{'' if singular else 's'} each) using very basic words that a young child could understand. Do not number each story or write a headline. Make the stories diverse by fully exploring the theme, but each story should be self-contained. Separate the stories by putting {END_STRING} in between.\nEach story "
    template = "should be about {theme}, include {topic}, be {style} in its writing style and ideally feature {feature}.{grammar} If you need to use proper names, make them from space-separated common words. Either don't give characters a name, or select from Mia, Alex, Jean, Samuel, Lily, Leo, Jose, Kim, Alice, Lena, Rita, Emmanuel, Anne, Peter, Maria or Luis. Complex story structure is great, but please remember to only use simple words."
    if singular:
        template = template_singular + template
    else:
        template = template_plural + template
    
    params = params.copy()
    if params['grammar']:
        params['grammar'] = f" The most important thing is to write an engaging easy story, but where it makes sense, demonstrate the use of {params['grammar']}."

    prompt = template.format(**params)
    return prompt, num_stories_per_completion

def create_simple_story_prompt_ja(params):
    num_stories_per_completion = MAX_STORIES_PER_COMPLETION // max(3, params['num_paragraphs'])

    num_chars = params["num_paragraphs"] * 100
    singular = num_stories_per_completion == 1
    template_singular = f"非常に簡単な単語を使用して、子供でも分かるような約{num_chars}文字の物語を書いてください。"
    template_plural = f"非常に簡単な単語を使用して、子供でも理解できるような約{num_chars}文字の物語を{num_stories_per_completion}個書いてください。各物語に番号やタイトルをつけないでください。テーマを十分に探求して物語に多様性を持たせつつ、各物語は独立したものであるべきです。物語それぞれは{END_STRING}という言葉で区切ってください。"
    template = "「{theme}」が話題となっていて、{topic}について書けば良いです。{style}文体で、なるべく{feature}を利用してください.固有名詞を使う場合は、スペースで区切られた一般的な単語で組み立てるが良い。キャラクターに名前を付けないか以下の名前から選んでください　ー　恭子、まりや、そうすけ、あかり、はやと、れいな、佐藤、なつほ、えみり、智也、太郎、智史、夏目、石川、坂口、田中。複雑なストリー構造を使っても構いませんが、必ずしも簡単な言葉のみを使用してください。"
    if singular:
        template = template_singular + template
    else:
        template = template_plural + template
    
    params = params.copy()
    if params['grammar']:
        params['grammar'] = f"わかりやすくて魅力的な物語を書くことが最優先ですが、出来れば自然と{params['grammar']}という文型も織り込んで下さい。"

    prompt = template.format(**params)
    return prompt, num_stories_per_completion

create_simple_story_prompt = {x.split("_")[-1]:locals()[x] for x in locals() if x[:26] == "create_simple_story_prompt"}[LANGUAGE]

def generate_content(gen_model, prompt):
    assert "gpt" in gen_model or "claude" in gen_model, "Invalid model name"
    if "gpt" in gen_model:  # OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY_SIMPLESTORIES"])
        completion = client.chat.completions.create(
            model=gen_model,
            top_p=0.95,
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


# Reference models: ["gpt-4o", "gpt-4o-mini", "claude-sonnet-3.5-20240620"]
if __name__ == '__main__':
    NUM_COMPLETIONS = 3
    main(NUM_COMPLETIONS, num_threads=50, model="gpt-4o-mini")