# Marked for deletion: This only contains code that is not needed for a well-prompted run.
import json
import os

def separate_stories(stories, end_string):
    out = []
    for entry in stories:
        story = entry["story"]
        out.extend([entry | {"story": x} for x in story.split(end_string) if len(x) > 5])
        if end_string in story:
            print(len(out))
        print(entry)
    
    return out

filename = r"D:\simple_stories_generate\data\stories-claude-3-5-sonnet-20240620-2024-08-25-18-06-54.jsonl"
new_filename = f"{filename.split('.')[0]}_processed.{filename.split('.')[1]}"

try:
    os.remove(new_filename)
except FileNotFoundError:
    pass

print(filename)
with open(filename) as fp:
    lines = fp.readlines()
    for line in lines:
        data = json.loads(line)
        out = separate_stories(data, "THE END")
        
        for new_line in out:
            with open(new_filename, "a") as f:
                f.write(json.dumps(new_line) + "\n")