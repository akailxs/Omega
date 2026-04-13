import re

with open('build_mind_map.py', 'r') as f:
    content = f.read()

# Add extract_tags
if "def extract_tags" not in content:
    tag_func = """
def extract_tags(content: str):
    tags = re.findall(r'(#[a-zA-Z0-9_À-ÿ\-]+)', content)
    return list(set(tags))
"""
    content = content.replace("def extract_summary", tag_func + "\ndef extract_summary")

# Add tags to nodes
if '"tags":' not in content:
    content = content.replace('"summary": extract_summary(notes[path]),', '"summary": extract_summary(notes[path]),\n        "tags": extract_tags(notes[path]),')

with open('build_mind_map.py', 'w') as f:
    f.write(content)
