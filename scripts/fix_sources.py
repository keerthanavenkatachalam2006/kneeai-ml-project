"""Fix source strings in implant_database.py"""
path = r'E:\KneeAI\backend\app\implant_database.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    'source="Stryker Triathlon Surgical Protocol"',
    'source="Stryker Triathlon Surgical Protocol (2018), approximate values"'
)
content = content.replace(
    'source="Zimmer NexGen Complete Knee Solution"',
    'source="Zimmer NexGen Surgical Technique (2016), approximate values"'
)
content = content.replace(
    'source="Smith & Nephew LEGION Surgical Technique"',
    'source="Smith & Nephew LEGION Surgical Technique (2017), approximate values"'
)

with open(path, 'w') as f:
    f.write(content)

print('Updated source strings.')
