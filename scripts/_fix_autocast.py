"""Fix autocast deprecation in inference.py"""
path = r'E:\KneeAI\backend\app\segmentation\inference.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace(
    'with autocast(enabled=True):',
    'with autocast("cuda", enabled=True):'
)
content = content.replace(
    'with autocast(enabled=use_amp):',
    'with autocast("cuda", enabled=use_amp):'
)

with open(path, 'w') as f:
    f.write(content)
print('Fixed inference.py autocast calls')
