import re

def filename_format(text):
    
    text = re.sub(r'\s+', '_', text)
    
    text = re.sub(r'[^a-zA-Z0-9_\-]', '', text)
    
    return text

