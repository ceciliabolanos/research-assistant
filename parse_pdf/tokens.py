import json


def split_paragraphs(paragraph):
    sentences = paragraph.split('.')
    chunks = ['.'.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    return chunks

def get_paths(dictionary, current_path=""):
    paths = {}

    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if key == "Body Content":
                new_path = current_path
                paths.update(get_paths(value, new_path))
            elif key == "Figures":
                for figure in value:
                    if "label" in figure and "description" in figure:
                        label = figure["label"] if figure["label"] else "unknown"
                        new_path = f"{current_path}Figure {label}"
                        paths[new_path] = figure["description"]
            else:
                new_path = f"{current_path}/{key}" if current_path else key
                paths.update(get_paths(value, new_path))
    elif isinstance(dictionary, list):
        for value in dictionary:
            if isinstance(value, dict) and "title" in value:
                new_path = f"{current_path}/{value['title']}" if current_path else value['title']
                if "paragraphs" in value:
                    paragraphs = value["paragraphs"]
                    for paragraph in paragraphs:
                        chunks = split_paragraphs(paragraph)
                        for idx, chunk in enumerate(chunks):
                            chunk_path = f"{new_path}/paragraph_{idx+1}"
                            paths[chunk_path] = chunk
            else:
                new_path = current_path
                paths.update(get_paths(value, new_path))
    else:
        paths[current_path] = dictionary

    return paths

def extract_paper_snippets(json_path, paper_searcher, max_length=256):
    with open(json_path, 'r') as file:
        data = json.load(file)
    paths = get_paths(data)
    print(paths)
    for key, value in paths.items():
        print(value, key)
        if value is None:
            value = 'couldnt parse'
        embeddings = paper_searcher.generate_embeddings([value], [{'path': key}])
        

  