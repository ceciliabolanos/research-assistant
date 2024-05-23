import json

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
                    paths.update(get_paths(value["paragraphs"], new_path))
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
        

  