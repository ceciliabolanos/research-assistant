import argparse
import os
import json
from extract_code import get_project_structure, analyze_dependencies, process_atlas_data
from code_search import CodeSearcher


def main():
    parser = argparse.ArgumentParser(description='Parse a project structure into a JSON representation.')
    parser.add_argument('--project_path', type=str, required=True, help='Path to the root of the project.')
    parser.add_argument('--output_dir_path', type=str, required=True, help='Path to the output dir.')
    parser.add_argument('--model_path', type=str, default='C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\RA\\model (7).bin', help='Path to unixcoder model') # change to your model path
    parser.add_argument('--nl_query', type=str, default='need to know the loss function of seq2seq model')
    args = parser.parse_args()

    # Get the project structure
    project_structure = get_project_structure(args.project_path)

    # Save the project structure to a JSON file
    os.makedirs(args.output_dir_path, exist_ok=True)
    project_name = os.path.basename(args.project_path)
    output_path = os.path.join(args.output_dir_path, f'{project_name}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project_structure, f, ensure_ascii=False, indent=4)
    
    data = {}
    # Leer y analizar el archivo JSON
    try:
       with open(output_path, 'r') as file:
        data = json.load(file)
    except Exception as e:
        print(f"Error al leer el archivo JSON: {e}")

    # Proceso de extracción de fragmentos de código
    code_snippets = []
    try:
    # Iterar sobre todas las claves principales del diccionario JSON
        for main_key in data.keys():
            for filename, content in data[main_key].items():
                if 'functions' in content:
                    for function_name, function_content in content['functions'].items():
                        if 'code' in function_content:
                            code_snippets.append(function_content['code'])
    except Exception as e:
        code_snippets = f"Error al procesar el JSON: {e}"

    searcher = CodeSearcher(args.model_path, code_snippets)
    k=3
    t = searcher.get_similarity_search(args.nl_query, k)

    print("Top K similar items: \n")
    print(t)

if __name__ == '__main__':
    main()

