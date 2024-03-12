import argparse
import os
import json
from extract_code import get_project_structure
from code_search import CodeSearcher
from utils import *
from tokens import extract_code_snippets
from pdf_to_json import convert_pdf_to_json #Not used for now
from calling_mistral import mistral_process_nl_query

def main():
    # Arguments: github url, pdf, model, mistral? and nl query
    parser = argparse.ArgumentParser(description='Ask questions to the paper and its implementation.')
    parser.add_argument('--github_url', type=str, required=False, help='URL to the GitHub repository.', default="https://github.com/microsoft/autogen/")    
    parser.add_argument('--pdf_path', type=str, required=False, help='Path to the paper PDF', default="./autogen.pdf") 
    parser.add_argument('--model_path', type=str, default='unixcoder-ft.bin', help='Path to unixcoder model')
    parser.add_argument('--mistral', type=str, default='yes', help='Decide if you want to use mistral model to preprocess nl_query')
    parser.add_argument('--nl_query', type=str, required=False, default= 'By allowing custom agents that can converse with each other, conversable agents in AutoGen serve as a useful building block. However, to develop applications where agents make meaningful progresson tasks, developers also need to be able to specify and mold these multi-agent conversations. ### Conversation Specification: Conversations are specified by providing a set of rules for how the conversation should proceed. These rules are expressed using a simple language that allows specifying conditions on the current state of the conversation (e.g., which agent is currently speaking) and actions to take based on those conditions. The following example shows a rule that says if the current speaker is Agent A, then it should speak its next utterance, otherwise it should wait until the next turn:')
    args = parser.parse_args()

    #Path to save pdf and github 
    temp_dir = "./autogen" # Temporary directory to clone the repo


    ########################################## Convert Github and PDF to JSON
    ########### Github
    parts = [part for part in args.github_url.split('/') if part]
    project_name = os.path.basename(parts[-1])
    output_path_code = os.path.join(temp_dir, f'{project_name}.json')

    if not os.path.exists(temp_dir):
        clone_github_repo(args.github_url, temp_dir) 
        project_structure = get_project_structure(temp_dir)

        # Save the project structure to a JSON file
        os.makedirs(temp_dir, exist_ok=True)
        with open(output_path_code, 'w', encoding='utf-8') as f:
            json.dump(project_structure, f, ensure_ascii=False, indent=4)
    
    ########### PDF to JSON
    """We are not using the pdf_to_json file"""
    #project_name = os.path.basename(args.pdf_path)
    #output_path = os.path.join(temp_dir, f'{project_name}.json')
    #convert_pdf_to_json(args.pdf_path, output_dir=output_path)


    ################################# Read and analyze Github and PDF: Chunk y embedding.
    searcher = CodeSearcher(args.model_path, code_snippets=[])

    ######## Github 
    data = {}    
    try:
       with open(output_path_code, 'r', encoding='utf-8') as file:
        data = json.load(file)
    except Exception as e:
        print(f"Error al leer el archivo JSON: {e}")
    
    extract_code_snippets(data,searcher=searcher)

    #Code chunkeados
    new_output_path_code = './autogen/code_chunkeado.json' 
    
    try:
        with open(new_output_path_code, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error al escribir el archivo JSON: {e}")
   

    ######### Search process: Return the more k similar functions.
    
    k=2
    if args.mistral == 'yes':
       t = searcher.get_similarity_search(mistral_process_nl_query(args.nl_query), k)
    else:
       t = searcher.get_similarity_search(args.nl_query, k) 

    for key, value in t.items():
        a, b, c = searcher.get_index_info(value['index'])
        print(f'number:{key}')
        print(f'path: {a}')
        print(f'function name: {b}')
        print(f'code: {c}\n\n')

if __name__ == '__main__':
    main()

