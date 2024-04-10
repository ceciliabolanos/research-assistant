import argparse
import os
import json
from extract_code import get_project_structure
from utils import *
from tokens import extract_code_snippets
#from calling_mistral import mistral_process_nl_query
from langchain_faiss import CodeSearcher

from pdf_to_json import convert_pdf_to_json #Not used for now


def main():
    # Arguments: github url, pdf, model, mistral? and nl query
    parser = argparse.ArgumentParser(description='Ask questions to the paper and its implementation.')
    parser.add_argument('--github_url', type=str, required=False, help='URL to the GitHub repository.', default="https://github.com/ankitapasad/layerwise-analysis")    
    parser.add_argument('--pdf_path', type=str, required=False, help='Path to the paper PDF', default="./lawer-analysis.pdf") 
    parser.add_argument('--model_path', type=str, default='unixcoder-ft.bin', help='Path to unixcoder model')
    parser.add_argument('--mistral', type=str, default='no', help='Decide if you want to use mistral model to preprocess nl_query')
    parser.add_argument('--nl_query', type=str, required=False, default= 'Give me the function to get similarity between two layers')
    args = parser.parse_args()

    #Path to save pdf and github 
    temp_dir = "./layer-analysis" # Temporary directory to clone the repo
  

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
    """We are not using the pdf_to_json script"""
    #project_name = os.path.basename(args.pdf_path)
    #output_path = os.path.join(temp_dir, f'{project_name}.json')
    #convert_pdf_to_json(args.pdf_path, output_dir=output_path)

    
    new_output_path_code = os.path.join(f'/{project_name}', 'code_chunkeado.json')
    ################################# Read and analyze Github and PDF: Chunk y embedding.
   
    searcher = CodeSearcher(args.model_path, github_url= args.github_url, code_snippets=[])

    
    ######## Github 
    project_structure = {}    
    try:
       with open(output_path_code, 'r', encoding='utf-8') as file:
            project_structure = json.load(file)
    except Exception as e:
        print(f"Error al leer el archivo JSON: {e}")
      
    extract_code_snippets(project_structure, searcher, save = True)
    #Code chunkeados
    
    
    try:
        with open(new_output_path_code, 'w') as file:
            json.dump(project_structure, file, indent=4)
    except Exception as e:
        print(f"Error al escribir el archivo JSON: {e}")
   

    ######### Search process: Return the more k similar functions.
    
    k=2
    if args.mistral == 'yes':
       t = searcher.similarity_search(mistral_process_nl_query(args.nl_query), k)
    else:
       t = searcher.similarity_search(args.nl_query, k, use_nl_inputs=True) 


  
    for result_dict in t:
        for key in result_dict:
            print(key)
            top_result = result_dict[key]
            print(f'index: {top_result["index"]}')
            print(f'path: {top_result["path"]}')
            print(f'function name: {top_result["function_name"]}\n')


if __name__ == '__main__':
    main()

