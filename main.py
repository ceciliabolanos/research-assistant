import argparse
import os
import json
from extract_code import get_project_structure
from code_search import CodeSearcher
from utils import *
from tokens import extract_code_snippets
from pdf_to_json import convert_pdf_to_json
from pdfAnalyzer import *

def main():
    # Arguments: github url, pdf, model and nl query
    parser = argparse.ArgumentParser(description='Ask questions to the paper and its implementation.')

    parser.add_argument('--github_url', type=str, required=False, help='URL to the GitHub repository.', default="https://github.com/dr-aheydari/SoftAdapt")    
    parser.add_argument('--pdf_path', type=str, required=False, help='Path to the paper PDF', default="prueba_output\\2107.04734.pdf") #CAMBIAR EL REQUIRE
    parser.add_argument('--model_path', type=str, default='unixcoder-ft.bin', help='Path to unixcoder model')
    parser.add_argument('--nl_query', type=str, required=False, default='need to know the loss function of seq2seq model')
    args = parser.parse_args()

    #Path to save pdf and github 
    temp_dir = "C:\\Users\\chech\\Documents\\TPNLP\\CodeBERT-master\\research-assistant\\prueba_output" # Temporary directory to clone the repo
    
    ########################################## Convert Github and PDF to JSON
    ########### Github
    clone_github_repo(args.github_url, temp_dir) #
    project_structure = get_project_structure(temp_dir)

       # Save the project structure to a JSON file
    os.makedirs(temp_dir, exist_ok=True)
    project_name = os.path.basename(args.github_url)
    output_path_code = os.path.join(temp_dir, f'{project_name}.json')
    with open(output_path_code, 'w', encoding='utf-8') as f:
        json.dump(project_structure, f, ensure_ascii=False, indent=4)
    
    ########### PDF to JSON
    project_name = os.path.basename(args.pdf_path)
    output_path = os.path.join(temp_dir, f'{project_name}.json')
    convert_pdf_to_json(args.pdf_path, output_dir=output_path)

    ################################# Read and analyze Github and PDF: Chunk y embedding.
    
    ######## Github 
    data = {}    
    try:
       with open(output_path_code, 'r', encoding='utf-8') as file:
        data = json.load(file)
    except Exception as e:
        print(f"Error al leer el archivo JSON: {e}")
    
    code_snippets = extract_code_snippets(data)
    

    #Code and PDF chunkeados
    new_output_path_code = '.\\prueba_output\\code_chunkeado.json' 
    
    try:
        with open(new_output_path_code, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error al escribir el archivo JSON: {e}")
   

    ######### Search process
    searcher = CodeSearcher(args.model_path, code_snippets)
    k=3 #agregar para que sea por consola?
    t = searcher.get_similarity_search(args.nl_query, k)

    print("Top K similar items: \n")
    print(t)

if __name__ == '__main__':
    main()

