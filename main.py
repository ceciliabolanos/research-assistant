import argparse
import os
import json
from parse_pdf.pdf_to_json import convert_pdf_to_json
from parse_pdf.XMLParser import XMLParser
from parse_code.extract_code import get_project_structure
from parse_code.utils import *
from parse_code.tokens import extract_code_snippets
from parse_pdf.tokens import extract_paper_snippets
from retrieval.code_searcher import CodeSearcher
from retrieval.paper_searcher import PaperSearcher
from retrieval.conversation import Conversation
import getpass
from utils import read_xml_file 

def main():
    # Arguments: github url, pdf, model, mistral?
    parser = argparse.ArgumentParser(description='Ask questions to the paper and its implementation.')
    parser.add_argument('--github_url', type=str, required=False, help='URL to the GitHub repository.', default="https://github.com/ankitapasad/layerwise-analysis.git") 
    parser.add_argument('--pdf', type=str, required=False, help='Paper corresponding to the pdf', default='./pdf/2107.04734.pdf')
    parser.add_argument('--model_path', type=str, default='unixcoder-ft.bin', help='Path to unixcoder model')
    parser.add_argument('--mistral', type=str, default='no', help='Decide if you want to use mistral model to preprocess nl_query')
    parser.add_argument('--chat_model', type=str, default='gpt-3.5-turbo-0125')
    args = parser.parse_args()
    
    parts = [part for part in args.github_url.split('/') if part]
    project_name = os.path.basename(parts[-1])
    temp_dir = project_name # Path to clone the repo
  

    ####################### Github to embeddings
    clone_github_repo(args.github_url, temp_dir) 
    if not os.path.exists(os.path.join('./databases',temp_dir)):
        project_structure = get_project_structure(temp_dir)
        output_path_code = os.path.join(temp_dir, f'{project_name}.json')
        # Save the project structure to a JSON file
        os.makedirs(temp_dir, exist_ok=True)
        with open(output_path_code, 'w', encoding='utf-8') as f:
            json.dump(project_structure, f, ensure_ascii=False, indent=4)
        code_searcher = CodeSearcher(args.model_path, github_repo= temp_dir)
        extract_code_snippets(project_structure, code_searcher)
        searcher.save_to_disk()    
       
    ####################### Pdf to embeddings

    if not os.path.exists(os.path.join('./databases',temp_dir.replace('.git', '.paper'))):
        project_structure = convert_pdf_to_json(args.pdf, temp_dir)
        file_path = f'{temp_dir}/2107.04734.json'
        paper_searcher = PaperSearcher(paper_pdf= temp_dir.replace('.git', '.paper'))
        extract_paper_snippets(file_path, paper_searcher)
        paper_searcher.save_to_disk()   

    ######### Search process: Return the more k similar functions.

    with open('retrieval/tools.json', 'r', encoding='utf-8') as file:
        tools = json.load(file)

    system_message = """You are a helpful assistant. Use the available tools to help the user understand the codebase."""

    OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key:")
    path = os.path.join('./databases', temp_dir)
    searcher = CodeSearcher(args.model_path, github_repo= temp_dir, faiss_path=path)
    paper_conversation = Conversation(OPENAI_API_KEY, searcher=searcher, tools=tools, mistral_option=args.mistral, chat_model=args.chat_model)
    paper_conversation.add_message("system", system_message)
    paper_conversation.chat()
    
if __name__ == '__main__':
    main()

