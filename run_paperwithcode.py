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


def process_line(github_url, pdf_url, model_path):
    parts = [part for part in github_url.split('/') if part]
    project_name = os.path.basename(parts[-1])
    temp_dir = project_name  # Path to clone the repo
    
    try:
        # GitHub to embeddings
        clone_github_repo(github_url, temp_dir) 
        if not os.path.exists(os.path.join('./databases', temp_dir)):
            project_structure = get_project_structure(temp_dir)
            output_path_code = os.path.join(temp_dir, f'{project_name}.json')
            os.makedirs(temp_dir, exist_ok=True)
            with open(output_path_code, 'w', encoding='utf-8') as f:
                json.dump(project_structure, f, ensure_ascii=False, indent=4)
            code_searcher = CodeSearcher(model_path, github_repo=temp_dir)
            extract_code_snippets(project_structure, code_searcher)
            code_searcher.save_to_disk()    

        # PDF to embeddings
        if not os.path.exists(os.path.join('./databases', temp_dir.replace('.git', '.paper'))):
            project_structure = convert_pdf_to_json(pdf_url, temp_dir)
            
            # Extract the base name from the PDF URL without the .pdf extension
            pdf_base_name = os.path.basename(pdf_url).replace('.pdf', '')
            file_path = f'{temp_dir}/{pdf_base_name}.json'

            paper_searcher = PaperSearcher(paper_pdf=temp_dir.replace('.git', '.paper'))
            extract_paper_snippets(file_path, paper_searcher)
            paper_searcher.save_to_disk()
    finally:
        # Clean up cloned repository and downloaded paper
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        pdf_file_path = os.path.join('./output', os.path.basename(pdf_url))
        if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)

def main():
    parser = argparse.ArgumentParser(description='Process multiple .txt files containing PDF and GitHub URLs.')
    parser.add_argument('--txt_files', type=str, nargs='+', required=True, help='List of .txt files to process.')
    parser.add_argument('--model_path', type=str, default='unixcoder-ft.bin', help='Path to unixcoder model')
    args = parser.parse_args()

    for txt_file in args.txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                pdf_url, github_url = line.strip().split('\t')
                process_line(github_url, pdf_url, args.model_path)

if __name__ == '__main__':
    main()