import os
import json
import subprocess
from parse_pdf.XMLParser import XMLParser


def convert_pdf_to_json(pdf_file_path, output_dir='./output'):
    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Assuming GROBID is installed and set up correctly in the environment
    # Execute GROBID to process the PDF file
    subprocess.run([
    'java', '-Djava.awt.headless=true', '-Xmx4G',  # Set Java to headless
    '-jar', 'grobid-0.7.2/grobid-core/build/libs/grobid-core-0.7.2-onejar.jar',
    '-gH', 'grobid-0.7.2/grobid-home',
    '-dIn', os.path.dirname(pdf_file_path),  # Ensure this is the path to the PDFs
    '-dOut', output_dir,
    '-exe', 'processFullText'
], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Construct the expected XML file path from the PDF file name
    xml_file_name = os.path.basename(pdf_file_path).replace('.pdf', '.tei.xml')
    xml_file_path = os.path.join(output_dir, xml_file_name)

    # Read the XML file
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_data = file.read()
    except Exception as e:
        return f"Error reading XML file: {e}"

    parser = XMLParser(xml_data)

    # Extract data
    data = {
        "Title": parser.get_title(),
        "Abstract": parser.get_abstract(),
        "Body Content": parser.get_body_content(),
        "References": parser.get_references(),
        "Figures": parser.get_figures(),
        "Figures_references": parser.get_figure_references()
    }

    filename = os.path.join(output_dir, os.path.basename(pdf_file_path).replace('.pdf', '.json'))
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)