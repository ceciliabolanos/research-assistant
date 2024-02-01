
import os
import xmltodict
import json
import subprocess

def convert_pdf_to_json(pdf_file_path, output_dir='./output'):
    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Assuming GROBID is installed and set up correctly in the environment
    # Execute GROBID to process the PDF file
   
    subprocess.run([
        'java', '-Xmx4G', '-jar', 'grobid-0.7.2/grobid-core/build/libs/grobid-core-0.7.2-onejar.jar',
        '-gH', 'grobid-0.7.2/grobid-home', '-dIn', os.path.dirname(pdf_file_path), 
        '-dOut', output_dir, '-exe', 'processFullText'
    ], check=True)
    """
    subprocess.run([
        'java', '-Xmx4G', '-jar', 'grobid-0.7.2/grobid-core/localLibs/wapiti-1.5.0.jar',
        '-gH', 'grobid-0.7.2/grobid-home', '-dIn', os.path.dirname(pdf_file_path), 
        '-dOut', output_dir, '-exe', 'processFullText'
    ], check=True)
     """
    # Construct the expected XML file path from the PDF file name
    xml_file_name = os.path.basename(pdf_file_path).replace('.pdf', '.tei.xml')
    xml_file_path = os.path.join(output_dir, xml_file_name)

    # Read the XML file
    try:
        with open(xml_file_path, 'r', encoding='utf-8') as file:
            xml_data = file.read()
    except Exception as e:
        return f"Error reading XML file: {e}"

    # Convert the XML data to JSON
    try:
        parsed_xml = xmltodict.parse(xml_data)
        json_data = json.dumps(parsed_xml, indent=4)
    except Exception as e:
        return f"Error converting XML to JSON: {e}"

    # Construct JSON file path
    json_file_path = os.path.join(output_dir, xml_file_name.replace('.tei.xml', '.json'))

    # Write the JSON data to a file
    try:
        with open(json_file_path, "w", encoding="utf-8") as file:
            file.write(json_data)
        return f"JSON file saved successfully at {json_file_path}"
    except Exception as e:
        return f"Error saving JSON file: {e}"

# The function can now be called from a main.py script with a PDF file path as input.
