import fitz  # PyMuPDF
import argparse


def get_toc(doc):
    toc = doc.get_toc()
    json_toc = []
    for t in toc:
        json_toc.append({"lvl": t[0], "title": t[1], "page": t[2]})
    page_ranges = { json_toc[i]['title']: (json_toc[i]['page'], json_toc[i+1]['page']) for i in  range(len(json_toc)-1) }
    return page_ranges

def extract_text(doc, page):
    text = ""
    for page in doc:
        text += page.getText()
    return text

def extract_pages(doc, start, end):
    res = []
    start = max (int(start), 0)
    end = min (int(end), len(doc))

    for i in range(start, end+1):
        res.append(doc[i].get_text("text").strip())
    return res


def extract_section_content(doc, start_page, end_page):
    # Adjusting page numbers to be zero-based
    content = []
    for page_num in range(start_page - 1, end_page):  # end_page is exclusive
        page = doc.load_page(page_num)
        content.append(page.get_text("text"))
    return "\n".join(content).strip()

def get_sections_content(pdf_path, sections_dict):
    doc = fitz.open(pdf_path)
    sections_content = {}
    
    # Extract content for each section using the provided page ranges
    for section_title, (start_page, end_page) in sections_dict.items():
        sections_content[section_title] = extract_section_content(doc, start_page, end_page)
    
    return sections_content

'''
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="The pdf file to be processed")
    parser.add_argument("operation", help="The operation to be performed")
    parser.add_argument("start",help="The starting page")
    parser.add_argument("end",help="The ending page")

    args = parser.parse_args()
    return args



if __name__ == "__main__" :
    args = get_args()

    doc = fitz.open(args.pdf_file)
    operation = args.operation

    if operation == "toc":
        print(get_toc(doc))

    elif operation == "extract_pages":
        start = args.start
        end = args.end
        print(extract_pages(doc, start, end))

    else:
        print(f'Unknown operation {operation}')

'''        

