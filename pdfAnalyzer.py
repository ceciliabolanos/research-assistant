import fitz  # PyMuPDF
import sys
import json
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

        

