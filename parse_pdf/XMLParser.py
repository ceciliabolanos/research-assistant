import xml.etree.ElementTree as ET
import html

class XMLParser:
    def __init__(self, xml_content):
        self.root = ET.fromstring(xml_content)
        self.namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0'
        }


    def get_title(self):
        title = self.root.find('.//tei:titleStmt/tei:title', self.namespaces)
        return title.text if title is not None else None

    def get_abstract(self):
        abstract = self.root.find('.//tei:profileDesc/tei:abstract', self.namespaces)
        abstract_text = "".join(abstract.itertext())
        return html.unescape(abstract_text.strip()) if abstract is not None else None

    def get_references(self):
        refs = self.root.findall('.//tei:listBibl/tei:biblStruct', self.namespaces)
        references = []
        for ref in refs:
            title = ref.find('.//tei:title', self.namespaces)
            authors = ref.findall('.//tei:author/tei:persName', self.namespaces)
            author_list = [
                {
                    'forename': a.find('.//tei:forename', self.namespaces).text if a.find('.//tei:forename', self.namespaces) is not None else "Unknown",
                    'surname': a.find('.//tei:surname', self.namespaces).text if a.find('.//tei:surname', self.namespaces) is not None else "Unknown"
                }
                for a in authors
            ]
            references.append({
                'title': title.text if title is not None else "No title",
                'authors': author_list
            })
        return references
    
    def get_figures(self):
        figures = self.root.findall('.//tei:figure', self.namespaces)
        figure_list = []
        for fig in figures:
            label = fig.find('.//tei:label', self.namespaces).text if fig.find('.//tei:label', self.namespaces) is not None else "No label"
            desc = fig.find('.//tei:figDesc', self.namespaces).text if fig.find('.//tei:figDesc', self.namespaces) is not None else "No description"
            figure_list.append({'label': label, 'description': desc})
        return figure_list

    def get_body_content(self):
        body = self.root.find('.//tei:text/tei:body', self.namespaces)
        sections = body.findall('.//tei:div', self.namespaces)
        content = []
        footnotes = self.extract_footnotes()  # Extract footnotes first for later reference

        for section in sections:
            head = section.find('.//tei:head', self.namespaces)
            section_title = head.text if head is not None else "No title"
            section_number = head.get('n') if head is not None else "Unknown"  # Get the section number from the head element
            section_id = f"Section {section_number}: {section_title}"
            paragraphs = section.findall('.//tei:p', self.namespaces)
            section_content = {
                'title': section_id,
                'paragraphs': []
            }
            for para in paragraphs:
                para_text = html.unescape("".join(para.itertext()).strip())
                refs = para.findall('.//tei:ref[@type="foot"]', self.namespaces)
                for ref in refs:
                    ref_id = ref.get('target')[1:]  # Skip the '#' character
                    if ref_id in footnotes:
                        footnote_text = footnotes[ref_id]
                        para_text += f" [Footnote: {footnote_text}]"
                    else:
                        para_text += " [Footnote not found]"
                section_content['paragraphs'].append(para_text)
            content.append(section_content)
        return content

    def extract_footnotes(self):
        footnotes = {}
        for note in self.root.findall('.//tei:note[@place="foot"]', self.namespaces):
            note_id = note.get('{http://www.w3.org/XML/1998/namespace}id')  # Access using the namespace
            note_text = html.unescape("".join(note.itertext()).strip())
            if note_id:
                footnotes[note_id] = note_text
        return footnotes