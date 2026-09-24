import zipfile
import xml.etree.ElementTree as ET

path = r'd:\WorkingSpace\ac_project\casualvae\Technical_Proposal_V4.docx'
doc = zipfile.ZipFile(path)
root = ET.XML(doc.read('word/document.xml'))
text = '\n'.join(''.join(node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text) for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p') if ''.join(node.text for node in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text))
print(text)
