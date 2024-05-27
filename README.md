# --code
import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""
    
    # Extract text from each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    
    return text

def organize_text_into_dict(text):
    # Combined regex pattern for headings
    heading_pattern = re.compile(
        r'^(?:[IVXLCDM]+\.\s)?(?:\d+\)\s)?(?:[A-Z]\.\s)?[A-Z\s]+\n',
        re.MULTILINE
    )
    
    # Find all headings
    headings = heading_pattern.findall(text)
    
    # Split the text by headings
    content_parts = heading_pattern.split(text)
    
    # Create a dictionary with headings as keys and their corresponding content as values
    organized_text = {}
    for i in range(1, len(content_parts)):
        heading = headings[i-1].strip()
        content = content_parts[i].strip()
        organized_text[heading] = content
    
    return organized_text

def main():
    pdf_path = 's.pdf'
    text = extract_text_from_pdf(pdf_path)
    organized_text = organize_text_into_dict(text)
    
    for heading, content in organized_text.items():
        print(f"Heading: {heading}\nContent: {content}\n")
        

if __name__ == "__main__":
    main()
