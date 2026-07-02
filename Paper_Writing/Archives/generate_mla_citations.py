import os
import glob
import re
try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    import subprocess
    subprocess.check_call(['pip3', 'install', 'PyPDF2'])
    import PyPDF2

def extract_metadata_from_pdf(pdf_path):
    """Extract title, authors, year, and venue from PDF"""
    try:
        # Read PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) == 0:
                return None

            # Get first page text
            first_page = reader.pages[0].extract_text()
            lines = [line.strip() for line in first_page.split('\n') if line.strip()]

        metadata = {
            'title': '',
            'authors': [],
            'year': '',
            'venue': '',
            'volume': '',
            'issue': '',
            'pages': '',
            'doi': ''
        }

        # Extract title (usually first few non-empty lines)
        title_lines = []
        for i, line in enumerate(lines[:10]):
            if len(line) > 20 and not line.isupper():
                title_lines.append(line)
                if i > 0 and len(title_lines) >= 2:
                    break
        metadata['title'] = ' '.join(title_lines[:3])

        # Extract authors (look for patterns like "Name Name" or "Name, Name")
        author_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        full_text = ' '.join(lines[:50])
        potential_authors = re.findall(author_pattern, full_text)
        metadata['authors'] = potential_authors[:6]  # Limit to first 6 authors

        # Extract year from text and filename
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, full_text)
        if years:
            metadata['year'] = years[0]
        else:
            # Try filename
            filename = os.path.basename(pdf_path)
            years_in_filename = re.findall(year_pattern, filename)
            if years_in_filename:
                metadata['year'] = years_in_filename[-1]

        # Look for journal/conference venue
        venue_keywords = ['IEEE', 'Transactions', 'Journal', 'Conference', 'Proceedings', 'Robotica', 'Robotics']
        for line in lines[:30]:
            for keyword in venue_keywords:
                if keyword in line:
                    metadata['venue'] = line
                    break
            if metadata['venue']:
                break

        # Look for DOI
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', full_text)
        if doi_match:
            metadata['doi'] = doi_match.group(1)

        # Look for volume/issue
        vol_match = re.search(r'[Vv]ol\.?\s*(\d+)', full_text[:1000])
        if vol_match:
            metadata['volume'] = vol_match.group(1)

        issue_match = re.search(r'[Nn]o\.?\s*(\d+)', full_text[:1000])
        if issue_match:
            metadata['issue'] = issue_match.group(1)

        return metadata

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def format_mla_citation(metadata, paper_num):
    """Format metadata as MLA citation"""

    # Format authors
    if metadata['authors']:
        if len(metadata['authors']) == 1:
            authors = metadata['authors'][0]
        elif len(metadata['authors']) == 2:
            authors = f"{metadata['authors'][0]} and {metadata['authors'][1]}"
        else:
            authors = f"{metadata['authors'][0]}, et al."
    else:
        authors = "Unknown Author"

    # Build citation
    citation = f"[{paper_num}] {authors}."

    if metadata['title']:
        citation += f' "{metadata['title']}."'

    if metadata['venue']:
        citation += f" {metadata['venue']},"

    if metadata['volume']:
        citation += f" vol. {metadata['volume']},"

    if metadata['issue']:
        citation += f" no. {metadata['issue']},"

    if metadata['year']:
        citation += f" {metadata['year']}"

    if metadata['pages']:
        citation += f", pp. {metadata['pages']}"

    if metadata['doi']:
        citation += f". https://doi.org/{metadata['doi']}"

    citation += "."

    return citation

def main():
    papers_dir = "/Users/christopherwaight/Desktop/Agentic-AI/intro_writer/Papers"
    pdf_files = sorted(glob.glob(os.path.join(papers_dir, "*.pdf")))

    print(f"Found {len(pdf_files)} PDF files. Extracting metadata...")

    citations = []

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing [{i}/{len(pdf_files)}]: {filename}")

        metadata = extract_metadata_from_pdf(pdf_path)

        if metadata:
            citation = format_mla_citation(metadata, i)
            citations.append(citation)
            print(f"  Title: {metadata['title'][:80]}...")
            print(f"  Authors: {', '.join(metadata['authors'][:3])}")
            print(f"  Year: {metadata['year']}")
        else:
            citations.append(f"[{i}] ERROR: Could not extract metadata from {filename}")

    # Write to file
    output_file = "/Users/christopherwaight/Desktop/Agentic-AI/intro_writer/mla_citations.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("MLA Citations for Papers\n")
        f.write("=" * 80 + "\n\n")
        for citation in citations:
            f.write(citation + "\n\n")

    print(f"\n\nCitations saved to: {output_file}")
    print(f"Total citations generated: {len(citations)}")

if __name__ == "__main__":
    main()
