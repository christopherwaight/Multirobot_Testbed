import os
import glob
import re
import time
from scholarly import scholarly

def extract_title_from_filename(filename):
    """Extract a searchable title from the PDF filename"""
    # Remove path and extension
    base = os.path.basename(filename).replace('.pdf', '')

    # Remove the reference number in brackets
    base = re.sub(r'^\[\d+\]\s*', '', base)

    # Remove trailing author names and years
    base = re.sub(r'\s+-\s+\w+\s*$', '', base)
    base = re.sub(r'\s+-\s+\d{4}$', '', base)
    base = re.sub(r'\s+-\s+\d{2}$', '', base)

    # Clean up truncated titles
    base = base.strip()

    return base

def format_mla_citation(bib, index):
    """Format a scholarly bibtex entry as MLA citation"""
    try:
        authors = bib.get('author', 'Unknown Author')
        title = bib.get('title', 'Unknown Title')
        year = bib.get('year', 'n.d.')
        venue = bib.get('venue', bib.get('journal', ''))

        # Format authors
        if ',' in authors or ' and ' in authors:
            # Multiple authors
            author_list = authors.replace(' and ', ', ').split(', ')
            if len(author_list) > 3:
                formatted_authors = f"{author_list[0]}, et al."
            elif len(author_list) == 2:
                formatted_authors = f"{author_list[0]} and {author_list[1]}"
            else:
                formatted_authors = authors
        else:
            formatted_authors = authors

        # Build MLA citation
        citation = f"[{index}] {formatted_authors}. \"{title}.\""

        if venue:
            citation += f" {venue},"

        citation += f" {year}."

        return citation
    except Exception as e:
        return f"[{index}] ERROR formatting citation: {e}"

def main():
    papers_dir = "/Users/christopherwaight/Desktop/Agentic-AI/intro_writer/Papers"
    pdf_files = sorted(glob.glob(os.path.join(papers_dir, "*.pdf")))

    print(f"Found {len(pdf_files)} PDF files.")
    print(f"Searching Google Scholar for citations...")
    print("=" * 80)

    citations = []
    successful = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        title = extract_title_from_filename(filename)

        print(f"\n[{i}/{len(pdf_files)}] Searching for: {title[:70]}...")

        try:
            # Search Google Scholar
            search_query = scholarly.search_pubs(title)
            paper = next(search_query)

            # Get bibliographic info
            bib = paper['bib']

            # Format as MLA
            citation = format_mla_citation(bib, i)
            citations.append(citation)

            print(f"   ✓ Found: {bib.get('title', 'Unknown')[:60]}...")
            print(f"   Authors: {bib.get('author', 'Unknown')[:60]}...")
            print(f"   Year: {bib.get('year', 'Unknown')}")

            successful += 1

            # Be respectful to Google Scholar - rate limit
            time.sleep(3)

        except StopIteration:
            print(f"   ✗ Not found on Google Scholar")
            citations.append(f"[{i}] NOT FOUND: {filename}")
            failed += 1
        except Exception as e:
            print(f"   ✗ Error: {e}")
            citations.append(f"[{i}] ERROR: {filename} - {str(e)}")
            failed += 1
            time.sleep(5)  # Longer pause after error

    # Write results
    output_file = "/Users/christopherwaight/Desktop/Agentic-AI/intro_writer/mla_citations_from_scholar.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("MLA Citations from Google Scholar\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Success: {successful}/{len(pdf_files)}\n")
        f.write(f"Failed: {failed}/{len(pdf_files)}\n")
        f.write("=" * 80 + "\n\n")

        for citation in citations:
            f.write(citation + "\n\n")

    print("\n" + "=" * 80)
    print(f"\nCitations saved to: {output_file}")
    print(f"Successfully found: {successful}/{len(pdf_files)} papers")
    print(f"Failed to find: {failed}/{len(pdf_files)} papers")
    print("\nNote: Citations may need manual review and formatting adjustments.")

if __name__ == "__main__":
    main()
