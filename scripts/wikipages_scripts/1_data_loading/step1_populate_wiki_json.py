import re
import pandas as pd
import argparse
from utils import save_json

def read_data(file_path):
    """
    Read CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(file_path, sep='\t')
    df.fillna('', inplace=True)
    return df


def normalize_text(text):
    """
    Normalize the text for comparison.
    """
    if pd.isna(text):
        return ''
    # Remove invisible characters and normalize whitespace
    text = remove_invisible_characters(text)
    return re.sub(r'\s+', ' ', text.strip()).lower()


def remove_invisible_characters(text):
    """
    Remove various invisible characters from the text.
    """
    # List of invisible characters to remove
    invisible_chars = [
        '\u00A0',  # Non-breaking space
        '\u200F',  # Right-to-left mark
        '\u200E',  # Left-to-right mark
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\uFEFF',  # Zero-width no-break space
    ]
    for char in invisible_chars:
        text = text.replace(char, '')
    return text


def generate_wiki_pages(df):
    """
    Generate a dictionary of wiki pages with all related information.
    """
    wiki_pages = {}
    examine_count = 0

    for index, row in df.iterrows():
        examine_count += 1

        page_url = row['page_url']
        summary = row['context_page_description']
        section_title = row['section_title']
        hierarchical_section_title = row['hierarchical_section_title']
        section_description = row['context_section_description']
        image_url = row['image_url']

        # If page_url not in wiki_pages, initialize the entry
        if page_url not in wiki_pages:
            wiki_pages[page_url] = {
                "url": page_url,
                "title": row['page_title'],
                "date": None,  # Assuming no date information, can be added if available
                "language": row['language'],
                "texts": [],
                "images": []
            }

        # Normalize texts
        normalized_summary = normalize_text(summary)
        normalized_section_description = normalize_text(section_description) if pd.notna(section_description) else ''

        # Add summary section if not already added
        if not any(text['section_type'] == ["summary"] for text in wiki_pages[page_url]['texts']):
            summary_id = len(wiki_pages[page_url]['texts'])
            wiki_pages[page_url]['texts'].append({
                "section_title": "Summary",
                "hierarchical_section_title": hierarchical_section_title,
                "section_type": ["summary"],
                "section_id": summary_id,
                "content": [normalized_summary],
            })

        # Add section information, checking if it duplicates the summary or any existing section
        if pd.notna(section_description):
            existing_section = next(
                (text for text in wiki_pages[page_url]['texts'] if
                 text['content'][0] == normalized_section_description),
                None
            )
            if existing_section:
                section_id = existing_section['section_id']
            elif normalized_section_description != normalized_summary:
                section_id = len(wiki_pages[page_url]['texts'])
                wiki_pages[page_url]['texts'].append({
                    "section_title": section_title,
                    "hierarchical_section_title": hierarchical_section_title,
                    "section_type": ["context"],
                    "section_id": section_id,
                    "content": [normalized_section_description],
                })
            else:
                section_id = summary_id  # Use summary_id if section_description is not valid or equal to the summary

        elif pd.notna(image_url):
            # Handle cases with no content but with images
            section_id = len(wiki_pages[page_url]['texts'])
            wiki_pages[page_url]['texts'].append({
                "section_title": section_title,
                "hierarchical_section_title": hierarchical_section_title,
                "section_type": ["context"],
                "section_id": section_id,
                "content": [''],
            })

        if section_id is None:
            section_id = summary_id  # Default to summary_id if no other section_id is set

        # Add image information, associating with the correct section_id
        if pd.notna(image_url):
            wiki_pages[page_url]['images'].append({
                "image_url": image_url,
                "image_id": len(wiki_pages[page_url]['images']),
                "section_id": section_id
            })

        # Output the examined status
        if examine_count % 1000 == 0:
            print(f'Now processed {examine_count} values')

    # Convert dictionary to list
    wiki_pages_list = list(wiki_pages.values())
    print(f'Total number of wit data exmined: {examine_count}, generate {len(wiki_pages_list)} wiki pages')
    return wiki_pages_list


def main(input_file, output_file, language=None, dry_run=False):
    # Read the data
    df = read_data(input_file)

    # Filter the DataFrame by language if specified
    if language:
        df = df[df['language'] == language]
    
    # Only take 1 million data
    df = df.head(1000000)

    # Generate the wiki pages dictionary
    wiki_pages = generate_wiki_pages(df)

    if dry_run:
        print("Dry run enabled. Data processing completed but no output file was written.")
    else:
        # Save the results to a JSON file
        save_json(wiki_pages, output_file)
        print("Data has been successfully saved to", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some wiki pages.')
    parser.add_argument('input_file', type=str, help='The input TSV file path')
    parser.add_argument('output_file', type=str, help='The output JSON file path')
    parser.add_argument('--language', type=str, help='The language to filter by', default=None)
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run (process data but do not save)')

    args = parser.parse_args()

    main(args.input_file, args.output_file, args.language, args.dry_run)

