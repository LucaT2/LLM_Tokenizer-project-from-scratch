import requests
import re
import os

DATASET_FILE = r"E:\Personal Projects\LLM_from_scratch+tokenizer\pythonProject\files\scifi_llm_dataset.txt"
BOOK_IDS = {
     # H.G. Wells
    '35': 'The Time Machine by H. G. Wells',
    '36': 'The War of the Worlds by H. G. Wells',
    '5230': 'The Invisible Man by H. G. Wells',
    '159': 'The Island of Doctor Moreau by H. G. Wells',
    '1013': 'The First Men in the Moon by H. G. Wells',

    # Jules Verne
    '32': 'Journey to the Center of the Earth by Jules Verne',
    '105': 'From the Earth to the Moon by Jules Verne',
    '164': 'Twenty Thousand Leagues under the Sea by Jules Verne',
    '1268': 'The Mysterious Island by Jules Verne',
    '83': 'Around the Moon by Jules Verne',

    # Edgar Rice Burroughs
    '64317': 'A Princess of Mars by Edgar Rice Burroughs',
    '68': 'The Gods of Mars by Edgar Rice Burroughs',

    # Arthur Conan Doyle
    '1952': 'The Lost World by Arthur Conan Doyle',
    '22357': 'The Poison Belt by Arthur Conan Doyle',

    # Mary Shelley & R.L. Stevenson (Sci-Fi Origins)
    '84': 'Frankenstein; Or, The Modern Prometheus by Mary Wollstonecraft Shelley',
    '16': 'The Strange Case of Dr. Jekyll and Mr. Hyde by Robert Louis Stevenson',

    # Other Influential Classics
    '97': 'Flatland: A Romance of Many Dimensions by Edwin Abbott Abbott',
    '21970': 'The Scarlet Plague by Jack London',
    '32032': 'Second Variety by Philip K. Dick',
    '1513': 'Moby Dick; Or, The Whale by Herman Melville', # Proto-sci-fi
    '73723': 'The Machine Stops by E. M. Forster',
    '19141': 'Edison\'s Conquest of Mars by Garrett Putnam Serviss',
    '32338': 'The Metal Monster by Abraham Merritt',
    '18452': 'The Hampdenshire Wonder by J. D. Beresford',
    '521': 'The Last Man by Mary Wollstonecraft Shelley'
}
def word_count(text):
  return len(re.findall(r'\b\w+\b', text.lower()))

def download_data():
    collected_texts = []
    total_words = 0
    book_count = 0

    print("Starting direct download of selected books...")
    if os.path.exists(DATASET_FILE):
        print(f"Output file '{DATASET_FILE}' already exists. Skipping download.")
    else:
        for book_id, title in BOOK_IDS.items():
            url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

            try:
                # Download the book's text
                response = requests.get(url)
                response.raise_for_status()
                text = response.text

                # Clean the Gutenberg header and footer
                start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
                end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

                start_pos = text.find(start_marker)
                if start_pos != -1:
                    start_pos = text.find("\n", start_pos) + 1
                    text = text[start_pos:]

                end_pos = text.rfind(end_marker)  # Use rfind to find the last occurrence
                if end_pos != -1:
                    text = text[:end_pos]

                cleaned_text = text.strip()
                collected_texts.append(cleaned_text)

                current_words = word_count(cleaned_text)
                total_words += current_words
                book_count += 1
                print(f"Downloaded '{title}' | Words: {current_words:,}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to download book ID {book_id}: {e}")

        print("\nFinished downloading all books.")
        if book_count > 0:
            print(f"Final word count: {total_words:,} from {book_count} books.")
            print(f"Saving all collected text to '{DATASET_FILE}'...")
            with open(DATASET_FILE, "w", encoding="utf-8") as f:
                f.write("\n\n".join(collected_texts))
            print("Done")
        else:
            print("No books were downloaded.")