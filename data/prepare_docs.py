# prepare_docs.py
# This script makes:
# - a list of all abbreviations in the documents
# - a list of all the terms defined in the documents
# - a set of new documents stripped of irrelevant information

from docx import Document
import os
from tqdm import tqdm
import re
from Levenshtein import distance as levenshtein_distance

def extract_text_under_heading(doc_path, target_heading, target_level):
    doc = Document(doc_path)
    headings_and_text = []
    capture_text = False
    current_text = []

    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            try:
                heading_level = int(paragraph.style.name.split()[-1])
            except ValueError:
                continue
            if capture_text and heading_level <= target_level:
                break
            if paragraph.text.strip().lower().__contains__(target_heading.lower()) and heading_level == target_level:
                capture_text = True
                continue

        if capture_text:
            current_text.append(paragraph.text)
    
    if current_text:
        headings_and_text.append((target_heading, "\n".join(current_text)))

    return headings_and_text

def extract_from_directory(directory, target_heading, target_level):
    all_texts = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('.docx'):
            file_path = os.path.join(directory, filename)
            extracted_text = extract_text_under_heading(file_path, target_heading, target_level)
            if extracted_text:
                all_texts[filename] = extracted_text
    return all_texts

def normalize_text(text):
    # Convert to lowercase and strip leading/trailing whitespace
    text = text.lower().strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def is_subset(shorter, longer):
    return shorter.lower() in longer.lower()

def make_abbreviations(directory, abbrevs_heading, abbrevs_heading_level):
    all_texts = extract_from_directory(directory, abbrevs_heading, abbrevs_heading_level)
    abbreviations = {}
    normalized_forms = {}  # Dictionary to keep track of normalized forms
    
    for texts in all_texts.values():
        for heading, text in texts:
            for line in text.split("\n"):
                # Check if the line contains an abbreviation
                if len(line.split("\t")) == 2:
                    abbreviation, full_form = line.split("\t")

                    # Skip if the abbreviation is less than 2 characters long
                    if len(abbreviation.strip()) < 2:
                        continue
                                        
                    normalized_full_form = normalize_text(full_form)
                    
                    if abbreviation not in abbreviations:
                        abbreviations[abbreviation] = full_form
                        normalized_forms[abbreviation] = {normalized_full_form}
                    else:
                        # Check for subsets and supersets
                        is_subset_of_existing = any(is_subset(full_form, existing) for existing in abbreviations[abbreviation].split(" | "))
                        is_superset_of_existing = any(is_subset(existing, full_form) for existing in abbreviations[abbreviation].split(" | "))
                        
                        if is_subset_of_existing:
                            # If it's a subset of an existing form, skip it
                            continue
                        elif is_superset_of_existing:
                            # If it's a superset, replace all subsets with this new form
                            original_forms = [form for form in abbreviations[abbreviation].split(" | ") if not is_subset(form, full_form)]
                            original_forms.append(full_form)
                            abbreviations[abbreviation] = " | ".join(sorted(set(original_forms)))
                            normalized_forms[abbreviation] = {normalize_text(form) for form in original_forms}
                        elif any(levenshtein_distance(normalized_full_form, normalized_form) < 3 for normalized_form in normalized_forms[abbreviation]):
                            # If it's very similar to an existing form, skip it
                            continue
                        elif normalized_full_form not in normalized_forms[abbreviation]:
                            # If it's a new unique form, add it
                            original_forms = abbreviations[abbreviation].split(" | ")
                            original_forms.append(full_form)
                            abbreviations[abbreviation] = " | ".join(sorted(set(original_forms)))
                            normalized_forms[abbreviation].add(normalized_full_form)
    
    return abbreviations

def find_appearing_abbreviations(question):
    # read abbreviations
    with open('data/abbreviations.txt', 'r') as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            abbreviation, full_form = line.split(': ', 1)
            abbreviations[abbreviation.strip()] = full_form.strip()
    # sort abbreviations by length in descending order to handle cases
    #  where one abbreviation is a substring of another
    sorted_abbrevs = sorted(abbreviations.items(), key=lambda x: len(x[0]), reverse=True)
    assert isinstance(question, dict)
    appearing_abbreviations = set()  # Use a set to store unique abbreviations

    for abbreviation, full_form in sorted_abbrevs:
        # find the abbreviation in the text
        pattern = r'\b' + re.escape(abbreviation) + r'\b'
        # if the abbreviation is found:
        if re.search(pattern, (question['question'].split('?')[0])):
            appearing_abbreviations.add(abbreviation)
        for key in question:
            if key.startswith('option'):
                if re.search(pattern, question[key]):
                    appearing_abbreviations.add(abbreviation)
        if 'answer' in question:
            if re.search(pattern, question['answer']):
                appearing_abbreviations.add(abbreviation)
        if 'explanation' in question:
            if re.search(pattern, question['explanation']):
                appearing_abbreviations.add(abbreviation)
    
    # return a list of dicts with the abbreviation and its full form
    returned_abbreviations = [{abbrev: abbreviations[abbrev]} for abbrev in appearing_abbreviations]
    return returned_abbreviations