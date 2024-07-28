"""
Script Overview
----------------------------

Description:
------------
This script processes a collection of documents to extract and organize information relevant to telecommunications. 
It performs the following tasks:
1. Creates a list of all abbreviations found in the documents.
2. Creates a list of all terms defined in the documents.
3. Generates a set of new documents with irrelevant information stripped out.

Key Components:
---------------
1. **Document Indexing:** Uses the `extract_text_under_heading` function to extract text under specific headings from .docx files.
2. **Abbreviation Extraction:** The `make_abbreviations` function extracts abbreviations and their definitions, 
    handling various edge cases to ensure accuracy.
3. **Question Processing:** The `find_appearing_abbreviations` and `find_full_forms` functions identify and expand
    abbreviations within the questions, ensuring full context is provided.


Usage:
------
1. Ensure all required libraries are installed.
2. Place the document files in the `data/rel18` directory.
3. Prepare the question files (`data/qs_dev.txt`, `data/qs_eval.txt`, `data/qs_train.txt`, `data/366qs.txt`) in JSON format.
4. Run the script to extract abbreviations, expand abbreviations in questions, and process the documents.
5. Check the output files (`data/abbreviations.txt`, `data/qs_dev_full_forms.txt`, etc.) for results.


"""

from docx import Document
import os
from tqdm import tqdm
import re
import json
from Levenshtein import distance as levenshtein_distance


def extract_text_under_heading(doc_path, target_heading, target_level):
    """
    Extracts text from a document under a specified heading and heading level.

    This function reads a document and captures all text under a specified heading until it encounters
    another heading of the same or higher level.

    Args:
        doc_path (str): The path to the document file.
        target_heading (str): The heading under which text should be extracted.
        target_level (int): The heading level to match for the target heading.

    Returns:
        list of tuple: A list containing a tuple with the target heading and the extracted text.
    """
    # Load the document
    doc = Document(doc_path)

    # Initialize variables to store headings and text
    headings_and_text = []
    capture_text = False
    current_text = []

    # Iterate through each paragraph in the document
    for paragraph in doc.paragraphs:
        # Check if the paragraph is a heading
        if paragraph.style.name.startswith("Heading"):
            try:
                # Extract the heading level from the style name
                heading_level = int(paragraph.style.name.split()[-1])
            except ValueError:
                # Skip if the heading level is not an integer
                continue

            # Stop capturing text if a heading of the same or higher level is encountered
            if capture_text and heading_level <= target_level:
                break

            # Start capturing text if the target heading and level are matched
            if (
                paragraph.text.strip().lower().__contains__(target_heading.lower())
                and heading_level == target_level
            ):
                capture_text = True
                continue

        # Append the paragraph text to the current text if capturing is active
        if capture_text:
            current_text.append(paragraph.text)

    # Add the captured text to the list if any text was captured
    if current_text:
        headings_and_text.append((target_heading, "\n".join(current_text)))

    return headings_and_text


def extract_from_directory(directory, target_heading, target_level):
    """
    Extracts text from all .docx files in a specified directory under a given heading and heading level.

    This function iterates through all .docx files in the provided directory, extracts text under the specified
    heading and heading level using the `extract_text_under_heading` function, and stores the results in a dictionary.

    Args:
        directory (str): The path to the directory containing .docx files.
        target_heading (str): The heading under which text should be extracted.
        target_level (int): The heading level to match for the target heading.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of tuples containing headings and extracted text.
    """
    all_texts = {}
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".docx"):
            file_path = os.path.join(directory, filename)
            extracted_text = extract_text_under_heading(
                file_path, target_heading, target_level
            )
            if extracted_text:
                all_texts[filename] = extracted_text
    return all_texts


def normalize_text(text):
    """
    Normalizes the input text by converting it to lowercase, stripping leading and trailing whitespace,
    and replacing multiple spaces with a single space.

    Args:
        text (str): The text to be normalized.

    Returns:
        str: The normalized text.
    """
    # Convert to lowercase and strip leading/trailing whitespace
    text = text.lower().strip()
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text


def is_subset(shorter, longer):
    """
    Checks if the 'shorter' string is a subset of the 'longer' string, ignoring case.

    Args:
        shorter (str): The string to check if it is a subset.
        longer (str): The string to check against.

    Returns:
        bool: True if 'shorter' is a subset of 'longer', False otherwise.
    """
    return shorter.lower() in longer.lower()


def make_abbreviations(directory, abbrevs_heading, abbrevs_heading_level):
    """
    Extracts and processes abbreviations from .docx files in a specified directory.

    This function reads all .docx files in the given directory, extracts text under the specified heading and heading level,
    and processes lines containing abbreviations. It normalizes the text, checks for subsets and supersets, and handles
    similar forms using Levenshtein distance.

    Args:
        directory (str): The path to the directory containing .docx files.
        abbrevs_heading (str): The heading under which abbreviations are listed.
        abbrevs_heading_level (int): The heading level to match for the abbreviations heading.

    Returns:
        dict: A dictionary where keys are abbreviations and values are the corresponding full forms.
    """

    all_texts = extract_from_directory(
        directory, abbrevs_heading, abbrevs_heading_level
    )
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
                        is_subset_of_existing = any(
                            is_subset(full_form, existing)
                            for existing in abbreviations[abbreviation].split(" | ")
                        )
                        is_superset_of_existing = any(
                            is_subset(existing, full_form)
                            for existing in abbreviations[abbreviation].split(" | ")
                        )

                        if is_subset_of_existing:
                            # If it's a subset of an existing form, skip it
                            continue
                        elif is_superset_of_existing:
                            # If it's a superset, replace all subsets with this new form
                            original_forms = [
                                form
                                for form in abbreviations[abbreviation].split(" | ")
                                if not is_subset(form, full_form)
                            ]
                            original_forms.append(full_form)
                            abbreviations[abbreviation] = " | ".join(
                                sorted(set(original_forms))
                            )
                            normalized_forms[abbreviation] = {
                                normalize_text(form) for form in original_forms
                            }
                        elif any(
                            levenshtein_distance(normalized_full_form, normalized_form)
                            < 3
                            for normalized_form in normalized_forms[abbreviation]
                        ):
                            # If it's very similar to an existing form, skip it
                            continue
                        elif normalized_full_form not in normalized_forms[abbreviation]:
                            # If it's a new unique form, add it
                            original_forms = abbreviations[abbreviation].split(" | ")
                            original_forms.append(full_form)
                            abbreviations[abbreviation] = " | ".join(
                                sorted(set(original_forms))
                            )
                            normalized_forms[abbreviation].add(normalized_full_form)

    return abbreviations


def find_appearing_abbreviations(question):
    """
    Identifies abbreviations appearing in a given question and its options.

    This function reads abbreviations from a file, checks if they appear in the question text or its options,
    and returns a list of dictionaries containing the abbreviations and their full forms.

    Args:
        question (dict): A dictionary representing a question with keys like 'question', 'option1', 'option2', etc.

    Returns:
        list: A list of dictionaries where each dictionary contains an abbreviation and its full form.
    """
    # read abbreviations
    with open("data/abbreviations.txt", "r") as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            abbreviation, full_form = line.split(": ", 1)
            abbreviations[abbreviation.strip()] = full_form.strip()
    # sort abbreviations by length in descending order to handle cases
    #  where one abbreviation is a substring of another
    sorted_abbrevs = sorted(
        abbreviations.items(), key=lambda x: len(x[0]), reverse=True
    )
    assert isinstance(question, dict)
    appearing_abbreviations = set()  # Use a set to store unique abbreviations

    for abbreviation, full_form in sorted_abbrevs:
        # find the abbreviation in the text
        pattern = r"\b" + re.escape(abbreviation) + r"\b"
        # if the abbreviation is found:
        if re.search(pattern, (question["question"].split("?")[0])):
            appearing_abbreviations.add(abbreviation)
        for key in question:
            if key.startswith("option") and question[key] is not None:
                if re.search(pattern, question[key]):
                    appearing_abbreviations.add(abbreviation)

    # return a list of dicts with the abbreviation and its full form
    returned_abbreviations = [
        {abbrev: abbreviations[abbrev]} for abbrev in appearing_abbreviations
    ]
    return returned_abbreviations


def find_full_forms():
    """
    Reads abbreviations from a file and replaces them with their full forms in question files.

    This function reads abbreviations from 'data/abbreviations.txt', processes question files to replace abbreviations
    with their full forms followed by the abbreviation in parentheses, and writes the updated questions back to new files.

    The function processes the following files:
    - 'data/qs_dev.txt'
    - 'data/qs_eval.txt'
    - 'data/qs_train.txt'
    - 'data/366qs.txt'

    The updated files are saved with the suffix '_full_forms' added to the original filenames.

    Raises:
        Warning: If a key in the question dictionary is not a string or is null/missing.
    """
    # read abbreviations
    with open("data/abbreviations.txt", "r") as f_abbrevs:
        abbreviations = {}
        for line in f_abbrevs:
            abbreviation, full_form = line.split(": ", 1)
            abbreviations[abbreviation.strip()] = full_form.strip()
    # sort abbreviations by length in descending order to handle cases
    #  where one abbreviation is a substring of another
    sorted_abbrevs = sorted(
        abbreviations.items(), key=lambda x: len(x[0]), reverse=True
    )

    # for each of the files
    for filename in [
        "data/qs_dev.txt",
        "data/qs_eval.txt",
        "data/qs_train.txt",
        "data/366qs.txt",
    ]:
        with open(filename, "r") as f:
            questions = json.load(f)

        # for each question
        for qid, question_dict in tqdm(
            questions.items(), desc=f"Processing {filename}"
        ):

            # find all options
            options = [
                k
                for k in question_dict.keys()
                if k.startswith("option") and question_dict[k] is not None
            ]

            # process question and options
            for key in ["question"] + options:
                if key in question_dict and question_dict[key] is not None:
                    if isinstance(question_dict[key], str):
                        text = question_dict[key]
                        for abbrev, full_form in sorted_abbrevs:
                            # regex to find abbreviation not already in brackets
                            pattern = r"\b" + re.escape(abbrev) + r"\b(?!\s*\))"
                            text = re.sub(pattern, f"{full_form} ({abbrev})", text)
                        question_dict[key] = text
                    else:
                        print(f"Warning: {key} is not a string in question {qid}")
                else:
                    print(f"Warning: {key} is null or missing in question {qid}")

        # write updated questions back to file with suffix '_full_forms'
        with open(filename.replace(".txt", "_full_forms.txt"), "w") as f:
            json.dump(questions, f, indent=4)
