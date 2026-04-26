import os
import json
from PyPDF2 import PdfReader
import docx
from app.resume_parser import parse_resume


CACHE_FILE = "resumes_cache.json"


# 🔥 PDF extraction
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# 🔥 DOCX extraction
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


# 🔥 Load cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


# 🔥 Save cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def load_candidates_from_resumes(folder_path="resumes"):
    cache = load_cache()
    updated_cache = cache.copy()

    candidates = []

    # 🔥 Handle missing folder
    if not os.path.exists(folder_path):
        print(f"Resumes folder not found: {folder_path}")
        return candidates

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        file_name_lower = file_name.lower()

        # 🔥 Skip non-files
        if not os.path.isfile(file_path):
            continue

        # 🔥 If cached → use it
        if file_name in cache:
            print(f"Using cache for {file_name}")
            candidates.append(cache[file_name])
            continue

        try:
            print(f"Parsing {file_name}")

            # 🔥 Extract text
            if file_name_lower.endswith(".pdf"):
                with open(file_path, "rb") as f:
                    text = extract_text_from_pdf(f)

            elif file_name_lower.endswith(".docx"):
                with open(file_path, "rb") as f:
                    text = extract_text_from_docx(f)

            else:
                print(f"Skipping unsupported file: {file_name}")
                continue

            # 🔥 Parse with LLM
            parsed = parse_resume(text)

            if parsed:
                candidates.append(parsed)
                updated_cache[file_name] = parsed  # cache it

        except Exception as e:
            print(f"Error parsing {file_name}: {e}")

    # 🔥 Save cache
    save_cache(updated_cache)

    return candidates