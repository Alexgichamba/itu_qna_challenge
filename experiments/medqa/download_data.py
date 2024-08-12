import gdown

file_id = "1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "data_clean.zip"
gdown.download(url, output, quiet=False)