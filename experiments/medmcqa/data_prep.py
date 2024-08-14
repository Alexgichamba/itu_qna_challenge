import gdown

file_id = "1jbS0TaqgA-DOyi7khOOLIVYPiBt-LRG_"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "data.tar"
gdown.download(url, output, quiet=False)