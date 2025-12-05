import os
import zipfile

def zip_folder(folder_path: str, out_zip: str):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, folder_path)
                zf.write(full, arc)
    return out_zip
