import os
import ftplib
import zipfile
import glob
from docx import Document

# The 8 critical RAN specifications for 5G
TARGET_SPECS = [
    "38.331", # RRC
    "38.211", # PHY channels
    "38.212", # PHY multiplexing
    "38.213", # PHY control
    "38.214", # PHY data
    "38.321", # MAC
    "38.322", # RLC
    "38.323"  # PDCP
]

FTP_HOST = "ftp.3gpp.org"
FTP_DIR = "/Specs/latest/Rel-17/38_series/"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TMP_DIR = os.path.join(DATA_DIR, "tmp")

def setup_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        print(f"  [!] Failed to parse {docx_path}: {e}")
        return ""

def download_and_parse():
    setup_dirs()
    print(f"Connecting to {FTP_HOST}...")
    
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login() # Anonymous login
    ftp.cwd(FTP_DIR)
    
    print(f"Fetching file list from {FTP_DIR}...")
    files = ftp.nlst()
    
    for spec in TARGET_SPECS:
        # Convert "38.331" to "38331" to match filenames like "38331-h00.zip"
        spec_prefix = spec.replace(".", "")
        
        # Find the zip file for this spec
        matching_files = [f for f in files if f.startswith(spec_prefix) and f.endswith(".zip")]
        if not matching_files:
            print(f"[-] Could not find zip file for {spec}")
            continue
            
        # Usually there's only one latest zip per spec in the 'latest' folder
        target_file = matching_files[0]
        zip_path = os.path.join(TMP_DIR, target_file)
        
        print(f"[+] Downloading {target_file} for TS {spec}...")
        with open(zip_path, 'wb') as f:
            ftp.retrbinary(f"RETR {target_file}", f.write)
            
        print(f"  > Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TMP_DIR)
            
        # Find the extracted .docx file
        docx_files = glob.glob(os.path.join(TMP_DIR, "*.docx"))
        if not docx_files:
            # Some older specs might be .doc, but Rel-17 should be .docx
            print(f"  [!] No .docx found inside {target_file}")
            
            # Clean up whatever was extracted
            for ext_file in os.listdir(TMP_DIR):
                os.remove(os.path.join(TMP_DIR, ext_file))
            continue
            
        docx_path = docx_files[0]
        
        print(f"  > Parsing text from {os.path.basename(docx_path)}...")
        text = extract_text_from_docx(docx_path)
        
        if text:
            # Save to data directory
            out_filename = f"ts_{spec.replace('.', '_')}_rel17.txt"
            out_path = os.path.join(DATA_DIR, out_filename)
            
            # Prepend metadata header for the chunker
            header = f"3GPP TS {spec} Release 17\n====================\n\n"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(header + text)
            print(f"  > Saved to {out_filename} ({len(text)} chars)")
            
        # Clean up tmp folder for the next spec
        for ext_file in os.listdir(TMP_DIR):
            os.remove(os.path.join(TMP_DIR, ext_file))
            
    ftp.quit()
    print("Done! Check the data/ directory.")

if __name__ == "__main__":
    download_and_parse()
