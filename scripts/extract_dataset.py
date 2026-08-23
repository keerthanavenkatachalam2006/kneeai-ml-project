"""Extract OAIZIB-CM dataset from ZIPs for training."""
import sys
sys.path.insert(0, r'E:\KneeAI\backend')

from app.segmentation.dataset import extract_all_data
from app.config import PROCESSED_DATA

extract_dir = PROCESSED_DATA / "extracted"
print(f"Extracting to: {extract_dir}")
extract_all_data(extract_dir)
print("Done!")
