"""
Data Preparation Script for Neural Machine Translation

MEMORY-OPTIMIZED VERSION:
- Uses streaming to avoid loading all data into memory
- Uses reservoir sampling to select test set without full shuffle
- Writes output in chunks to prevent memory exhaustion

Datasets:
- VNJPTranslate: Japanese <-> Vietnamese
- JParaCrawl: English <-> Japanese  
- OpenSubtitles: English <-> Vietnamese

Output:
- data/train.jsonl: Training data
- data/test_set.jsonl: Locked test set (2000 samples) - NEVER train on this!
"""

import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterator

# Try to import pyarrow for parquet support
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print("WARNING: pyarrow not installed. JParaCrawl (parquet) will be skipped.")
    print("Install with: pip install pyarrow")

# Configuration
RANDOM_SEED = 42
TEST_SET_SIZE = 2000
DATA_DIR = Path(__file__).parent.parent / "data"

# Dataset paths
VNJP_CSV = DATA_DIR / "vn_ja" / "VNJPTranslate" / "cleaned_merged_two_batch.csv"
JPARACRAWL_DIR = DATA_DIR / "en_ja" / "JParaCrawl" / "data"
OPENSUBTITLES_EN = DATA_DIR / "en_vi" / "OpenSubtitles.en-vi.en"
OPENSUBTITLES_VI = DATA_DIR / "en_vi" / "OpenSubtitles.en-vi.vi"

# Output paths
TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
TEST_OUTPUT = DATA_DIR / "test_set.jsonl"
TEMP_OUTPUT = DATA_DIR / "temp_all.jsonl"


def normalize_text(text: str) -> str:
    """
    Normalize text by removing HTML tags, emojis, and extra whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    
    # Remove emojis and other unicode symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def create_instruction_sample(
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str
) -> dict | None:
    """Create an instruction-format sample."""
    source_clean = normalize_text(source_text)
    target_clean = normalize_text(target_text)
    
    if len(source_clean) < 2 or len(target_clean) < 2:
        return None
    
    return {
        "instruction": f"Translate {source_lang} to {target_lang}",
        "input": source_clean,
        "output": target_clean
    }


def stream_opensubtitles() -> Iterator[dict]:
    """Stream OpenSubtitles English-Vietnamese samples."""
    if not OPENSUBTITLES_EN.exists() or not OPENSUBTITLES_VI.exists():
        print(f"  ERROR: OpenSubtitles files not found!")
        return
    
    with open(OPENSUBTITLES_EN, 'r', encoding='utf-8') as f_en, \
         open(OPENSUBTITLES_VI, 'r', encoding='utf-8') as f_vi:
        for en_line, vi_line in zip(f_en, f_vi):
            sample = create_instruction_sample("En", "Vi", en_line, vi_line)
            if sample:
                yield sample


def stream_vnjp_translate() -> Iterator[dict]:
    """Stream VNJPTranslate Japanese-Vietnamese samples."""
    if not VNJP_CSV.exists():
        print(f"  ERROR: VNJPTranslate CSV not found!")
        return
    
    with open(VNJP_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jp_text = row.get('jp', '')
            vn_text = row.get('vn', '')
            sample = create_instruction_sample("Ja", "Vi", jp_text, vn_text)
            if sample:
                yield sample


def stream_jparacrawl() -> Iterator[dict]:
    """Stream JParaCrawl English-Japanese samples."""
    if not HAS_PYARROW:
        return
    
    if not JPARACRAWL_DIR.exists():
        print(f"  ERROR: JParaCrawl directory not found!")
        return
    
    parquet_files = sorted(JPARACRAWL_DIR.glob("*.parquet"))
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        df = table.to_pandas()
        
        if 'translation' in df.columns:
            for _, row in df.iterrows():
                trans = row['translation']
                if isinstance(trans, dict):
                    en_text = trans.get('en', '')
                    ja_text = trans.get('ja', '')
                    sample = create_instruction_sample("En", "Ja", en_text, ja_text)
                    if sample:
                        yield sample


def reservoir_sample(iterator, k, seed=42):
    """
    Reservoir sampling: Select k random items from an iterator of unknown size.
    O(n) time, O(k) space - perfect for large datasets!
    """
    random.seed(seed)
    reservoir = []
    
    for i, item in enumerate(iterator):
        if i < k:
            reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    
    return reservoir


def main():
    """Main function with memory-optimized processing."""
    print("=" * 60)
    print("Neural Machine Translation - Data Preparation")
    print("(Memory-Optimized Version)")
    print("=" * 60)
    
    random.seed(RANDOM_SEED)
    print(f"\nRandom seed: {RANDOM_SEED}")
    print(f"Test set size: {TEST_SET_SIZE:,}")
    
    # Clean up old files
    for f in [TRAIN_OUTPUT, TEST_OUTPUT, TEMP_OUTPUT]:
        if f.exists():
            os.chmod(f, 0o644)  # Make writable
            f.unlink()
    
    # PASS 1: Stream all data to temp file and count
    print("\n" + "-" * 40)
    print("Pass 1: Streaming all data to temp file...")
    print("-" * 40)
    
    total_count = 0
    with open(TEMP_OUTPUT, 'w', encoding='utf-8') as f_out:
        # OpenSubtitles
        print("  Processing OpenSubtitles (En-Vi)...")
        os_count = 0
        for sample in stream_opensubtitles():
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            os_count += 1
            if os_count % 500000 == 0:
                print(f"    {os_count:,} samples...")
        print(f"  -> {os_count:,} samples from OpenSubtitles")
        total_count += os_count
        
        # VNJPTranslate
        print("  Processing VNJPTranslate (Ja-Vi)...")
        vn_count = 0
        for sample in stream_vnjp_translate():
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            vn_count += 1
            if vn_count % 100000 == 0:
                print(f"    {vn_count:,} samples...")
        print(f"  -> {vn_count:,} samples from VNJPTranslate")
        total_count += vn_count
        
        # JParaCrawl
        if HAS_PYARROW:
            print("  Processing JParaCrawl (En-Ja)...")
            jp_count = 0
            for sample in stream_jparacrawl():
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                jp_count += 1
            print(f"  -> {jp_count:,} samples from JParaCrawl")
            total_count += jp_count
    
    print(f"\nTotal samples: {total_count:,}")
    print(f"Temp file size: {TEMP_OUTPUT.stat().st_size / (1024*1024):.1f} MB")
    
    # PASS 2: Use reservoir sampling to select test set
    print("\n" + "-" * 40)
    print("Pass 2: Selecting test set with reservoir sampling...")
    print("-" * 40)
    
    def temp_file_iterator():
        with open(TEMP_OUTPUT, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    
    test_samples = reservoir_sample(temp_file_iterator(), TEST_SET_SIZE, RANDOM_SEED)
    test_inputs = {s['input'] for s in test_samples}
    print(f"  Selected {len(test_samples):,} samples for test set")
    
    # PASS 3: Write train set (everything not in test set)
    print("\n" + "-" * 40)
    print("Pass 3: Writing train and test sets...")
    print("-" * 40)
    
    train_count = 0
    with open(TRAIN_OUTPUT, 'w', encoding='utf-8') as f_train:
        with open(TEMP_OUTPUT, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                sample = json.loads(line)
                if sample['input'] not in test_inputs:
                    f_train.write(line)
                    train_count += 1
                    if train_count % 500000 == 0:
                        print(f"    {train_count:,} train samples written...")
    
    print(f"  -> {train_count:,} training samples written")
    
    # Write test set
    with open(TEST_OUTPUT, 'w', encoding='utf-8') as f_test:
        for sample in test_samples:
            f_test.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"  -> {len(test_samples):,} test samples written")
    
    # Lock test set
    os.chmod(TEST_OUTPUT, 0o444)
    print(f"  -> LOCKED {TEST_OUTPUT}")
    
    # Clean up temp file
    TEMP_OUTPUT.unlink()
    print(f"  -> Cleaned up temp file")
    
    # Final stats
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nTrain set: {train_count:,} samples ({TRAIN_OUTPUT.stat().st_size / (1024*1024):.1f} MB)")
    print(f"Test set:  {len(test_samples):,} samples ({TEST_OUTPUT.stat().st_size / 1024:.1f} KB)")
    
    print("\nSample from train set:")
    with open(TRAIN_OUTPUT, 'r', encoding='utf-8') as f:
        print(json.dumps(json.loads(f.readline()), ensure_ascii=False, indent=2))
    
    print("\nSample from test set:")
    print(json.dumps(test_samples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
