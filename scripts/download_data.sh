# Exit if anything fails
set -e

# Base data directory
DATA_DIR="data"

echo "=== Preparing directory structure ==="
mkdir -p $DATA_DIR/vn_ja
mkdir -p $DATA_DIR/en_ja
mkdir -p $DATA_DIR/en_vi

echo "=== Checking git-lfs ==="
if ! command -v git-lfs &> /dev/null
then
    echo "git-lfs not found. Please install via:"
    echo "sudo apt install git-lfs"
    exit 1
fi

# 1) VNJPTranslate  (Vi <-> Ja)
echo "=== Downloading VNJPTranslate (Vi-Ja) ==="
cd $DATA_DIR/vn_ja
if [ ! -d "VNJPTranslate" ]; then
    git clone https://huggingface.co/datasets/haiFrHust/VNJPTranslate
else
    echo "VNJPTranslate already exists — skipping clone"
fi
cd ../../

# 2) JParaCrawl (En <-> Ja)
echo "=== Downloading JParaCrawl (En-Ja) ==="
cd $DATA_DIR/en_ja
if [ ! -d "JParaCrawl" ]; then
    git clone https://huggingface.co/datasets/nntsuzu/JParaCrawl
else
    echo "JParaCrawl already exists — skipping clone"
fi
cd ../../

# 3) OpenSubtitles EN <-> VI (OPUS)
echo "=== Downloading OpenSubtitles (En-Vi) ==="
cd $DATA_DIR/en_vi

# OPUS link – zipped parallel data
if [ ! -f "OpenSubtitles.en-vi.txt.zip" ]; then
    wget -O OpenSubtitles.en-vi.txt.zip \
        "https://opus.nlpl.eu/download.php?f=OpenSubtitles/v2018/moses/en-vi.txt.zip"
    echo "Unzipping OpenSubtitles..."
    unzip OpenSubtitles.en-vi.txt.zip
else
    echo "OpenSubtitles already downloaded — skipping"
fi

cd ../../

echo "=== Done! All datasets saved under ./data/ ==="
