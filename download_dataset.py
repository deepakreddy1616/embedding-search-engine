from sklearn.datasets import fetch_20newsgroups
import os

def download_dataset(max_docs=200):
    data_dir = os.path.join("data", "docs")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Downloading 20 Newsgroups dataset (limiting to {max_docs} documents)...")
    dataset = fetch_20newsgroups(
        subset='train', 
        remove=('headers', 'footers', 'quotes')
    )
    
    # Limit to max_docs
    total_docs = min(len(dataset['data']), max_docs)
    
    print(f"Saving {total_docs} documents...")
    for idx in range(total_docs):
        text = dataset["data"][idx]
        filepath = os.path.join(data_dir, f"doc_{idx:03d}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
    
    print(f"✅ Downloaded and saved {total_docs} documents to {data_dir}")

if __name__ == "__main__":
    download_dataset(max_docs=200)
