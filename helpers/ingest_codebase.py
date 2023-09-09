import os
import time
import matplotlib.pyplot as plt
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATASET_PATH = os.environ["DATASET_PATH"]
DB_FAISS_PATH = './vectorstore/db_faiss'

sentence_transformer_model = "msmarco-distilbert-base-tas-b"
start_time = time.time()
doc_count = []
times = [] 

# file types that you want to ingest all coding languages
extensions = ['.py', '.java', '.js', '.ts' , '.md', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.html', '.css', '.xml', '.json', '.yaml', '.yml', '.sh', '.bat', '.txt', '.rst', '.sql', '.rb', '.pl', '.swift', '.m', '.mm', '.kt', '.gradle', '.groovy', '.scala', '.clj', '.cljs', '.cljc', '.edn', '.lua', '.coffee', 'pdf']

documents = []

def run_fast_scandir(dir, ext):
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)


    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

subfolders, files = run_fast_scandir(DATASET_PATH, extensions)

for file in files:
    try:
        loader = TextLoader(file, encoding='utf-8')
        documents.extend(loader.load_and_split())
    except RuntimeError as e:
        if isinstance(e.__cause__, UnicodeDecodeError):
            try:
                loader = TextLoader(file, encoding='ISO-8859-1')
                documents.extend(loader.load_and_split())
            except Exception as e_inner:
                print(f"Failed to load {file} due to error: {e_inner}")
        else:
            print(f"Failed to load {file} due to an unexpected error: {e}")
    doc_count.append(len(documents))
    times.append(time.time() - start_time)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

texts = text_splitter.split_documents(documents)

print(f"Total number of documents pre-processed: {len(documents)}")

embeddings = HuggingFaceEmbeddings(model_name=f'sentence-transformers/{sentence_transformer_model}',
                                    model_kwargs={'device': 'cpu'})

# instead of saving it to a database, we can save it locally in a folder
db = FAISS.from_documents(texts, embeddings)
db.save_local(DB_FAISS_PATH)


# Plotting the progress
plt.figure()
plt.plot(times, doc_count)
plt.xlabel('Time (s)')
plt.ylabel('Number of Documents Processed')
plt.title('Documents Processed Over Time')
plt.savefig('progress_plot.png')
print(f"Progress plot saved as 'progress_plot.png'")
print(f"The script took {time.time() - start_time} seconds to complete.")