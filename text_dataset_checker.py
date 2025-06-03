import os
import openai
from datasets import load_dataset
import json
from tqdm import tqdm
import concurrent.futures
from huggingface_hub import login

login(token = os.getenv("HF_TOKEN"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ds_1 = load_dataset("sailor2/Vietnamese_RAG", "expert")

# fields = set()
# for i in range(len(ds_1["train"])):
#     fields.add(ds_1["train"][i]["field"])

# print(fields)

# # count fields 'Kỹ thuật và Công nghệ'
# count = 0
# for i in range(len(ds_1["train"])):
#     if ds_1["train"][i]["field"] in ["Kỹ thuật và Công nghệ", "Hoá học", "Vật lý và Thiên văn học", "toán học"]:
#         count += 1

# print(count)

# ds_2 = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG")
# print(ds_2['train'][0].keys())

ds_3 = load_dataset("taidng/UIT-ViQuAD2.0")

titles = set()
for i in range(len(ds_3["train"])):
    titles.add(ds_3["train"][i]["title"])

count = 0
for i in range(len(ds_3["train"])):
    if ds_3["train"][i]["title"] in  [
        'Thiên văn học',
        'Phần mềm giáo dục',
        'Sao Kim',
        'Erwin Schrödinger',
        'Pyrit',
        'Voyager 1',
        'Terbi',
        'Công nghệ thông tin và truyền thông',
        'Antimon',
        'Enrico Fermi',
        'Sao Hải Vương',
        'Nhôm',
        'Zirconi',
        'Albert Einstein',
        'Hệ thống nhúng',
        'Tầng đối lưu',
        'Ngôn ngữ máy',
        'Xeri',
        'John von Neumann',
        'Johannes Kepler',
        'Lý thuyết tập hợp',
        'Isaac Newton',
        'Samsung Electronics',
        'Crom'
    ]:
        count += 1

print(count)


