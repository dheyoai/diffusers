# from datasets import Dataset, Image, load_dataset, DatasetDict
# import pandas as pd
# import os
# from huggingface_hub import login, create_repo
# from dotenv import load_dotenv 

# HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")

# dataset_dir = "balayya_dataset"
# image_dir = os.path.join(dataset_dir, "images")
# metadata_path = os.path.join(dataset_dir, "metadata.csv")


# df = pd.read_csv(metadata_path)

# data = {
#     "image": [os.path.join(image_dir, fname) for fname in df["file_name"]],
#     "caption": df["caption"].tolist()
# }


# dataset = Dataset.from_dict(data)

# dataset = dataset.cast_column("image", Image())

# final_dataset = DatasetDict({"train": dataset})

# print(final_dataset)
# # print(final_dataset[0])
# # dataset.save_to_disk("balayya_dataset_local")

# login(token=HF_WRITE_TOKEN)
# create_repo("shivanvitha21/balayya_dataset_v1", repo_type="dataset", private=True)
# final_dataset.push_to_hub("shivanvitha21/balayya_dataset_v1", private=True)



from datasets import Dataset, Image, load_dataset, DatasetDict
import pandas as pd
import os
from huggingface_hub import login, create_repo
from dotenv import load_dotenv 

HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")

dataset_dir = "aishwarya_rajeshan_dataset"
image_dir = os.path.join(dataset_dir, "images")
metadata_path = os.path.join(dataset_dir, "metadata.csv")


df = pd.read_csv(metadata_path)

data = {
    "image": [os.path.join(image_dir, fname) for fname in df["file_name"]],
    "caption": df["caption"].tolist()
    # "special_token": df["special_token"].tolist()
}


dataset = Dataset.from_dict(data)

dataset = dataset.cast_column("image", Image())

dataset = dataset.shuffle(seed=42)

final_dataset = DatasetDict({"train": dataset})

print(final_dataset)
# print(final_dataset[0])
# dataset.save_to_disk("balayya_dataset_local")

login(token=HF_WRITE_TOKEN)
create_repo("shivanvitha21/aishwarya_rajeshan_dataset_v2", repo_type="dataset", private=True)
final_dataset.push_to_hub("shivanvitha21/aishwarya_rajeshan_dataset_v2", private=True)