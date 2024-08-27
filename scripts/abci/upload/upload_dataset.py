import os
import argparse
from tqdm import tqdm
from huggingface_hub import HfApi, create_repo


def upload_directory(api, local_dir, repo_name, repo_type, branch_name, start_range, end_range):
    for root, dirs, files in os.walk(local_dir):
        # Filter directories and files within the specified range
        filtered_dirs = [
            d for d in dirs
            if d.startswith("obelics-train-")
            and "of-01335" in d
            and start_range <= int(d.split('-')[2]) <= end_range
        ]

        filtered_files = [
            file for file in files
            if file.startswith("obelics-train-")
            and "of-01335" in file
            and start_range <= int(file.split('-')[2]) <= end_range
        ]

        # Upload filtered directories
        for dir_name in tqdm(filtered_dirs, desc=f"Uploading directories in {root}"):
            local_path = os.path.join(root, dir_name)
            repo_path = os.path.relpath(local_path, local_dir)

            print(f"Uploading {repo_path} to branch {branch_name}...")
            api.upload_folder(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_name,
                repo_type=repo_type,
                commit_message=f"Upload {repo_path}",
                revision=branch_name,
            )
            print(f"Successfully uploaded {repo_path}!")

        # Upload filtered files
        for file in tqdm(filtered_files, desc=f"Uploading files in {root}"):
            local_path = os.path.join(root, file)
            repo_path = os.path.relpath(local_path, local_dir)

            print(f"Uploading {repo_path} to branch {branch_name}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_name,
                repo_type=repo_type,
                commit_message=f"Upload {repo_path}",
                revision=branch_name,
            )
            print(f"Successfully uploaded {repo_path}!")


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=str, help="Path to the checkpoint directory")
parser.add_argument("--repo-name", type=str, help="Name of the Hugging Face repository")
parser.add_argument("--branch-name", type=str, default="main", help="Branch name in the repository")
parser.add_argument("--start-range", type=int, help="Start of the file or directory range (e.g., 0 for 00000)")
parser.add_argument("--end-range", type=int, help="End of the file or directory range (e.g., 99 for 00099)")
args = parser.parse_args()

converted_ckpt: str = args.ckpt_path
repo_name: str = args.repo_name
branch_name: str = args.branch_name
start_range: int = args.start_range
end_range: int = args.end_range

try:
    create_repo(repo_name, repo_type="dataset", private=True)
except Exception as e:
    print(f"Repository {repo_name} already exists! Error: {e}")

api = HfApi()
if branch_name != "main":
    try:
        api.create_branch(
            repo_id=repo_name,
            repo_type="dataset",
            branch=branch_name,
        )
    except Exception as e:
        print(f"Branch {branch_name} already exists. Error: {e}")

# List of directories to upload
directories = ["OBELICS_converted_parquet", "OBELICS_converted_sample", "OBELICS_img"]

for directory in directories:
    dir_path = os.path.join(converted_ckpt, directory)
    if os.path.exists(dir_path):
        print(f"Starting upload of directory: {dir_path}")
        upload_directory(api, dir_path, repo_name, "dataset", branch_name, start_range, end_range)
    else:
        print(f"Directory {dir_path} does not exist. Skipping.")

print("Upload completed successfully!")
