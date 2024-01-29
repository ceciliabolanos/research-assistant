import argparse
import os
import json
import subprocess
import shutil
import tempfile


def get_project_structure(project_path):
    res = {}
    repo_name = os.path.basename(project_path)

    for root, dirs, files in os.walk(project_path):
        # Trim the root to start from the repository's root directory
        trimmed_root = root[len(project_path):].lstrip(os.sep)
        path_parts = trimmed_root.split(os.sep)
        current_level = res

        # Construct the nested dictionary structure
        for part in path_parts:
            current_level = current_level.setdefault(part, {})

        for file in files:
            try:
                file_path = os.path.join(root, file)
                if file.endswith(".py") or file.endswith(".ipynb"):
                    current_level[file] = get_file_code(file_path)
                elif file.endswith(".md"):
                    current_level[file] = parse_md(file_path)
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")

    # Wrap in the repository's name
    return {repo_name: res}

def clone_github_repo(github_url, dest_folder):
    """
    Clone a GitHub repository to the specified local directory, even if the directory is not empty.
    """
    try:
        # Create a temporary directory to clone the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            # Clone the repository into the temporary directory
            subprocess.run(["git", "clone", github_url, temp_dir], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Move the contents of the cloned repo to the destination folder
            repo_name = os.path.basename(github_url.rstrip('/').split('/')[-1])
            source_folder = os.path.join(temp_dir, repo_name)
            for item in os.listdir(source_folder):
                s = os.path.join(source_folder, item)
                d = os.path.join(dest_folder, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
            print("Repository cloned successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e.stderr.decode().strip()}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise
