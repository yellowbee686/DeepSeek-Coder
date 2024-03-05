import os
from pathlib import Path
from datasets import Dataset, DatasetDict

def collect_files(input_dir, extensions):
    """
    遍历目录，收集指定扩展名的文件。
    """
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(Path(root) / filename)
    return files

def preprocess_file(file_path, input_dir):
    """
    读取文件内容，并在最前面添加文件路径注释。
    """
    relative_path = file_path.relative_to(input_dir)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content_with_path = f"# {relative_path}\n{content}"
    return str(relative_path), content_with_path

def create_dataset(files, input_dir):
    """
    为收集到的文件创建一个dataset。
    """
    data = {'path': [], 'text': []}
    for file_path in files:
        path, content = preprocess_file(file_path, input_dir)
        data['path'].append(path)
        data['text'].append(content)
    return Dataset.from_dict(data)

def save_dataset(dataset, output_dir):
    """
    将dataset保存到指定目录。
    """
    dataset.save_to_disk(output_dir)

def main(input_dir, output_dir):
    extensions = ['.py', '.h', '.cc', '.cu']
    files = collect_files(input_dir, extensions)
    dataset = create_dataset(files, input_dir)
    save_dataset(dataset, output_dir)

# 示例用法
input_dir = '/data01/home/xiao.cheng/code/autra_code'
output_dir = '/data01/home/xiao.cheng/code/autra_dataset'
main(input_dir, output_dir)
