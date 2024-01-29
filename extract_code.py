import ast
import markdown
import json
from collections import defaultdict
import importlib.util


def code_to_json(code_lines, tree):
    res = {}
    functions = {}
    classes = {}
    direct_imports = {}
    from_imports = defaultdict(list)

    # Extract module-level docstring, if present
    module_docstring = ast.get_docstring(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_code = "\n".join(code_lines[node.lineno - 1: node.end_lineno])
            function_lines = {"start": node.lineno - 1, "end": node.end_lineno}
            function_docstring = ast.get_docstring(node)
            functions[node.name] = {"code": function_code, "lines": function_lines, "docstring": function_docstring}
        elif isinstance(node, ast.ClassDef):
            class_code = "\n".join(code_lines[node.lineno - 1: node.end_lineno])
            class_lines = {"start": node.lineno - 1, "end": node.end_lineno}
            class_docstring = ast.get_docstring(node)
            classes[node.name] = {"code": class_code, "lines": class_lines, "docstring": class_docstring}
        elif isinstance(node, ast.Import):
            for alias in node.names:
                direct_imports[alias.name] = {"alias": alias.asname, "lineno": node.lineno}
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                from_imports[node.module].append({
                    "name": alias.name,
                    "alias": alias.asname,
                    "lineno": node.lineno
                })

    res["functions"] = functions
    res["classes"] = classes
    res["direct_imports"] = direct_imports
    res["from_imports"] = dict(from_imports)  # Convert back to a regular dict for JSON serialization
    res["module_docstring"] = module_docstring

    return res


def get_file_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.py'):
            code = f.read()
            code_lines = code.split('\n')
            tree = ast.parse(code)
            res = code_to_json(code_lines, tree)
        elif file_path.endswith('.ipynb'):
            notebook = json.load(f)
            res = {'cells': [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']}
    return res


def parse_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return markdown.markdown(f.read())


def get_project_structure(project_path):
    res = {}
    file_path = None

    project_name = os.path.basename(project_path.rstrip(os.sep)) # Get the project folder name

    for root, dirs, files in os.walk(project_path):
        # Skip if it's before the project directory
        if project_name not in root:
            continue

        path_parts = root.split(os.sep)
        # Start from the project directory in the path_parts
        start_index = path_parts.index(project_name)
        relevant_path_parts = path_parts[start_index:]

        current_level = res
        for part in relevant_path_parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        structure = current_level

        for directory in dirs:
            structure[directory] = {}

        for file in files:
            try:
                file_path = os.path.join(root, file)
                if file.endswith(".py") or file.endswith(".ipynb"):
                    structure[file] = get_file_code(file_path)
                elif file.endswith(".md"):
                    structure[file] = parse_md(file_path)
            except Exception as e:
                print(f"Failed to process file {file_path}: {e}")

    return res

def summarize_structure(project_structure, indent=''):
    python_files = 0
    ipynb_files = 0
    md_files = 0
    function_counts = []
    class_counts = []
    import_counts = []

    for key, value in project_structure.items():
        if isinstance(value, dict):
            if key.endswith('.py') or key.endswith('.ipynb'):
                python_files += 1
                function_counts.append(len(value.get('functions', {})))
                class_counts.append(len(value.get('classes', {})))
                import_counts.append(len(value.get('imports', {})))
            elif key.endswith('.md'):
                md_files += 1
            else:
                # recurse into subdirectories
                sub_python_files, sub_ipynb_files, sub_md_files, sub_function_counts, sub_class_counts, sub_import_counts = summarize_structure(
                    value, indent + '  ')
                python_files += sub_python_files
                ipynb_files += sub_ipynb_files
                md_files += sub_md_files
                function_counts.extend(sub_function_counts)
                class_counts.extend(sub_class_counts)
                import_counts.extend(sub_import_counts)

    return python_files, ipynb_files, md_files, function_counts, class_counts, import_counts


def get_imports(tree):
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # We have a relative import, which we'll ignore for now
                continue

            module = node.module
            for alias in node.names:
                if module is None:
                    imports.add(alias.name)
                else:
                    imports.add(f"{module}.{alias.name}")

    return imports


def analyze_codebase(directory):
    results = {}

    # Step 2-3: Traverse directory and parse each Python file
    for file_path in traverse_directory(directory):
        with open(file_path, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        code_lines = code.splitlines()
        results[file_path] = code_to_json(code_lines, tree)

    # Step 4-5: Analyze imports to find dependencies
    dependencies = analyze_dependencies(results)

    return results, dependencies


import os


def traverse_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                yield os.path.join(root, file)


def is_external(module):
    try:
        spec = importlib.util.find_spec(module)
        return spec is not None and 'site-packages' in spec.origin
    except ModuleNotFoundError:
        return False


def analyze_dependencies(results):
    dependencies = {file: set() for file in results.keys()}

    for file, structure in results.items():
        if 'direct_imports' in structure:
            for imported_module in structure['direct_imports']:
                if not is_external(imported_module):
                    # Note: This code assumes the imported file is in the same directory
                    # This might not be the case in your project.
                    dependencies[file].add(imported_module)

    return dependencies


def process_atlas_data(atlas_data):
    # Placeholder for the documents
    documents = []

    # Iterate over the files in the atlas data
    for file_path, file_data in atlas_data.items():
        # Check if the file data is a dictionary
        if isinstance(file_data, dict):
            # Process the functions
            for function_name, function_data in file_data.get('functions', {}).items():
                # Construct the document ID
                document_id = f"{file_path}/function/{function_name}"

                # Prepare the document content and metadata
                document_content = json.dumps(function_data)
                document_metadata = {'file_path': file_path, 'type': 'function', 'name': function_name}

                # Add the document to the list
                documents.append((document_id, document_content, document_metadata))

            # Process the classes
            for class_name, class_data in file_data.get('classes', {}).items():
                # Construct the document ID
                document_id = f"{file_path}/class/{class_name}"

                # Prepare the document content and metadata
                document_content = json.dumps(class_data)
                document_metadata = {'file_path': file_path, 'type': 'class', 'name': class_name}

                # Add the document to the list
                documents.append((document_id, document_content, document_metadata))

    return documents