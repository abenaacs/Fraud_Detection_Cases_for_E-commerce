import os

def create_project_structure(base_dir):
    # Define the folder structure
    structure = {
        base_dir: {
            "data": [
                "Fraud_Data.csv",  # Placeholder for raw data
                "IpAddress_to_Country.csv",
                "creditcard.csv"
            ],
            "src": [
                "__init__.py",
                "data_loader.py",
                "data_cleaner.py",
                "feature_engineer.py",
                "main.py"
            ],
            "tests": [
                "test_data_loader.py",
                "test_data_cleaner.py",
                "test_feature_engineer.py"
            ],
            "notebooks": [
                "exploratory_data_analysis.ipynb"
            ],
            "requirements.txt": "",
            "README.md": ""
        }
    }

    # Create directories and files
    for root, contents in structure.items():
        os.makedirs(root, exist_ok=True)  # Create the base directory
        for dir_name in contents:
            if isinstance(contents[dir_name], list):  # Subdirectories or files
                for file_name in contents[dir_name]:
                    file_path = os.path.join(root, dir_name, file_name)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    open(file_path, 'a').close()  # Create empty file
            else:  # Root-level files
                file_path = os.path.join(root, dir_name)
                open(file_path, 'a').close()  # Create empty file

if __name__ == "__main__":
    # Specify the base directory for the project
    BASE_DIR = "fraud_detection"
    create_project_structure(BASE_DIR)
    print(f"Project structure created at '{BASE_DIR}'.")