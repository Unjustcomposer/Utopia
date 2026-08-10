import os
import shutil

def replace_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            return # skip binary files
            
    new_content = content.replace('NexusAI', 'Utopia')
    new_content = new_content.replace('nexusai', 'utopia')
    new_content = new_content.replace('NEXUSAI', 'UTOPIA')
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")

def main():
    root_dir = os.path.abspath('.')
    
    if os.path.exists('nexusai'):
        os.rename('nexusai', 'utopia')
        print("Renamed directory nexusai -> utopia")
        
    if os.path.exists('nexusai_sdk.py'):
        os.rename('nexusai_sdk.py', 'utopia_sdk.py')
        print("Renamed file nexusai_sdk.py -> utopia_sdk.py")
        
    if os.path.exists('nexusai.egg-info'):
        import shutil
        shutil.rmtree('nexusai.egg-info')
        print("Removed nexusai.egg-info")
        
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude directories
        dirnames[:] = [d for d in dirnames if d not in ('.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', 'build', 'dist', '.agents')]
        
        for filename in filenames:
            if filename.endswith(('.pyc', '.db', '.pdf', '.npz', '.png', '.jpg', '.jpeg', '.lock', '.exe', '.dll')):
                continue
            if filename == 'rename_script.py':
                continue
            
            file_path = os.path.join(dirpath, filename)
            replace_in_file(file_path)

if __name__ == '__main__':
    main()
