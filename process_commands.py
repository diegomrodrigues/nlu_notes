import argparse
from pathlib import Path
from dotenv import load_dotenv
import os
from agent.command_processor import CommandProcessor
from agent.processor import TaskProcessor

def load_tasks_config(tasks_dir: str = './agent/tasks') -> dict:
    """Load all YAML files from the tasks directory into a single config dictionary."""
    tasks_config = {}
    tasks_path = Path(tasks_dir)
    
    for yaml_file in tasks_path.glob('*.yaml'):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            tasks_config.update(config)
    
    return tasks_config

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process files with command annotations')
    parser.add_argument('directory', type=str, nargs='?', default='.',
                       help='Directory to process (default: current directory)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    args = parser.parse_args()

    # Initialize components
    tasks_config = load_tasks_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")

    processor = TaskProcessor(api_key=api_key)
    command_processor = CommandProcessor(processor, tasks_config)
    
    # Process target directory
    target_dir = Path(args.directory)
    if not target_dir.exists():
        raise ValueError(f"Directory not found: {target_dir}")

    print(f"Processing directory: {target_dir.resolve()}")
    command_processor.process_directory(target_dir)
    print("Processing completed")

if __name__ == "__main__":
    main()