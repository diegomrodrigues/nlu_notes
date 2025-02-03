from pathlib import Path
import os
import yaml
from agent.processor import TaskProcessor
from agent.directory_processor import DirectoryProcessor
import argparse
from dotenv import load_dotenv

BASE_DIR = "./01. Transformers"

CONTEXT = "Estudo Avançado de Large Language Models e Natural Language Understanding"
PERSPECTIVES = [
    "Foque nos fundamentos teóricos e matemáticos por trás dos large language models, incluindo modelos probabilísticos, information theory e conceitos matemáticos relevantes.",
    "Foque nos conceitos fundamentais e arquiteturas dos large language models, incluindo attention mechanisms, transformers, tokenization e model architectures.",
    "Foque nos aspectos práticos de implementação, incluindo training procedures, optimization techniques, deployment strategies e considerações de scaling."
]

TARGET_FOLDERS = [
]
EXCLUDED_FOLDERS = [
]

# Processing parameters
NUM_TOPICS = None
MAX_WORKERS = 2
JSONS_PER_PERSPECTIVE = 3
NUM_CONSOLIDATION_STEPS = 2
MAX_PREVIOUS_TOPICS = 5

def load_tasks_config(tasks_dir: str = './agent/tasks') -> dict:
    """Load all YAML files from the tasks directory into a single config dictionary."""
    tasks_config = {}
    tasks_path = Path(tasks_dir)
    
    for yaml_file in tasks_path.glob('*.yaml'):
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            tasks_config.update(config)
    
    return tasks_config

def get_numbered_folders(base_dir: Path) -> list[str]:
    """
    Get all numbered folders from the base directory, excluding specified folders.
    If TARGET_FOLDERS is specified, only return those folders.
    Returns folders sorted numerically.
    """
    if TARGET_FOLDERS:
        return sorted([folder for folder in TARGET_FOLDERS if folder not in EXCLUDED_FOLDERS])
    
    folders = [
        folder.name for folder in base_dir.iterdir()
        if folder.is_dir() 
        and folder.name.strip()[0].isdigit()
        and folder.name not in EXCLUDED_FOLDERS
    ]
    return sorted(folders)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process study materials with topic generation')
    parser.add_argument('--debug', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable debug mode (true/false)')
    args = parser.parse_args()
    
    # Load configuration and initialize processor
    tasks_config = load_tasks_config()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    processor = TaskProcessor(api_key=api_key)
    directory_processor = DirectoryProcessor(
        processor, 
        tasks_config, 
        CONTEXT,
        debug=args.debug
    )
    
    # Define base directory and settings
    base_dir = Path(BASE_DIR)
    target_folders = get_numbered_folders(base_dir)
    
    # Process each target directory
    for folder in target_folders:
        directory = base_dir / folder
        if directory.exists():
            directory_processor.process_with_topics(
                directory,
                perspectives=PERSPECTIVES,
                num_topics=NUM_TOPICS or len(PERSPECTIVES),
                max_workers=MAX_WORKERS,
                jsons_per_perspective=JSONS_PER_PERSPECTIVE,
                num_consolidation_steps=NUM_CONSOLIDATION_STEPS,
                max_previous_topics=MAX_PREVIOUS_TOPICS
            )
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main()