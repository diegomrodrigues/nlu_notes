from pathlib import Path
from typing import Set
from dataclasses import dataclass
import re
from .chain import TaskChain, ChainStep
from .processor import TaskProcessor

@dataclass
class FilenameResult:
    filename: str
    path: Path
    exists: bool = False

class FilenameHandler:
    def __init__(self, processor: TaskProcessor, tasks_config: dict):
        self.processor = processor
        self.tasks_config = tasks_config

    def _get_existing_numbers(self, directory: Path) -> Set[int]:
        """Get set of existing file numbers in the directory."""
        existing_files = [f for f in directory.glob("*.md")]
        return {
            int(f.name.split('.')[0]) 
            for f in existing_files 
            if f.name[0].isdigit()
        }

    def _check_similar_exists(self, directory: Path, suggested_name: str) -> str:
        """Check if a similar filename exists and return it if found."""
        for existing_file in directory.glob("*.md"):
            existing_name = ' '.join(existing_file.name.split(' ')[1:]).replace('.md', '')
            if existing_name.lower() == suggested_name.lower():
                return existing_file.name
        return False

    def create_section_directory(self, base_dir: Path, section_name: str) -> Path:
        """Create a section directory with proper sequential numbering."""
        # Remove any existing number prefixes from the input section name
        section_name = re.sub(r'^\d+\.\s*', '', section_name)
        clean_section_name = section_name.lower().strip()
        
        # Get existing section directories
        existing_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        
        # First check for existing similar sections
        for existing_dir in existing_dirs:
            # Remove the number prefix and clean up for comparison
            existing_name = re.sub(r'^\d+\.\s*', '', existing_dir.name).lower().strip()
            if existing_name == clean_section_name:
                print(f"✓ Using existing section: {existing_dir.name}")
                return existing_dir
        
        # If no existing section found, create new one with next available number
        existing_numbers = {
            int(d.name.split('.')[0]) 
            for d in existing_dirs 
            if d.name[0].isdigit()
        }
        
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1
        
        section_dir_name = f"{next_num:02d}. {section_name}"
        section_dir = base_dir / section_dir_name
        
        if not section_dir.exists():
            section_dir.mkdir(parents=True)
            print(f"✓ Created new section: {section_dir_name}")
            
        return section_dir

    def generate_filename(self, directory: Path, topic: str) -> FilenameResult:
        """Generate a filename for the topic with proper numbering."""
        # Create chain for filename generation
        steps = [
            ChainStep(
                name="Generate Filename",
                tasks=["generate_filename_task"]
            )
        ]
        chain = TaskChain(self.processor, self.tasks_config, steps)
        
        # Get suggested name
        suggested_name = chain.run(topic)
        if not suggested_name:
            raise ValueError("Failed to generate filename")
        suggested_name = suggested_name.strip()
        
        # Check for existing similar file
        existing_file = self._check_similar_exists(directory, suggested_name)
        if existing_file:
            # Try to generate an alternative name by running the chain again
            # with additional context about avoiding the existing name
            alternative_prompt = f"""
            Generate a different name for this topic.
            Current topic: {topic}
            Existing name to avoid: {suggested_name}
            """
            alternative_name = chain.run(alternative_prompt)
            
            if alternative_name and alternative_name.strip() != suggested_name:
                suggested_name = alternative_name.strip()
                # Check if the alternative name also exists
                existing_file = self._check_similar_exists(directory, suggested_name)
                
            if existing_file:
                # If we still have a conflict, fall back to adding a number
                base_name = suggested_name
                counter = 1
                while self._check_similar_exists(directory, f"{base_name} {counter}"):
                    counter += 1
                suggested_name = f"{base_name} {counter}"
        
        # Get next available number prefix
        existing_numbers = self._get_existing_numbers(directory)
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1
        
        # Create new filename
        new_filename = f"{next_num:02d}. {suggested_name}.md"
        return FilenameResult(
            filename=new_filename,
            path=directory / new_filename,
            exists=False
        ) 