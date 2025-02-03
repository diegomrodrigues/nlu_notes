from pathlib import Path
from typing import List, Optional
import json
from concurrent.futures import ThreadPoolExecutor
from .processor import TaskProcessor
from .topic_generator import TopicGenerator
from .topic_processor import TopicProcessor
from .utils import retry_on_error

class DirectoryProcessor:
    """Handles the processing of directories and their contents."""
    
    def __init__(self, processor: TaskProcessor, tasks_config: dict, context: str, debug: bool = False):
        self.processor = processor
        self.tasks_config = tasks_config
        self.topic_generator = TopicGenerator(processor, tasks_config, debug=debug)
        self.topic_processor = TopicProcessor(processor, tasks_config, context, debug=debug)
        self.debug = debug

    @retry_on_error(max_retries=3)
    def process_with_topics(
        self,
        directory: Path,
        perspectives: Optional[List[str]] = None,
        num_topics: Optional[int] = None,
        max_workers: Optional[int] = 3,
        jsons_per_perspective: Optional[int] = 3,
        num_consolidation_steps: Optional[int] = 2,
        max_previous_topics: Optional[int] = 5
    ):
        """Generate topics and process them for a directory with retries."""
        try:
            # First, generate topics if they don't exist
            topics_file = directory / "topics.json"
            if not topics_file.exists():
                self.topic_generator.generate(
                    directory, 
                    perspectives, 
                    num_topics, 
                    jsons_per_perspective, 
                    num_consolidation_steps
                )
            
            # Then process the topics
            self.process_directory(directory, max_workers, max_previous_topics)
            
        except Exception as e:
            print(f"❌ Failed to process directory: {directory}")
            print(f"Error: {str(e)}")
            raise

    def process_directory(self, directory: Path, max_workers: int, max_previous_topics: int) -> None:
        """Process a single directory with its sections using parallel processing."""
        print(f"\nProcessing directory: {directory}")
        
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in the directory, skipping...")
            return

        try:
            topics_data = self._read_topics_file(directory)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for topic_obj in topics_data["topics"]:
                    section_name = topic_obj["topic"]
                    section_topics = topic_obj["sub_topics"]
                    
                    future = executor.submit(
                        self.topic_processor.process_section,
                        directory,
                        section_name,
                        section_topics,
                        pdf_files,
                        max_previous_topics
                    )
                    futures.append(future)

                for future in futures:
                    future.result()

        except Exception as e:
            print(f"❌ Failed to process directory: {directory}")
            print(f"Error: {str(e)}")
            raise

    def _read_topics_file(self, directory: Path) -> dict:
        """Read and parse the topics.json file."""
        topics_file = directory / "topics.json"
        if topics_file.exists():
            content = topics_file.read_text(encoding='utf-8')
            return json.loads(content)
        raise FileNotFoundError("topics.json not found in the specified directory") 