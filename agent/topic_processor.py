from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .processor import TaskProcessor, retry_on_error
from .chain import TaskChain, ChainStep
from .filename_handler import FilenameHandler
import threading

@dataclass
class TopicResult:
    """Represents the result of topic processing."""
    topic: str
    content: str
    success: bool = False

class TopicProcessor:
    """Handles the processing of topics and sections with improved organization and error handling."""
    
    def __init__(self, processor: TaskProcessor, tasks_config: dict, context: str, debug: bool = False):
        self.processor = processor
        self.tasks_config = tasks_config
        self.context = context
        self.filename_handler = FilenameHandler(processor, tasks_config)
        self.debug = debug  # Store debug flag

    def process_section(
        self, 
        directory: Path, 
        section_name: str, 
        section_topics: List[str],
        pdf_files: List[Path], 
        max_previous_topics: int = 5,
        max_iterations: int = 3
    ) -> None:
        """
        Process all topics within a section sequentially.
        
        Args:
            directory: Base directory for saving topics
            section_name: Name of the current section
            section_topics: List of topics to process
            pdf_files: List of PDF files to use as reference
            max_previous_topics: Maximum number of previous topics to use as context
            max_iterations: Maximum number of iterations for task retries
        """
        if self.debug:
            print(f"\n🔵 DEBUG: Processing Section - {section_name}")
            print(f"Topics to process: {section_topics}")
            print(f"PDF Files: {[f.name for f in pdf_files]}")
            print(f"Max Previous Topics: {max_previous_topics}\n")
        
        print(f"\nProcessing section: {section_name} ({len(section_topics)} topics)")
        
        processed_topics: List[TopicResult] = []
        
        for topic in section_topics:
            try:
                result = self._process_sub_topic(
                    directory,
                    section_name,
                    topic,
                    pdf_files,
                    processed_topics,
                    max_previous_topics,
                    max_iterations
                )
                if result and result.success:
                    processed_topics.append(result)
            except Exception as e:
                print(f"❌ Exception processing topic '{topic}': {str(e)}")

    def _process_sub_topic(
        self,
        directory: Path,
        section_name: str,
        topic: str,
        pdf_files: List[Path],
        processed_topics: List[TopicResult],
        max_previous_topics: int,
        max_iterations: int
    ) -> Optional[TopicResult]:
        """Process a single sub-topic with enhanced error handling and retries."""
        try:
            if self.debug:
                print(f"\n🔍 DEBUG: Starting _process_sub_topic for '{topic}'")
                print(f"  - Directory: {directory}")
                print(f"  - Section: {section_name}")
                print(f"  - Previous topics count: {len(processed_topics)}")
            
            current_processed = processed_topics[-max_previous_topics:] if max_previous_topics else processed_topics.copy()
            
            if self.debug:
                print(f"  - Using {len(current_processed)} previous topics for context")
            
            result = self._process_single_topic(
                directory=directory,
                section_name=section_name,
                topic=topic,
                pdf_files=pdf_files,
                previous_topics=current_processed,
                max_previous_topics=max_previous_topics,
                max_iterations=max_iterations
            )
            
            if self.debug:
                print(f"  - Initial processing result: {'Success' if result and result.success else 'Failed'}")
            
            if result and result.success:                                
                save_success = self._save_topic_content(directory, section_name, topic, result.content)
                if not save_success:
                    print("❌ Error saving content to file")
                    return None

            return result
        except Exception as e:
            if self.debug:
                print(f"❌ DEBUG: Exception in _process_sub_topic:")
                print(f"  - Topic: {topic}")
                print(f"  - Error: {str(e)}")
            print(f"❌ Failed to process sub-topic '{topic}': {str(e)}")
            return None

    @retry_on_error(max_retries=3)
    def _process_single_topic(
        self, 
        directory: Path, 
        section_name: str, 
        topic: str,
        pdf_files: List[Path], 
        previous_topics: List[TopicResult],
        max_previous_topics: int,
        max_iterations: int
    ) -> Optional[TopicResult]:
        """Process a single topic and handle its result with enhanced retry logic."""
        try:
            if self.debug:
                print(f"\n🔍 DEBUG: Starting _process_single_topic")
                print(f"  - Topic: {topic[:50]}...")
                print(f"  - PDF files: {[f.name for f in pdf_files]}")
            
            content = self._generate_with_retries(
                directory=directory,
                section_name=section_name,
                topic=topic,
                pdf_files=pdf_files,
                previous_topics=previous_topics,
                max_previous_topics=max_previous_topics,
                task="generate_draft_task",
                max_iterations=max_iterations
            )
            
            if self.debug:
                print(f"  - Content generation {'successful' if content else 'failed'}")
                if content:
                    print(f"  - Generated content length: {len(content)}")
            
            if not content:
                return None

            return TopicResult(topic=topic, content=content, success=True)

        except Exception as e:
            if self.debug:
                print(f"❌ DEBUG: Exception in _process_single_topic:")
                print(f"  - Topic: {topic}")
                print(f"  - Error: {str(e)}")
            print(f"❌ Error processing topic '{topic}': {str(e)}")
            return None

    def _generate_with_retries(
        self,
        directory: Path,
        section_name: str,
        topic: str,
        pdf_files: List[Path],
        previous_topics: List[TopicResult],
        max_previous_topics: int,
        task: str,
        max_iterations: int,
        chain: Optional[TaskChain] = None
    ) -> Optional[str]:
        """Generate content with retry logic based on markers and iterations."""
        if self.debug:
            print(f"\n🔍 DEBUG: Starting _generate_with_retries")
            print(f"  - Task: {task}")
            print(f"  - Max iterations: {max_iterations}")
            print(f"  - Using existing chain: {chain is not None}")
        
        iterations = 0
        content = None
        while iterations < max_iterations:
            try:
                if self.debug:
                    print(f"\n  - Attempt {iterations + 1}/{max_iterations}")
                
                if not chain:
                    context = self._build_previous_topics_context(previous_topics, max_previous_topics)
                    chain = self._create_processing_chain(pdf_files, context)
                
                input_text = self._build_input_text(directory, section_name, topic)
                content = chain.run(input_text)
                
                if self.debug:
                    print(f"  - Content generated, length: {len(content)}")
                    print(f"  - END markers found: {content.count('<!-- END -->')}")
                
                if content.count("<!-- END -->") == 1:
                    break
                elif content.count("<!-- END -->") > 1:
                    print(f"⚠️ Multiple <!-- END --> markers found. Retrying ({iterations + 1}/{max_iterations})...")
                else:
                    print(f"⚠️ <!-- END --> marker not found. Continue ({iterations + 1}/{max_iterations})...")
                
            except Exception as e:
                if self.debug:
                    print(f"  ❌ Attempt {iterations + 1} failed: {str(e)}")
                print(f"❌ Error during '{task}': {str(e)}")
            
            iterations += 1
        
        if self.debug:
            print(f"  - Final result: {'Success' if content else 'Failed'}")
            if content:
                print(f"  - Final content length: {len(content)}")
        
        if content and content.count("<!-- END -->") == 1:
            return content
        elif content and content.count("<!-- END -->") > 1:
            return None
        elif content:
            return content + "\n<!-- END -->"
        return None

    def _create_processing_chain(self, pdf_files: List[Path], context: str) -> TaskChain:
        """Create the processing chain with defined steps."""
        draft_step = self._create_draft_step(pdf_files, context)
        enhancement_step = self._create_enhancement_step()
        return TaskChain(self.processor, self.tasks_config, [draft_step, enhancement_step], debug=self.debug)

    def _create_draft_step(self, pdf_files: List[Path], context: str) -> ChainStep:
        """Create the initial draft generation step."""
        return ChainStep(
            name="Generate Initial Draft",
            tasks=["generate_draft_task"],
            input_files=pdf_files,
            stop_at="<!-- END -->",
            max_iterations=3,
            additional_context=context
        )

    def _create_enhancement_step(self) -> ChainStep:
        """Create the enhancement and review step."""
        return ChainStep(
            name="Review and Enhance",
            tasks=[
                "cleanup_task",
                "generate_logical_steps_task",
                "generate_step_proofs_task",
                "generate_examples_task",
                "format_math_task",
                "cleanup_task"
            ],
            stop_at="<!-- END -->",
            max_iterations=3
        )

    def _build_previous_topics_context(self, previous_topics: List[TopicResult], 
                                     max_previous_topics: int) -> str:
        """Build context string from previous topics."""
        if not previous_topics:
            return ""
            
        context_parts = ["\n\nPrevious Topics:"]
        for topic_result in previous_topics[-max_previous_topics:]:
            context_parts.extend([
                f"--- START {topic_result.topic} ---",
                topic_result.content,
                f"--- END {topic_result.topic} ---"
            ])
        
        return "\n".join(context_parts)

    def _build_input_text(self, directory: Path, section_name: str, topic: str) -> str:
        """Build the input text for the processing chain."""
        return "\n".join([
            f"CONTEXT_PLACEHOLDER = {self.context}",
            f"DIRECTORY_PLACEHOLDER = {directory}",
            f"SECTION_PLACEHOLDER = {section_name}",
            f"SUBTOPIC_PLACEHOLDER = {topic}"
        ])

    def _save_topic_content(self, directory: Path, section_name: str, 
                          topic: str, content: str) -> bool:
        """Save the processed topic content to a file."""
        try:
            section_dir = self.filename_handler.create_section_directory(directory, section_name)
            result = self.filename_handler.generate_filename(section_dir, topic)
                        
            result.path.write_text(content, encoding='utf-8')
            print(f"✔️ Saved topic to: {result.filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save topic: {str(e)}")
            return False 