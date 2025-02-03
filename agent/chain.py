from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from .processor import TaskProcessor
import json
import tempfile
from .utils import retry_on_error

@dataclass
class ChainStep:
    """Represents a step in the task chain with its associated configuration.
    
    Core Fields:
        name: Human-readable name for the step
        tasks: List of task names to execute in this step
    
    Input Configuration:
        input_files: Optional list of input files to process
        use_previous_result: Whether to use previous step's result
        additional_context: Additional context to prepend to content
        
    Output Control:
        expect_json: Whether to expect JSON output
        extract_json: Whether to extract JSON from text response
        stop_at: String pattern to stop generation
        max_iterations: Maximum number of continuation attempts
    """
    # Core fields
    name: str
    tasks: List[str]
    
    # Input configuration
    input_files: Optional[List[Path]] = None
    use_previous_result: bool = False
    additional_context: Optional[str] = None
    
    # Output control
    expect_json: bool = False
    extract_json: bool = False
    stop_at: Optional[str] = None
    max_iterations: int = 5

    def __post_init__(self):
        """Validate the step configuration after initialization."""
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.tasks:
            raise ValueError("Step must contain at least one task")

class TaskChain:
    """Manages the sequential processing of tasks in steps."""
    
    def __init__(self, task_processor: TaskProcessor, tasks_config: Dict[str, Dict[str, Any]], steps: List[ChainStep], debug: bool = False):
        """Initialize the task chain with a processor, configuration, and steps."""
        self.processor = task_processor
        self.tasks_config = tasks_config
        self.steps = steps
        self.previous_result = None  # Store previous step result
        self.debug = debug  # Store debug flag
        self.validate_config()
        self.validate_steps()
    
    def validate_config(self):
        """Validate that all required tasks exist in the configuration."""
        if not self.tasks_config:
            raise ValueError("Tasks configuration cannot be empty")
        
        required_keys = ['system_instruction', 'user_message']
        for task_name, task_config in self.tasks_config.items():
            missing_keys = [key for key in required_keys if key not in task_config]
            if missing_keys:
                raise ValueError(f"Task '{task_name}' is missing required keys: {missing_keys}")
    
    def validate_steps(self):
        """Validate all steps and their tasks exist in configuration."""
        if not self.steps:
            raise ValueError("Steps list cannot be empty")
            
        for step in self.steps:
            missing_tasks = [task for task in step.tasks if task not in self.tasks_config]
            if missing_tasks:
                raise ValueError(f"Step '{step.name}' contains undefined tasks: {missing_tasks}")

    
    @retry_on_error(max_retries=3)
    def run(self, initial_content: str) -> Optional[str]:
        """Run all steps in the chain sequentially with retries."""
        if self.debug:
            print("\n🔍 DEBUG: Starting chain execution")
            print(f"  - Number of steps: {len(self.steps)}")
            print(f"  - Initial content length: {len(initial_content)}")
        
        print("🔄 Starting task chain execution...")
        
        current_content = initial_content
        for i, step in enumerate(self.steps, 1):
            if self.debug:
                print(f"\n🔍 DEBUG: Running step {i}/{len(self.steps)}")
                print(f"  - Step name: {step.name}")
                print(f"  - Tasks: {step.tasks}")
            
            result = self.process_step(step, current_content)
            if result:
                if self.debug:
                    print(f"  ✔️ Step completed successfully")
                    print(f"  - Result length: {len(result)}")
                current_content = result
            else:
                if self.debug:
                    print(f"  ❌ Step failed")
                raise Exception(f"❌ Chain failed at step: {step.name}")
        
        if self.debug:
            print("\n✨ DEBUG: Chain execution completed")
            print(f"  - Final content length: {len(current_content)}")
        
        print("✨ Task chain completed successfully")
        return current_content

    @retry_on_error(max_retries=3)
    def process_step(self, step: ChainStep, content: str) -> Optional[str]:
        """Process all tasks in a single step with retries."""
        if self.debug:
            print("\n🔍 DEBUG: Processing step")
            print(f"  - Step name: {step.name}")
            print(f"  - Input content length: {len(content)}")
            print(f"  - Max iterations: {step.max_iterations}")
            print(f"  - Expect JSON: {step.expect_json}")
            print(f"  - Extract JSON: {step.extract_json}")
        
        content = self._prepare_step_content(step, content)
        uploaded_files = self._handle_file_uploads(step)
        
        if self.debug and uploaded_files:
            print(f"  - Uploaded files: {len(uploaded_files)}")
        
        current_content = content
        last_valid_json = None
        
        for i, task_name in enumerate(step.tasks):
            if self.debug:
                print(f"\n    - Starting task: {i+1}/{len(step.tasks)} {task_name}")
            
            iterations = 0
            while iterations < step.max_iterations:
                print(f"\n→ Executing task {task_name} ({iterations + 1}/{step.max_iterations})")
                
                task_config = self._prepare_task_config(task_name, step, current_content, iterations)
                result = self._execute_task(task_name, task_config, current_content, step, uploaded_files, last_valid_json)
                
                if result is None:
                    if self.debug:
                        print("    ❌ Task execution failed")
                    return None
                
                if self.debug:
                    print(f"    ✔️ Task execution successful")
                    print(f"    - Result length: {len(result)}")
                
                current_content, last_valid_json = self._process_task_result(
                    result, current_content, step, iterations, last_valid_json
                )
                
                if self._should_stop_iteration(current_content, step, last_valid_json):
                    if self.debug:
                        print("    - Task iteration stop condition met")
                    break
                
                iterations += 1
        
        if self.debug:
            print("\n  ✔️ Step processing completed")
            print(f"  - Final content length: {len(current_content)}")
        
        self.previous_result = current_content
        return current_content

    def _prepare_step_content(self, step: ChainStep, content: str) -> str:
        """Prepare initial content for step processing."""
        if step.use_previous_result and self.previous_result:
            content = self.previous_result
            
        if step.additional_context:
            content = f"Additional Context:\n{step.additional_context}\n\n{content}"
        
        return content

    def _prepare_task_config(self, task_name: str, step: ChainStep, current_content: str, iteration: int) -> Dict[str, Any]:
        """Prepare task configuration based on iteration and content type."""
        task_config = self.tasks_config[task_name].copy()
        
        if iteration == 0:
            return task_config
            
        if step.expect_json:
            task_config["user_message"] = self._prepare_json_continuation_prompt(current_content)
        else:
            last_chunk = self._get_last_chunk(current_content)
            task_config["user_message"] = self._prepare_text_continuation_prompt(last_chunk)
            
        return task_config

    def _prepare_json_continuation_prompt(self, content: str) -> str:
        """Prepare prompt for continuing JSON structure."""
        return (
            "Continue completing this JSON structure.\n"
            "Do not repeat any previous content, only provide the missing parts.\n"
            "Exactly from you left and remember to end with the marker <!-- END -->.\n"
        )

    def _prepare_text_continuation_prompt(self, last_chunk: str) -> str:
        """Prepare prompt for continuing text content."""
        return (
            "Continue exactly from where this text ends"
            "Do not repeat any previous content. Here's the last part:\n\n"
            f"{last_chunk}\n\n"
            "Continue the text from this point, providing only new content:\n"
        )

    def _execute_task(self, task_name: str, task_config: Dict[str, Any], content: str, 
                     step: ChainStep, files: Optional[List[Any]], last_valid_json: Optional[str]) -> Optional[str]:
        """Execute a single task with error handling."""
        try:
            result = self.processor.process_task(
                task_name, 
                task_config, 
                content,
                expect_json=step.expect_json,
                extract_json=step.extract_json,
                files=files
            )
            
            if self.debug:
                print(f"Model Response: \n {result[:500]}")

            if not result:
                print(f"❌ Step failed at task: {task_name}")
                return None
                
            return result
            
        except Exception as e:
            print(f"❌ Error in task {task_name}: {str(e)}")
            return None

    def _process_task_result(self, result: str, current_content: str, step: ChainStep, 
                           iteration: int, last_valid_json: Optional[str]) -> tuple[str, Optional[str]]:
        """Process task result and handle JSON/text content appropriately."""
        should_stop = False
        
        if self.debug:
            print("\n" + "═" * 50)
            print(f"🔍 DEBUG: Task Result (Iteration {iteration + 1})")
            print(f"Step: {step.name}")
            print(f"Raw Result:\n{result[:1000]}")
            print("═" * 50 + "\n")
        
        if iteration > 0:
            if step.expect_json:
                current_content = self._handle_json_continuation(current_content, result)
            else:
                current_content = self._handle_text_continuation(current_content, result)
        else:
            current_content = result
            
        if step.expect_json or step.extract_json:
            try:
                json.loads(current_content)
                last_valid_json = current_content
                # Only stop if we have valid JSON and this is the last task
            except json.JSONDecodeError:
                last_valid_json = None

        return current_content, last_valid_json

    def _should_stop_iteration(self, content: str, step: ChainStep, last_valid_json: Optional[str]) -> bool:
        """Determine if iteration should stop based on content and step configuration."""
        # If we expect JSON, only stop if we have valid JSON
        if step.expect_json or step.extract_json:
            return last_valid_json is not None
        
        # If we have a stop pattern, only stop when it's found
        if step.stop_at:
            return step.stop_at in content
        
        # If no special conditions, stop after first iteration
        return True

    def _combine_json_content(self, current: str, new: str) -> str:
        """Attempt to combine two JSON contents intelligently."""
        try:
            current_json = json.loads(current) if current.strip() else {}
            new_json = json.loads(new) if new.strip() else {}
            
            # If both are valid JSON, merge them
            if isinstance(current_json, dict) and isinstance(new_json, dict):
                # Merge logic for topics
                if "topics" in current_json and "topics" in new_json:
                    current_json["topics"].extend(new_json["topics"])
                    # Remove duplicates based on topic names
                    seen = set()
                    unique_topics = []
                    for topic in current_json["topics"]:
                        if topic["topic"] not in seen:
                            seen.add(topic["topic"])
                            unique_topics.append(topic)
                    current_json["topics"] = unique_topics
                
                return json.dumps(current_json, indent=2)
        except json.JSONDecodeError:
            pass
        
        return new  # Return new content if combination fails

    def _handle_file_uploads(self, step: ChainStep) -> Optional[List[Any]]:
        """Handle file uploads for a step."""
        if not step.input_files:
            return None
            
        uploaded_files = []
        for file_path in step.input_files:
            mime_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else None
            uploaded_file = self.processor.upload_file(str(file_path), mime_type=mime_type)
            uploaded_files.append(uploaded_file)
        
        self.processor.wait_for_files_active(uploaded_files)
        return uploaded_files
    
    def _get_last_chunk(self, text: str, chunk_size: int = 200) -> str:
        """Get the last meaningful chunk of text (complete sentence or JSON structure)."""
        if not text:
            return ""        
        text_chunk = text[-chunk_size:]
        return text_chunk

    def _find_overlap(self, original: str, new_text: str, min_overlap: int = 20) -> int:
        """Enhanced overlap detection between original and new text."""
        if not original or not new_text:
            return 0
            
        original_end = original[-500:]  # Look at last 500 chars of original
        
        # Try different overlap sizes, from largest to smallest
        for i in range(len(original_end), min_overlap - 1, -1):
            if original_end[-i:].lower() == new_text[:i].lower():  # Case-insensitive comparison
                return i
                    
        return 0

    def _handle_json_continuation(self, current_content: str, new_content: str) -> str:
        """Handle continuation of JSON content by intelligently combining old and new."""
        # First try to combine as proper JSON
        combined = self._combine_json_content(current_content, new_content)
        if combined != new_content:
            return combined
            
        # If JSON combination fails, try to find and remove overlap
        overlap = self._find_overlap(current_content, new_content)
        if overlap > 0:
            return current_content + new_content[overlap:]
        
        return current_content + new_content

    def _handle_text_continuation(self, current_content: str, new_content: str) -> str:
        """Handle continuation of text content by removing overlaps."""
        overlap = self._find_overlap(current_content, new_content)
        if overlap > 0:
            return current_content + new_content[overlap:]
        
        return current_content + new_content
