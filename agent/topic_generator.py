from pathlib import Path
from typing import List, Optional, Dict
from .chain import TaskChain, ChainStep
from .processor import TaskProcessor
import json
import re
from .utils import retry_on_error

class TopicGenerator:
    """Generates and manages hierarchical topic structures from PDF documents."""
    
    def __init__(self, processor: TaskProcessor, tasks_config: dict, debug: bool = False):
        self.processor = processor
        self.tasks_config = tasks_config
        self.debug = debug  # Store debug flag

    def generate(
        self,
        directory: Path,
        perspectives: Optional[List[str]] = None,
        num_topics: Optional[int] = None,
        jsons_per_perspective: int = 3,
        num_consolidation_steps: int = 1
    ) -> str:
        if self.debug:
            print("\n🔵 DEBUG MODE ACTIVATED 🔵")
            print(f"Initial Parameters:")
            print(f"- Directory: {directory}")
            print(f"- Perspectives: {perspectives or 'None'}")
            print(f"- Num Topics: {num_topics}")
            print(f"- JSONs per Perspective: {jsons_per_perspective}")
            print(f"- Consolidation Steps: {num_consolidation_steps}\n")
        
        self._validate_input(perspectives, num_topics)
        pdf_files = self._get_pdf_files(directory)
        
        # Phase 1: Generate initial topics
        perspective_results = self._generate_initial_topics(
            pdf_files=pdf_files,
            perspectives=perspectives,
            num_topics=num_topics or len(perspectives),
            jsons_per_perspective=jsons_per_perspective
        )

        if self.debug:
            print("\n🔵 DEBUG: Initial Topics Results")
            for i, result in enumerate(perspective_results):
                print(f"Perspective {i+1}:\n{result[:500]}...\n")  # Show first 500 chars

        # Phase 2: Merge and restructure
        final_topics = self._process_and_restructure(
            directory=directory,
            perspective_results=perspective_results,
            num_consolidation_steps=num_consolidation_steps
        )

        # Save final results
        self._save_topics(directory, final_topics)
        return final_topics

    def _generate_initial_topics(
        self,
        pdf_files: List[Path],
        perspectives: Optional[List[str]],
        num_topics: int,
        jsons_per_perspective: int
    ) -> List[str]:
        """Generate initial topic sets."""
        print("\n📚 Generating initial topic sets...")
        perspective_results = []
        
        for i in range(num_topics):
            perspective = perspectives[i] if perspectives else f"Perspective {i+1}"
            print(f"\n🔍 Processing perspective {i+1}/{num_topics}: {perspective}")
            
            # Generate multiple topic sets for this perspective
            topic_sets = self._generate_topic_sets(
                pdf_files=pdf_files,
                perspective=perspective,
                perspective_index=i,
                num_sets=jsons_per_perspective
            )
            
            # Merge sets for this perspective
            merged_perspective = self._merge_topic_sets(topic_sets, i)
            perspective_results.append(merged_perspective)
            print(f"✔️ Completed perspective {i+1}/{num_topics}")
            
        return perspective_results

    @retry_on_error(max_retries=3)
    def _generate_topic_sets(
        self,
        pdf_files: List[Path],
        perspective: str,
        perspective_index: int,
        num_sets: int
    ) -> List[str]:
        """Generate multiple topic sets for a single perspective."""
        if self.debug:
            print(f"\n🔍 DEBUG: Starting _generate_topic_sets")
            print(f"  - Perspective: {perspective}")
            print(f"  - Perspective Index: {perspective_index}")
            print(f"  - Number of sets: {num_sets}")
            print(f"  - PDF Files: {[f.name for f in pdf_files]}")
        
        topic_sets = []
        
        for set_index in range(num_sets):
            if self.debug:
                print(f"\n  - Generating set {set_index + 1}/{num_sets}")
            
            result = self._generate_single_set(
                pdf_files=pdf_files,
                perspective=perspective,
                perspective_index=perspective_index,
                set_index=set_index
            )
            
            if result:
                if self.debug:
                    print(f"    ✔️ Set {set_index + 1} generated successfully")
                    print(f"    - Content length: {len(result)}")
                topic_sets.append(result)
            else:
                if self.debug:
                    print(f"    ❌ Failed to generate set {set_index + 1}")
        
        if self.debug:
            print(f"\n  - Total sets generated: {len(topic_sets)}/{num_sets}")
        
        if not topic_sets:
            raise Exception(f"Failed to generate any valid topic sets for perspective {perspective_index+1}")
        return topic_sets

    def _generate_single_set(
        self,
        pdf_files: List[Path],
        perspective: str,
        perspective_index: int,
        set_index: int
    ) -> Optional[str]:
        """Generate a single set of topics using the task chain."""
        if self.debug:
            print(f"\n🔍 DEBUG: Starting _generate_single_set")
            print(f"  - Set Index: {set_index}")
            print(f"  - Creating chain for perspective {perspective_index + 1}")
        
        chain = TaskChain(self.processor, self.tasks_config, [
            ChainStep(
                name=f"Generate Topics Set {set_index+1} for Perspective {perspective_index+1}",
                tasks=["create_topics"],
                input_files=pdf_files,
                expect_json=True,
                stop_at="<!-- END -->",
                max_iterations=5
            )
        ], debug=self.debug)
        
        for attempt in range(3):
            if self.debug:
                print(f"  - Attempt {attempt + 1}/3")
            
            result = chain.run(perspective)
            if result:
                if self.debug:
                    print(f"    ✔️ Generation successful")
                    print(f"    - Result length: {len(result)}")
                return result
            
            if self.debug:
                print(f"    ⚠️ Attempt {attempt + 1} failed")
            print(f"⚠️ Retrying set {set_index+1} (attempt {attempt+1}/3)")
        return None

    def _validate_json(self, json_str: str) -> bool:
        """Validate JSON structure."""
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict) or 'topics' not in data:
                return False
            return True
        except json.JSONDecodeError:
            return False

    def _process_and_restructure(
        self,
        directory: Path,
        perspective_results: List[str],
        num_consolidation_steps: int
    ) -> str:
        """Merge all perspectives and restructure based on existing directory structure."""
        if self.debug:
            print("\n🔍 DEBUG: Starting _process_and_restructure")
            print(f"  - Number of perspectives: {len(perspective_results)}")
            print(f"  - Consolidation steps: {num_consolidation_steps}")
        
        print("\n🔄 Merging and restructuring topics...")
        
        # Merge all perspective results
        if self.debug:
            print("  - Starting perspective merge")
        merged_topics = self._merge_all_perspectives(perspective_results)
        
        if self.debug:
            print("  - Getting existing directories")
        existing_dirs = self._get_existing_directories(directory)
        
        if self.debug:
            print(f"  - Found {len(existing_dirs)} existing directories")
            print("  - Starting topic restructuring")
        
        result = self._restructure_topics(
            merged_topics=merged_topics,
            existing_dirs=existing_dirs,
            num_steps=num_consolidation_steps
        )
        
        if self.debug:
            print("  ✔️ Processing and restructuring complete")
            print(f"  - Final result length: {len(result)}")
        
        return result

    def _merge_all_perspectives(self, perspective_results: List[str]) -> str:
        """Merge all perspective results into a single topic structure."""
        parsed_results = [json.loads(result) for result in perspective_results]
        merged = self._merge_topic_contents(parsed_results)
        return json.dumps(merged, indent=2, ensure_ascii=False)

    def _merge_topic_sets(self, topic_sets: List[str], perspective_index: int) -> str:
        """Merge multiple topic sets for a single perspective."""
        parsed_sets = [json.loads(topic_set) for topic_set in topic_sets]
        merged = self._merge_topic_contents(parsed_sets)
        return json.dumps(merged, indent=2, ensure_ascii=False)

    def _restructure_topics(
        self,
        merged_topics: str,
        existing_dirs: List[str],
        num_steps: int
    ) -> str:
        """Restructure topics with JSON extraction."""
        context = json.dumps({"existing_directories": existing_dirs})
        
        chain = TaskChain(self.processor, self.tasks_config, [
            ChainStep(
                name=f"Consolidate Subtopics step {i}",
                tasks=["consolidate_subtopics"],
                extract_json=True,
                max_iterations=5,
                use_previous_result=True,
                additional_context=context,
                stop_at="<!-- END -->"
            )
            for i in range(1, num_steps + 1)
        ], debug=self.debug)
        
        result = chain.run(merged_topics)
        if not result:
            raise Exception("Failed to restructure topics")
        return result

    def _validate_input(self, perspectives: Optional[List[str]], num_topics: Optional[int]):
        """Validate input parameters."""
        if perspectives is None and num_topics is None:
            raise ValueError("You should provide either perspectives or num_topics")

    def _get_pdf_files(self, directory: Path) -> List[Path]:
        """Get all PDF files in the directory."""
        return list(directory.glob("*.pdf"))

    def _get_existing_directories(self, directory: Path) -> List[str]:
        """Get existing directory names without numbering."""
        return [
            re.sub(r'^\d+\.\s*', '', d.name)
            for d in directory.iterdir() 
            if d.is_dir()
        ]

    def _merge_topic_contents(self, json_list: List[dict]) -> dict:
        """Merge multiple topic JSONs by combining topics and their sub-topics."""
        if self.debug:
            print("\n🔍 DEBUG: Starting _merge_topic_contents")
            print(f"  - Number of JSONs to merge: {len(json_list)}")
        
        topic_map = {}
        
        # Process each JSON document
        for i, json_doc in enumerate(json_list):
            if self.debug:
                print(f"  - Processing JSON {i + 1}/{len(json_list)}")
                print(f"    - Topics in document: {len(json_doc.get('topics', []))}")
            
            for topic_item in json_doc.get('topics', []):
                topic = topic_item['topic']
                sub_topics = set(topic_item.get('sub_topics', []))
                
                if topic in topic_map:
                    if self.debug:
                        print(f"    - Merging subtopics for existing topic: {topic[:50]}...")
                    topic_map[topic].update(sub_topics)
                else:
                    if self.debug:
                        print(f"    - Creating new topic: {topic[:50]}...")
                    topic_map[topic] = sub_topics
        
        if self.debug:
            print(f"  - Final number of unique topics: {len(topic_map)}")
        
        # Convert back to the required format
        merged_result = {
            "topics": [
                {
                    "topic": topic,
                    "sub_topics": sorted(list(sub_topics))
                }
                for topic, sub_topics in topic_map.items()
            ]
        }
        
        return merged_result

    def _save_topics(self, directory: Path, content: str) -> None:
        """Save the generated topics to a file."""
        output_file = directory / "topics.json"
        output_file.write_text(content, encoding='utf-8')
        print(f"✔️ Topics generated and saved to: {output_file}") 