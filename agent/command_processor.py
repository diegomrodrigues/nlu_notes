import re
from pathlib import Path
from typing import Dict, List, Any
import yaml
from .chain import TaskChain, ChainStep
from .processor import TaskProcessor

class CommandParser:
    """Parses command directives from text content."""
    
    @staticmethod
    def find_commands(content: str) -> List[Dict[str, Any]]:
        commands = []
        pattern = r"<!--\s*(\w+)\((.*?)\)\s*-->(?!\s*END)"
        
        for match in re.finditer(pattern, content):
            command_name = match.group(1)
            args_str = match.group(2).strip()
            
            # Parse arguments safely
            args = []
            if args_str:
                try:
                    args = [arg.strip().strip('"\'') for arg in args_str.split(",")]
                except Exception as e:
                    continue
            
            commands.append({
                "name": command_name,
                "args": args,
                "start": match.start(),
                "end": match.end()
            })
        
        return commands

class CommandLoader:
    """Loads and validates command configurations from YAML files."""
    
    def __init__(self, commands_dir: Path = Path("agent/commands")):
        self.commands_dir = commands_dir
        self.command_cache: Dict[str, Dict] = {}
        
        if not self.commands_dir.exists():
            raise ValueError(f"Commands directory not found: {commands_dir}")

    def load_command(self, command_name: str) -> Dict:
        if command_name in self.command_cache:
            return self.command_cache[command_name]
            
        cmd_file = self.commands_dir / f"{command_name}.yaml"
        if not cmd_file.exists():
            raise ValueError(f"Command not found: {command_name}")
            
        with open(cmd_file, "r") as f:
            config = yaml.safe_load(f)
            
        self._validate_config(config)
        self.command_cache[command_name] = config
        return config

    def _validate_config(self, config: Dict):
        required_keys = {"chain_steps"}
        if not required_keys.issubset(config.keys()):
            missing = required_keys - config.keys()
            raise ValueError(f"Invalid command config - missing keys: {missing}")

class CommandProcessor:
    """Processes files and executes commands through task chains."""
    
    def __init__(self, processor: TaskProcessor, tasks_config: Dict):
        self.parser = CommandParser()
        self.loader = CommandLoader()
        self.processor = processor
        self.tasks_config = tasks_config

    def process_directory(self, base_dir: Path):
        """Process all markdown files recursively"""        
        for file_path in base_dir.rglob("*.md"):
            self.process_file(file_path)

    def process_file(self, file_path: Path):
        """Process a single file with commands"""
        content = file_path.read_text(encoding="utf-8")
        commands = self.parser.find_commands(content)
        
        if not commands:
            return

        print(f"Processing {len(commands)} commands in {file_path.name}")
        
        for cmd in commands:
            config = self.loader.load_command(cmd["name"])
            chain = self._create_chain(config)
            processed_content = chain.run(content)
            
            if processed_content:
                content = processed_content
                
        # Save modified content
        file_path.write_text(content, encoding="utf-8")

    def _create_chain(self, config: Dict) -> TaskChain:
        """Create task chain from command config"""
        chain_steps = []
        
        for step_config in config["chain_steps"]:
            step = ChainStep(
                name=step_config["name"],
                tasks=step_config["tasks"],
                input_files=step_config.get("input_files"),
                expect_json=step_config.get("expect_json", False),
                extract_json=step_config.get("extract_json", False),
                stop_at=step_config.get("stop_at"),
                max_iterations=step_config.get("max_iterations", 3)
            )
            chain_steps.append(step)
            
        return TaskChain(
            self.processor,
            self.tasks_config,
            chain_steps,
            debug=False
        ) 