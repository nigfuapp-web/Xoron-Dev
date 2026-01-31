"""Document handling dataset generator for training document token understanding."""

import json
import os
import random
from typing import Dict, List, Any
from dataclasses import dataclass

from config.special_tokens import SPECIAL_TOKENS


@dataclass
class DocumentExample:
    """A single document handling example."""
    text: str  # Fully formatted text with tokens
    type: str  # document, multi_document, etc.
    file_type: str  # txt, md, json, etc.
    task: str  # summarize, explain, convert, etc.


class DocumentGenerator:
    """
    Generates synthetic document handling examples.
    
    Produces diverse examples for:
    - Plain text documents (.txt)
    - Markdown documents (.md)
    - JSON files (.json)
    - YAML/config files (.yaml, .toml, .ini)
    - XML files (.xml)
    - CSV data (.csv)
    - Log files (.log)
    - Multi-document tasks
    """
    
    def __init__(self, tokens: Dict[str, str] = None, seed: int = 42):
        self.tokens = tokens or SPECIAL_TOKENS
        self.rng = random.Random(seed)
        self.t = self.tokens
    
    def _wrap_sequence(self, text: str) -> str:
        """Wrap text with BOS/EOS tokens."""
        return f"{self.t['bos']}{text}{self.t['eos']}"
    
    def _wrap_file(self, content: str, file_type: str, filename: str = None) -> str:
        """Wrap content with file type tokens."""
        start_key = f"file_{file_type}"
        end_key = f"file_{file_type}_end"
        
        parts = []
        if filename:
            parts.append(f"{self.t['filename_start']}{filename}{self.t['filename_end']}")
        
        if start_key in self.t and end_key in self.t:
            parts.append(f"{self.t[start_key]}\n{content}\n{self.t[end_key]}")
        else:
            parts.append(f"{self.t['doc_start']}\n{content}\n{self.t['doc_end']}")
        
        return "\n".join(parts)
    
    def _format_qa(self, user_content: str, assistant_content: str) -> str:
        """Format as Q&A conversation."""
        return (
            f"{self.t['user_start']}\n{user_content}\n{self.t['user_end']}\n"
            f"{self.t['assistant_start']}\n{assistant_content}\n{self.t['assistant_end']}"
        )
    
    # ==================== TEXT DOCUMENT GENERATORS ====================
    
    def generate_txt_summarize(self) -> DocumentExample:
        """Generate a text summarization task."""
        topics = [
            ("climate change", "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors like volcanic eruptions can cause climate shifts, human activities have been the main driver since the 1800s, primarily due to burning fossil fuels like coal, oil, and gas. This releases greenhouse gases that trap heat in Earth's atmosphere, leading to global warming. Effects include rising sea levels, more frequent extreme weather events, and disruptions to ecosystems worldwide."),
            ("artificial intelligence", "Artificial intelligence (AI) is the simulation of human intelligence by machines. It encompasses machine learning, where systems learn from data, and deep learning, which uses neural networks. AI applications range from virtual assistants and recommendation systems to autonomous vehicles and medical diagnosis. While AI offers tremendous benefits, it also raises concerns about job displacement, privacy, and the need for ethical guidelines."),
            ("renewable energy", "Renewable energy comes from sources that naturally replenish, such as sunlight, wind, rain, tides, and geothermal heat. Solar panels convert sunlight into electricity, while wind turbines harness wind power. These clean energy sources produce little to no greenhouse gas emissions, making them crucial for combating climate change. The cost of renewable energy has dropped significantly, making it increasingly competitive with fossil fuels."),
            ("space exploration", "Space exploration involves investigating outer space using astronomy and space technology. Since the first satellite Sputnik in 1957, humanity has sent probes to every planet, landed on the Moon, and established the International Space Station. Current goals include returning humans to the Moon, sending astronauts to Mars, and searching for extraterrestrial life. Private companies like SpaceX are now playing major roles in space exploration."),
            ("quantum computing", "Quantum computing harnesses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously. This enables them to solve certain problems exponentially faster than classical computers. Applications include cryptography, drug discovery, and optimization problems."),
        ]
        
        topic, content = self.rng.choice(topics)
        filename = f"{topic.replace(' ', '_')}_overview.txt"
        
        doc_wrapped = self._wrap_file(content, "txt", filename)
        
        summaries = {
            "climate change": "Climate change is primarily caused by human activities releasing greenhouse gases, leading to global warming and environmental disruptions.",
            "artificial intelligence": "AI simulates human intelligence through machine learning and neural networks, with applications ranging from assistants to autonomous vehicles.",
            "renewable energy": "Renewable energy from sources like solar and wind produces minimal emissions and is becoming cost-competitive with fossil fuels.",
            "space exploration": "Space exploration has progressed from early satellites to plans for Mars missions, with private companies now playing major roles.",
            "quantum computing": "Quantum computers use qubits and quantum phenomena to solve certain problems exponentially faster than classical computers.",
        }
        
        user_content = f"Please summarize the following document:\n\n{doc_wrapped}"
        assistant_content = summaries[topic]
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="txt",
            task="summarize"
        )
    
    def generate_txt_questions(self) -> DocumentExample:
        """Generate a Q&A task about a text document."""
        docs = [
            {
                "content": "The Python programming language was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python is dynamically typed and garbage-collected.",
                "filename": "python_history.txt",
                "question": "Who created Python and when was it first released?",
                "answer": "Python was created by Guido van Rossum and was first released in 1991."
            },
            {
                "content": "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, and other materials. Built along the historical northern borders of China, it was constructed to protect against various nomadic groups. The wall spans approximately 13,171 miles (21,196 kilometers) and was built over many centuries, with the most well-known sections built during the Ming Dynasty (1368-1644).",
                "filename": "great_wall.txt",
                "question": "How long is the Great Wall of China?",
                "answer": "The Great Wall of China spans approximately 13,171 miles (21,196 kilometers)."
            },
            {
                "content": "Photosynthesis is the process by which plants convert light energy into chemical energy. Using sunlight, water, and carbon dioxide, plants produce glucose and oxygen. The process occurs primarily in the leaves, where chlorophyll absorbs light energy. Photosynthesis is essential for life on Earth as it produces oxygen and forms the base of most food chains.",
                "filename": "photosynthesis.txt",
                "question": "What are the inputs and outputs of photosynthesis?",
                "answer": "The inputs of photosynthesis are sunlight, water, and carbon dioxide. The outputs are glucose and oxygen."
            },
        ]
        
        doc = self.rng.choice(docs)
        doc_wrapped = self._wrap_file(doc["content"], "txt", doc["filename"])
        
        user_content = f"Based on this document:\n\n{doc_wrapped}\n\n{doc['question']}"
        assistant_content = doc["answer"]
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="txt",
            task="question_answering"
        )
    
    # ==================== MARKDOWN GENERATORS ====================
    
    def generate_md_explain(self) -> DocumentExample:
        """Generate a markdown explanation task."""
        md_docs = [
            {
                "content": """# API Documentation

## Authentication

All API requests require authentication using an API key.

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### GET /users
Returns a list of all users.

### POST /users
Creates a new user.

**Request Body:**
- `name` (string, required): User's name
- `email` (string, required): User's email

### DELETE /users/{id}
Deletes a user by ID.""",
                "filename": "api_docs.md",
                "question": "Explain how to authenticate with this API and list the available endpoints.",
                "answer": "To authenticate with this API, you need to include an Authorization header with a Bearer token containing your API key. The available endpoints are: GET /users (retrieves all users), POST /users (creates a new user with name and email), and DELETE /users/{id} (deletes a specific user by their ID)."
            },
            {
                "content": """# Project Setup Guide

## Prerequisites
- Node.js v16+
- npm or yarn
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/example/project.git
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create environment file:
   ```bash
   cp .env.example .env
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

## Configuration

Edit `.env` to configure:
- `DATABASE_URL`: Your database connection string
- `API_KEY`: Your API key""",
                "filename": "SETUP.md",
                "question": "What are the steps to set up this project?",
                "answer": "To set up this project: 1) Ensure you have Node.js v16+, npm/yarn, and Git installed. 2) Clone the repository using git clone. 3) Run npm install to install dependencies. 4) Copy .env.example to .env and configure your DATABASE_URL and API_KEY. 5) Start the development server with npm run dev."
            },
        ]
        
        doc = self.rng.choice(md_docs)
        doc_wrapped = self._wrap_file(doc["content"], "md", doc["filename"])
        
        user_content = f"Read this markdown document:\n\n{doc_wrapped}\n\n{doc['question']}"
        assistant_content = doc["answer"]
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="md",
            task="explain"
        )
    
    # ==================== JSON GENERATORS ====================
    
    def generate_json_parse(self) -> DocumentExample:
        """Generate a JSON parsing/understanding task."""
        json_docs = [
            {
                "content": json.dumps({
                    "user": {
                        "id": 12345,
                        "name": "John Doe",
                        "email": "john@example.com",
                        "roles": ["admin", "editor"],
                        "settings": {
                            "theme": "dark",
                            "notifications": True
                        }
                    }
                }, indent=2),
                "filename": "user_data.json",
                "question": "What roles does this user have and what theme are they using?",
                "answer": "The user has two roles: 'admin' and 'editor'. They are using the 'dark' theme."
            },
            {
                "content": json.dumps({
                    "products": [
                        {"id": 1, "name": "Laptop", "price": 999.99, "in_stock": True},
                        {"id": 2, "name": "Mouse", "price": 29.99, "in_stock": True},
                        {"id": 3, "name": "Keyboard", "price": 79.99, "in_stock": False}
                    ],
                    "total_products": 3
                }, indent=2),
                "filename": "inventory.json",
                "question": "Which products are in stock and what is the most expensive item?",
                "answer": "The products in stock are the Laptop ($999.99) and the Mouse ($29.99). The most expensive item is the Laptop at $999.99."
            },
            {
                "content": json.dumps({
                    "config": {
                        "database": {
                            "host": "localhost",
                            "port": 5432,
                            "name": "myapp_db"
                        },
                        "cache": {
                            "enabled": True,
                            "ttl": 3600
                        },
                        "logging": {
                            "level": "info",
                            "file": "/var/log/app.log"
                        }
                    }
                }, indent=2),
                "filename": "config.json",
                "question": "What database port is configured and is caching enabled?",
                "answer": "The database is configured to use port 5432. Yes, caching is enabled with a TTL (time-to-live) of 3600 seconds."
            },
        ]
        
        doc = self.rng.choice(json_docs)
        doc_wrapped = self._wrap_file(doc["content"], "json", doc["filename"])
        
        user_content = f"Analyze this JSON file:\n\n{doc_wrapped}\n\n{doc['question']}"
        assistant_content = doc["answer"]
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="json",
            task="parse"
        )
    
    def generate_json_modify(self) -> DocumentExample:
        """Generate a JSON modification task."""
        original = {
            "name": "MyApp",
            "version": "1.0.0",
            "dependencies": {
                "express": "^4.18.0",
                "lodash": "^4.17.0"
            }
        }
        
        modified = {
            "name": "MyApp",
            "version": "1.1.0",
            "dependencies": {
                "express": "^4.18.0",
                "lodash": "^4.17.0",
                "axios": "^1.4.0"
            }
        }
        
        doc_wrapped = self._wrap_file(json.dumps(original, indent=2), "json", "package.json")
        
        user_content = f"Given this package.json:\n\n{doc_wrapped}\n\nUpdate the version to 1.1.0 and add axios ^1.4.0 as a dependency."
        assistant_content = f"Here's the updated package.json:\n\n{self._wrap_file(json.dumps(modified, indent=2), 'json', 'package.json')}"
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="json",
            task="modify"
        )
    
    # ==================== YAML/CONFIG GENERATORS ====================
    
    def generate_yaml_explain(self) -> DocumentExample:
        """Generate a YAML configuration explanation task."""
        yaml_content = """# Docker Compose Configuration
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgres://db:5432/myapp
    depends_on:
      - db
      - redis

  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_PASSWORD=secret

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:"""
        
        doc_wrapped = self._wrap_file(yaml_content, "yaml", "docker-compose.yml")
        
        user_content = f"Explain this Docker Compose configuration:\n\n{doc_wrapped}"
        assistant_content = "This Docker Compose file defines three services: 1) 'web' - the main application that builds from the current directory, exposes port 3000, runs in production mode, and depends on the database and Redis. 2) 'db' - a PostgreSQL 14 database with persistent storage via a named volume. 3) 'redis' - a Redis cache using the Alpine image on port 6379. The web service connects to PostgreSQL via the DATABASE_URL environment variable."
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="yaml",
            task="explain"
        )
    
    # ==================== CSV GENERATORS ====================
    
    def generate_csv_analyze(self) -> DocumentExample:
        """Generate a CSV analysis task."""
        csv_content = """name,department,salary,years_employed
Alice,Engineering,85000,5
Bob,Marketing,65000,3
Charlie,Engineering,92000,7
Diana,Sales,72000,4
Eve,Engineering,78000,2
Frank,Marketing,68000,6"""
        
        doc_wrapped = self._wrap_file(csv_content, "csv", "employees.csv")
        
        questions = [
            ("What is the average salary in the Engineering department?", 
             "The Engineering department has three employees: Alice ($85,000), Charlie ($92,000), and Eve ($78,000). The average salary is ($85,000 + $92,000 + $78,000) / 3 = $85,000."),
            ("Who has been employed the longest?",
             "Charlie has been employed the longest at 7 years, working in the Engineering department with a salary of $92,000."),
            ("How many employees are in each department?",
             "Engineering has 3 employees (Alice, Charlie, Eve), Marketing has 2 employees (Bob, Frank), and Sales has 1 employee (Diana)."),
        ]
        
        question, answer = self.rng.choice(questions)
        
        user_content = f"Analyze this CSV data:\n\n{doc_wrapped}\n\n{question}"
        assistant_content = answer
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="csv",
            task="analyze"
        )
    
    # ==================== LOG FILE GENERATORS ====================
    
    def generate_log_analyze(self) -> DocumentExample:
        """Generate a log file analysis task."""
        log_content = """2024-01-15 10:23:45 INFO  Application started successfully
2024-01-15 10:23:46 INFO  Connected to database
2024-01-15 10:24:12 WARN  High memory usage detected: 85%
2024-01-15 10:25:33 ERROR Failed to process request: Connection timeout
2024-01-15 10:25:34 INFO  Retrying connection...
2024-01-15 10:25:36 INFO  Connection restored
2024-01-15 10:30:00 ERROR Database query failed: Table 'users' not found
2024-01-15 10:30:01 WARN  Falling back to cache"""
        
        doc_wrapped = self._wrap_file(log_content, "log", "app.log")
        
        user_content = f"Analyze this log file and identify any issues:\n\n{doc_wrapped}"
        assistant_content = "The log shows two main issues: 1) A connection timeout error at 10:25:33, which was resolved after a retry at 10:25:36. 2) A database error at 10:30:00 where the 'users' table was not found, causing the system to fall back to cache. There was also a warning about high memory usage (85%) at 10:24:12 that should be monitored."
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type="log",
            task="analyze"
        )
    
    # ==================== MULTI-DOCUMENT GENERATORS ====================
    
    def generate_multi_doc_compare(self) -> DocumentExample:
        """Generate a multi-document comparison task."""
        doc1 = {
            "content": json.dumps({"version": "1.0", "features": ["auth", "api"], "database": "mysql"}, indent=2),
            "filename": "config_v1.json",
            "file_type": "json"
        }
        
        doc2 = {
            "content": json.dumps({"version": "2.0", "features": ["auth", "api", "websocket"], "database": "postgres"}, indent=2),
            "filename": "config_v2.json", 
            "file_type": "json"
        }
        
        doc1_wrapped = self._wrap_file(doc1["content"], doc1["file_type"], doc1["filename"])
        doc2_wrapped = self._wrap_file(doc2["content"], doc2["file_type"], doc2["filename"])
        
        user_content = f"Compare these two configuration files:\n\n{doc1_wrapped}\n\n{self.t['separator']}\n\n{doc2_wrapped}"
        assistant_content = "The main differences between v1 and v2 are: 1) Version updated from 1.0 to 2.0. 2) A new 'websocket' feature was added in v2. 3) The database was changed from MySQL to PostgreSQL."
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="multi_document",
            file_type="json",
            task="compare"
        )
    
    # ==================== CONVERSION GENERATORS ====================
    
    def generate_format_conversion(self) -> DocumentExample:
        """Generate a format conversion task."""
        conversions = [
            {
                "from_type": "json",
                "to_type": "yaml",
                "from_content": json.dumps({"server": {"host": "localhost", "port": 8080}, "debug": True}, indent=2),
                "to_content": "server:\n  host: localhost\n  port: 8080\ndebug: true",
                "from_file": "config.json",
                "to_file": "config.yaml"
            },
            {
                "from_type": "csv",
                "to_type": "json",
                "from_content": "name,age,city\nAlice,30,NYC\nBob,25,LA",
                "to_content": json.dumps([
                    {"name": "Alice", "age": 30, "city": "NYC"},
                    {"name": "Bob", "age": 25, "city": "LA"}
                ], indent=2),
                "from_file": "data.csv",
                "to_file": "data.json"
            },
        ]
        
        conv = self.rng.choice(conversions)
        from_wrapped = self._wrap_file(conv["from_content"], conv["from_type"], conv["from_file"])
        to_wrapped = self._wrap_file(conv["to_content"], conv["to_type"], conv["to_file"])
        
        user_content = f"Convert this {conv['from_type'].upper()} to {conv['to_type'].upper()}:\n\n{from_wrapped}"
        assistant_content = f"Here's the converted {conv['to_type'].upper()}:\n\n{to_wrapped}"
        
        text = self._wrap_sequence(self._format_qa(user_content, assistant_content))
        
        return DocumentExample(
            text=text,
            type="document",
            file_type=conv["from_type"],
            task="convert"
        )
    
    # ==================== MAIN GENERATION ====================
    
    def generate_example(self) -> DocumentExample:
        """Generate a random document example."""
        generators = [
            self.generate_txt_summarize,
            self.generate_txt_questions,
            self.generate_md_explain,
            self.generate_json_parse,
            self.generate_json_modify,
            self.generate_yaml_explain,
            self.generate_csv_analyze,
            self.generate_log_analyze,
            self.generate_multi_doc_compare,
            self.generate_format_conversion,
        ]
        
        generator = self.rng.choice(generators)
        return generator()
    
    def generate_dataset(self, num_examples: int = 1000, output_path: str = None) -> List[Dict]:
        """Generate a dataset of document examples."""
        examples = []
        
        for i in range(num_examples):
            self.rng.seed(42 + i)  # Reproducible but varied
            example = self.generate_example()
            examples.append({
                "text": example.text,
                "type": example.type,
                "file_type": example.file_type,
                "task": example.task
            })
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                for ex in examples:
                    f.write(json.dumps(ex) + '\n')
            print(f"✅ Generated {num_examples} document examples to {output_path}")
        
        return examples


def generate_document_dataset(num_examples: int = 1000, output_dir: str = "synth/data"):
    """Convenience function to generate document dataset."""
    generator = DocumentGenerator()
    output_path = os.path.join(output_dir, "document_dataset.jsonl")
    return generator.generate_dataset(num_examples, output_path)


# Removed standalone execution - use as module instead
# if __name__ == "__main__":
    # # Generate dataset when run directly
    # generate_document_dataset(1000)
