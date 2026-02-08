# 🧪 Synth Module Documentation

The Synth module generates high-quality synthetic training data for specialized capabilities like chain-of-thought reasoning, anti-hallucination, and agentic behaviors.

## 📁 File Structure

```
synth/
├── __init__.py
├── generator.py                    # Chain-of-thought generator
├── anti_hallucination_generator.py # Anti-hallucination data
├── agentic_dataset_generator.py    # Agentic coding data
├── document_generator.py           # Document handling data
├── system_admin_generator.py       # System administration data
├── unique_generator.py             # Unique/diverse data
├── quality_utils.py                # Quality filtering
├── templates.py                    # Prompt templates
├── generate_dataset.py             # Main generation script
└── data/                           # Generated datasets
    ├── cot_dataset.jsonl
    ├── idk_dataset.jsonl
    ├── fact_check_dataset.jsonl
    └── ...
```

---

## 🧠 Chain-of-Thought Generator

### Overview

Generates diverse reasoning examples with explicit thinking steps.

### Example Output

```json
{
  "question": "Calculate: 15 × 23 + 47",
  "thinking": "<|think|>\nLet me break this down step by step.\n\n<|plan|>\n1. First multiply 15 × 23\n2. Then add 47 to the result\n<|/plan|>\n\n<|step|>15 × 23 = 15 × 20 + 15 × 3 = 300 + 45 = 345<|/step|>\n<|step|>345 + 47 = 392<|/step|>\n\n<|verify|>Let me check: 15 × 23 = 345 ✓, 345 + 47 = 392 ✓<|/verify|>\n<|/think|>",
  "answer": "The answer is 392.",
  "category": "arithmetic",
  "difficulty": "medium"
}
```

### Generator Class

```python
class ChainOfThoughtGenerator:
    """Generates high-quality chain-of-thought reasoning examples."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
    def generate_arithmetic(self, difficulty: str = 'medium') -> ReasoningExample:
        """Generate arithmetic reasoning problem."""
        if difficulty == 'easy':
            a, b = self.rng.randint(1, 20), self.rng.randint(1, 20)
            op = self.rng.choice(['+', '-', '*'])
        elif difficulty == 'medium':
            a, b = self.rng.randint(10, 100), self.rng.randint(10, 100)
            op = self.rng.choice(['+', '-', '*', '/'])
        else:
            a, b, c = self.rng.randint(10, 50), self.rng.randint(10, 50), self.rng.randint(1, 20)
            # Multi-step problem
        
        # Generate thinking process
        thinking = self._generate_thinking(a, b, op)
        answer = self._compute_answer(a, b, op)
        
        return ReasoningExample(
            question=f"Calculate: {a} {op} {b}",
            thinking=thinking,
            answer=f"The answer is {answer}.",
            category='arithmetic',
            difficulty=difficulty,
        )
    
    def generate_word_problem(self) -> ReasoningExample:
        """Generate word problem with reasoning."""
        # Select template
        template = self.rng.choice(WORD_PROBLEM_TEMPLATES)
        
        # Fill in values
        name = self.rng.choice(NAMES)
        item, item_singular, currency = self.rng.choice(SHOPPING_ITEMS)
        quantity = self.rng.randint(2, 10)
        price = self.rng.randint(1, 20)
        
        problem = template.format(
            name=name,
            item=item,
            quantity=quantity,
            price=price,
            currency=currency,
        )
        
        # Generate solution
        thinking = self._generate_word_problem_thinking(...)
        answer = quantity * price
        
        return ReasoningExample(
            question=problem,
            thinking=thinking,
            answer=f"{name} spent {currency}{answer} in total.",
            category='word_problem',
            difficulty='medium',
        )
    
    def generate_logic(self) -> ReasoningExample:
        """Generate logical reasoning problem."""
        # ...
    
    def generate_code_debug(self) -> ReasoningExample:
        """Generate code debugging problem."""
        # ...
```

### Categories

| Category | Description | Example |
|----------|-------------|---------|
| Arithmetic | Math calculations | "Calculate: 15 × 23" |
| Word Problems | Story-based math | "Alice bought 5 apples..." |
| Logic | Logical deduction | "If A implies B..." |
| Code Debug | Find and fix bugs | "Fix this code..." |
| Algebra | Equation solving | "Solve for x: 2x + 5 = 15" |

---

## 🚫 Anti-Hallucination Generator

### Overview

Generates training data to teach the model to:
- Say "I don't know" when appropriate
- Express uncertainty
- Provide citations
- Self-correct

### Dataset Types

#### 1. IDK (I Don't Know)

```json
{
  "question": "What will the stock price of Apple be on January 1, 2030?",
  "response": "<|uncertain|>I cannot predict future stock prices. Stock markets are influenced by countless unpredictable factors including economic conditions, company performance, and global events. <|idk|>I don't have the ability to forecast specific future prices.<|/idk|><|/uncertain|>"
}
```

#### 2. Knowledge Cutoff

```json
{
  "question": "Who won the 2028 Olympics?",
  "response": "<|uncertain|>My knowledge has a cutoff date, and I don't have information about events that occurred after my training. <|needs_verify|>You should check current news sources for information about the 2028 Olympics.<|/needs_verify|><|/uncertain|>"
}
```

#### 3. Fact Check

```json
{
  "question": "Is it true that the Great Wall of China is visible from space?",
  "response": "<|think|>Let me verify this common claim.\n<|verify|>This is actually a popular misconception. The Great Wall, while long, is only about 15-30 feet wide, making it too narrow to be visible from space with the naked eye.<|/verify|>\n<|/think|>\n\nNo, this is a common myth. The Great Wall of China is not visible from space with the naked eye. While it's very long, it's too narrow to be distinguished from orbit."
}
```

#### 4. Uncertainty Expression

```json
{
  "question": "What causes dark matter?",
  "response": "<|uncertain|>The nature and cause of dark matter is one of the biggest unsolved problems in physics. <|confidence level=\"low\"|>Scientists have proposed several theories including WIMPs, axions, and modified gravity, but none have been definitively proven.<|/confidence|><|/uncertain|>"
}
```

#### 5. Citation

```json
{
  "question": "What is the speed of light?",
  "response": "The speed of light in a vacuum is approximately 299,792,458 meters per second. <|citation|>This value is defined exactly by the International Bureau of Weights and Measures (BIPM) and is used to define the meter.<|/citation|>"
}
```

### Generator Class

```python
class AntiHallucinationGenerator:
    """Generates anti-hallucination training data."""
    
    def generate_idk_example(self) -> Dict:
        """Generate 'I don't know' example."""
        category = self.rng.choice([
            'future_prediction',
            'personal_opinion',
            'real_time_data',
            'private_information',
            'beyond_expertise',
        ])
        
        question = self._generate_unanswerable_question(category)
        response = self._generate_idk_response(category)
        
        return {
            'question': question,
            'response': response,
            'category': category,
            'type': 'idk',
        }
    
    def generate_uncertainty_example(self) -> Dict:
        """Generate uncertainty expression example."""
        # Topics with inherent uncertainty
        topic = self.rng.choice([
            'scientific_hypothesis',
            'historical_interpretation',
            'medical_advice',
            'economic_prediction',
        ])
        
        question = self._generate_uncertain_question(topic)
        response = self._generate_uncertain_response(topic)
        
        return {
            'question': question,
            'response': response,
            'type': 'uncertainty',
        }
```

---

## 🤖 Agentic Dataset Generator

### Overview

Generates training data for agentic coding capabilities.

### Dataset Types

#### 1. Fill-in-the-Middle (FIM)

```json
{
  "prefix": "def calculate_average(numbers):\n    if not numbers:\n        return 0\n    ",
  "middle": "total = sum(numbers)\n    count = len(numbers)\n    return total / count",
  "suffix": "\n\n# Test\nprint(calculate_average([1, 2, 3, 4, 5]))"
}
```

#### 2. Git Commits

```json
{
  "before": "def greet(name):\n    print('Hello ' + name)",
  "after": "def greet(name: str) -> None:\n    \"\"\"Greet a person by name.\"\"\"\n    print(f'Hello {name}')",
  "commit_message": "Add type hints and docstring to greet function"
}
```

#### 3. Diffs

```json
{
  "diff": "@@ -1,3 +1,5 @@\n def greet(name):\n-    print('Hello ' + name)\n+    \"\"\"Greet a person by name.\"\"\"\n+    print(f'Hello {name}')",
  "description": "Added docstring and converted to f-string"
}
```

#### 4. Code Execution

```json
{
  "code": "x = [1, 2, 3]\nprint(x[5])",
  "output": "<|exec_error|>IndexError: list index out of range<|/exec_error|>",
  "explanation": "The list has only 3 elements (indices 0, 1, 2), but we tried to access index 5."
}
```

#### 5. Shell Commands

```json
{
  "task": "Find all Python files in the current directory",
  "command": "find . -name '*.py' -type f",
  "output": "./main.py\n./utils/helpers.py\n./tests/test_main.py"
}
```

### Generator Class

```python
class AgenticDatasetGenerator:
    """Generates agentic coding training data."""
    
    def generate_fim_example(self) -> Dict:
        """Generate fill-in-the-middle example."""
        # Select code template
        template = self.rng.choice(CODE_TEMPLATES)
        
        # Split into prefix, middle, suffix
        lines = template.split('\n')
        split_point = self.rng.randint(1, len(lines) - 2)
        
        prefix = '\n'.join(lines[:split_point])
        middle = lines[split_point]
        suffix = '\n'.join(lines[split_point + 1:])
        
        return {
            'prefix': prefix,
            'middle': middle,
            'suffix': suffix,
            'type': 'fim',
        }
    
    def generate_commit_example(self) -> Dict:
        """Generate git commit example."""
        # Select transformation type
        transform = self.rng.choice([
            'add_docstring',
            'add_type_hints',
            'fix_bug',
            'refactor',
            'add_error_handling',
        ])
        
        before, after = self._apply_transformation(transform)
        message = self._generate_commit_message(transform)
        
        return {
            'before': before,
            'after': after,
            'commit_message': message,
            'type': 'commit',
        }
```

---

## 📄 Document Generator

### Overview

Generates training data for document handling and file operations.

### Dataset Types

| Type | Description |
|------|-------------|
| File Read | Reading file contents |
| File Write | Creating new files |
| File Edit | Modifying existing files |
| File Search | Finding files by pattern |
| Document Parse | Extracting information |

---

## 🖥️ System Admin Generator

### Overview

Generates training data for system administration tasks.

### Dataset Types

| Type | Description |
|------|-------------|
| Docker | Container management |
| SSH | Remote access |
| Package Install | apt/pip/npm commands |
| Service Management | systemctl operations |
| Network Config | IP/DNS configuration |

---

## 🎯 Quality Utilities

### Deduplication

```python
def deduplicate_examples(examples: List[Dict], key: str = 'question') -> List[Dict]:
    """Remove duplicate examples based on key."""
    seen = set()
    unique = []
    
    for ex in examples:
        hash_key = hashlib.md5(ex[key].encode()).hexdigest()
        if hash_key not in seen:
            seen.add(hash_key)
            unique.append(ex)
    
    return unique
```

### Quality Filtering

```python
def filter_quality(examples: List[Dict], min_length: int = 50) -> List[Dict]:
    """Filter low-quality examples."""
    filtered = []
    
    for ex in examples:
        # Check minimum length
        if len(ex.get('response', '')) < min_length:
            continue
        
        # Check for placeholder text
        if '[PLACEHOLDER]' in ex.get('response', ''):
            continue
        
        # Check for repetition
        if has_excessive_repetition(ex.get('response', '')):
            continue
        
        filtered.append(ex)
    
    return filtered
```

---

## 🚀 Generation Script

### Usage

```bash
# Generate all datasets
python synth/generate_dataset.py --all

# Generate specific dataset
python synth/generate_dataset.py --type cot --count 10000

# Generate with custom seed
python synth/generate_dataset.py --type anti_hallucination --seed 42
```

### Main Script

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['cot', 'anti_hallucination', 'agentic', 'all'])
    parser.add_argument('--count', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='synth/data')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.type == 'cot' or args.type == 'all':
        generator = ChainOfThoughtGenerator(seed=args.seed)
        examples = [generator.generate() for _ in range(args.count)]
        save_jsonl(examples, f'{args.output}/cot_dataset.jsonl')
    
    if args.type == 'anti_hallucination' or args.type == 'all':
        generator = AntiHallucinationGenerator(seed=args.seed)
        # Generate each type
        for dtype in ['idk', 'uncertainty', 'fact_check', 'citation']:
            examples = [generator.generate(dtype) for _ in range(args.count // 4)]
            save_jsonl(examples, f'{args.output}/{dtype}_dataset.jsonl')
    
    # ... similar for other types
```

---

## 📊 Generated Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| cot_dataset.jsonl | ~10K | Chain-of-thought reasoning |
| idk_dataset.jsonl | ~5K | "I don't know" responses |
| uncertainty_dataset.jsonl | ~5K | Uncertainty expressions |
| fact_check_dataset.jsonl | ~5K | Fact verification |
| citation_dataset.jsonl | ~5K | Citation examples |
| fim_dataset.jsonl | ~10K | Fill-in-the-middle |
| commit_dataset.jsonl | ~5K | Git commits |
| diff_dataset.jsonl | ~5K | Code diffs |
| jupyter_dataset.jsonl | ~5K | Jupyter execution |
| shell_dataset.jsonl | ~5K | Shell commands |

---

## 🔗 Related Documentation

- [Data Documentation](../data/README.md) - How synthetic data is used
- [Config Documentation](../config/README.md) - Dataset configuration
- [Training Documentation](../training/README.md) - Training with synthetic data
