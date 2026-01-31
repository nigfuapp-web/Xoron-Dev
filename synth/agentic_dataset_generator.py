#!/usr/bin/env python3
"""
Synthetic dataset generator for agentic coding tokens.

Generates training data for:
1. Fill-In-The-Middle (FIM) code completion
2. Git/Version Control operations
3. Jupyter/Code execution
4. File system operations

Usage:
    python -m synth.agentic_dataset_generator
"""

import os
import sys
import json
import random
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.special_tokens import SPECIAL_TOKENS

# Sample code snippets for various languages
PYTHON_SNIPPETS = [
    ("def calculate_sum(numbers):\n    ", "total = 0\n    for num in numbers:\n        total += num\n    return total", "\n\nresult = calculate_sum([1, 2, 3])"),
    ("class User:\n    def __init__(self, name, email):\n        ", "self.name = name\n        self.email = email", "\n\n    def __str__(self):\n        return f'{self.name} <{self.email}>'"),
    ("import requests\n\ndef fetch_data(url):\n    ", "response = requests.get(url)\n    return response.json()", "\n\ndata = fetch_data('https://api.example.com')"),
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n    ", "return fibonacci(n-1) + fibonacci(n-2)", "\n\nprint(fibonacci(10))"),
    ("async def process_items(items):\n    results = []\n    ", "for item in items:\n        result = await process_single(item)\n        results.append(result)", "\n    return results"),
]

JAVASCRIPT_SNIPPETS = [
    ("function greet(name) {\n    ", "return `Hello, ${name}!`;", "\n}\n\nconsole.log(greet('World'));"),
    ("const fetchUser = async (id) => {\n    ", "const response = await fetch(`/api/users/${id}`);\n    return response.json();", "\n};\n\nfetchUser(1).then(console.log);"),
    ("class Calculator {\n    constructor() {\n        ", "this.result = 0;", "\n    }\n\n    add(n) { this.result += n; return this; }\n}"),
    ("const arr = [1, 2, 3, 4, 5];\nconst doubled = arr.", "map(x => x * 2)", ";\nconsole.log(doubled);"),
    ("const handleSubmit = (event) => {\n    event.preventDefault();\n    ", "const formData = new FormData(event.target);\n    submitForm(formData);", "\n};"),
]

RUST_SNIPPETS = [
    ("fn main() {\n    let numbers = vec![1, 2, 3, 4, 5];\n    ", "let sum: i32 = numbers.iter().sum();", "\n    println!(\"Sum: {}\", sum);\n}"),
    ("struct Point {\n    x: f64,\n    y: f64,\n}\n\nimpl Point {\n    fn new(x: f64, y: f64) -> Self {\n        ", "Point { x, y }", "\n    }\n}"),
    ("fn process_result<T, E>(result: Result<T, E>) -> Option<T> {\n    ", "match result {\n        Ok(value) => Some(value),\n        Err(_) => None,\n    }", "\n}"),
]

GO_SNIPPETS = [
    ("func main() {\n    numbers := []int{1, 2, 3, 4, 5}\n    ", "sum := 0\n    for _, n := range numbers {\n        sum += n\n    }", "\n    fmt.Println(sum)\n}"),
    ("type User struct {\n    Name  string\n    Email string\n}\n\nfunc NewUser(name, email string) *User {\n    ", "return &User{Name: name, Email: email}", "\n}"),
]

# Commit message templates
COMMIT_MESSAGES = [
    "Fix bug in user authentication",
    "Add new feature for data export",
    "Refactor database connection handling",
    "Update dependencies to latest versions",
    "Improve error handling in API endpoints",
    "Add unit tests for payment module",
    "Optimize query performance",
    "Fix memory leak in cache manager",
    "Add support for dark mode",
    "Update documentation for API v2",
    "Fix race condition in worker threads",
    "Add input validation for forms",
    "Implement rate limiting",
    "Fix XSS vulnerability",
    "Add logging for debugging",
]

# Issue templates
ISSUE_TEMPLATES = [
    {"title": "Bug: Application crashes on startup", "body": "When I try to start the application, it crashes with a segmentation fault. Steps to reproduce:\n1. Run `./app`\n2. Observe crash\n\nExpected: App should start normally"},
    {"title": "Feature request: Add dark mode", "body": "It would be great to have a dark mode option for the UI. This would help reduce eye strain during night usage."},
    {"title": "Performance issue: Slow database queries", "body": "The dashboard takes 10+ seconds to load. Profiling shows the main bottleneck is in the user stats query."},
    {"title": "Documentation: Missing API examples", "body": "The API documentation is missing examples for the POST /users endpoint. Please add curl examples."},
    {"title": "Security: Update vulnerable dependency", "body": "CVE-2024-1234 affects our version of lodash. We need to update to 4.17.21 or later."},
]

# File operation templates
FILE_OPERATIONS = [
    {"action": "add", "path": "src/utils/helpers.py", "content": "def format_date(date):\n    return date.strftime('%Y-%m-%d')\n"},
    {"action": "edit", "path": "src/main.py", "old": "DEBUG = True", "new": "DEBUG = False"},
    {"action": "delete", "path": "src/deprecated/old_module.py"},
    {"action": "rename", "old_path": "src/utils.py", "new_path": "src/utils/index.py"},
]

# Jupyter execution templates
JUPYTER_CELLS = [
    {"code": "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()", "output": "   id  name  value\n0   1  Alice    100\n1   2  Bob      200"},
    {"code": "import numpy as np\nnp.random.seed(42)\narr = np.random.randn(5)\nprint(arr)", "output": "[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337]"},
    {"code": "x = 10\ny = 20\nprint(f'Sum: {x + y}')", "output": "Sum: 30"},
    {"code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nfactorial(5)", "output": "120"},
    {"code": "1/0", "error": "ZeroDivisionError: division by zero"},
]


def generate_fim_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a Fill-In-The-Middle code completion sample."""
    lang_snippets = random.choice([
        ("python", PYTHON_SNIPPETS),
        ("javascript", JAVASCRIPT_SNIPPETS),
        ("rust", RUST_SNIPPETS),
        ("go", GO_SNIPPETS),
    ])
    lang, snippets = lang_snippets
    prefix, middle, suffix = random.choice(snippets)
    
    # FIM format: <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE
    text = (
        f"{t['bos']}"
        f"{t['fim_prefix']}{prefix}"
        f"{t['fim_suffix']}{suffix}"
        f"{t['fim_middle']}{middle}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "fim", "language": lang}


def generate_commit_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a Git commit sample with before/after code."""
    lang_snippets = random.choice([
        ("python", PYTHON_SNIPPETS),
        ("javascript", JAVASCRIPT_SNIPPETS),
    ])
    lang, snippets = lang_snippets
    prefix, middle, suffix = random.choice(snippets)
    
    # Create a "before" version (with a bug or missing feature)
    before_code = prefix + "pass  # TODO: implement" + suffix
    after_code = prefix + middle + suffix
    
    commit_msg = random.choice(COMMIT_MESSAGES)
    
    text = (
        f"{t['bos']}"
        f"{t['commit_before']}\n{before_code}\n{t['commit_before_end']}"
        f"{t['commit_after']}\n{after_code}\n{t['commit_after_end']}"
        f"{t['commit_msg']}{commit_msg}{t['commit_msg_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "commit", "language": lang}


DIFF_TEMPLATES = [
    {
        "context": ["def process_data(data):", "    result = []", "    for item in data:", "        result.append(item)", "    return result"],
        "adds": ["    if not data:", "        return []", "        if item is not None:"],
        "explanation": "This diff adds input validation:\n1. Added early return for empty/None input\n2. Added None check before appending items\nThis improves robustness."
    },
    {
        "context": ["class User:", "    def __init__(self, name):", "        self.name = name"],
        "adds": ["        self.created_at = datetime.now()", "        self.is_active = True"],
        "explanation": "This diff adds timestamp and status tracking:\n1. Added created_at timestamp\n2. Added is_active flag for user status"
    },
    {
        "context": ["def calculate_total(items):", "    total = 0", "    for item in items:", "        total += item.price", "    return total"],
        "adds": ["    if not items:", "        return 0", "        if item.price > 0:"],
        "explanation": "This diff adds validation:\n1. Early return for empty list\n2. Only count positive prices\nPrevents errors with invalid data."
    },
    {
        "context": ["async def fetch_data(url):", "    response = await client.get(url)", "    return response.json()"],
        "adds": ["    try:", "    except Exception as e:", "        logger.error(f'Failed: {e}')", "        return None"],
        "explanation": "This diff adds error handling:\n1. Wrapped in try/except\n2. Added logging for failures\n3. Returns None on error"
    },
    {
        "context": ["def connect_db():", "    conn = psycopg2.connect(DATABASE_URL)", "    return conn"],
        "adds": ["    conn.set_session(autocommit=True)", "    conn.cursor().execute('SET timezone = UTC')"],
        "explanation": "This diff improves database connection:\n1. Enabled autocommit\n2. Set timezone to UTC for consistency"
    },
    {
        "context": ["def send_email(to, subject, body):", "    msg = MIMEText(body)", "    smtp.send_message(msg)"],
        "adds": ["    msg['Subject'] = subject", "    msg['To'] = to", "    msg['From'] = SENDER_EMAIL"],
        "explanation": "This diff fixes email headers:\n1. Added Subject header\n2. Added To/From headers\nEmails now have proper metadata."
    },
    {
        "context": ["@app.route('/api/users')", "def get_users():", "    users = User.query.all()", "    return jsonify(users)"],
        "adds": ["@login_required", "    page = request.args.get('page', 1)", "    users = User.query.paginate(page=page)"],
        "explanation": "This diff adds security and pagination:\n1. Added authentication requirement\n2. Added pagination support\nImproves API scalability."
    },
    {
        "context": ["def parse_config(path):", "    with open(path) as f:", "        return json.load(f)"],
        "adds": ["    if not os.path.exists(path):", "        return {}", "        config = json.load(f)", "        return validate_config(config)"],
        "explanation": "This diff adds config validation:\n1. Check if file exists\n2. Validate config structure\nPrevents runtime errors."
    },
]

def generate_diff_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a diverse diff/patch sample."""
    template = random.choice(DIFF_TEMPLATES)
    
    # Build diff text with random additions
    diff_lines = []
    add_idx = 0
    for i, line in enumerate(template["context"]):
        diff_lines.append(f"{t['diff_context']} {line}")
        # Randomly insert additions after certain lines
        if add_idx < len(template["adds"]) and random.random() > 0.4:
            diff_lines.append(f"{t['diff_add']} {template['adds'][add_idx]}")
            add_idx += 1
    
    # Add remaining additions
    while add_idx < len(template["adds"]):
        diff_lines.append(f"{t['diff_add']} {template['adds'][add_idx]}")
        add_idx += 1
    
    diff_text = f"{t['diff_start']}\n" + "\n".join(diff_lines) + f"\n{t['diff_end']}"
    
    prompts = [
        "Review this diff and explain the changes:",
        "What does this diff do?",
        "Explain the modifications in this patch:",
        "Summarize the changes in this diff:",
        "What improvements does this diff make?",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(prompts)}\n{diff_text}\n{t['user_end']}"
        f"{t['assistant_start']}\n{template['explanation']}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "diff"}


def generate_issue_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a GitHub issue sample."""
    issue = random.choice(ISSUE_TEMPLATES)
    is_closed = random.choice([True, False])
    
    status = t['issue_closed'] if is_closed else t['issue_open']
    
    text = (
        f"{t['bos']}"
        f"{t['issue_start']}\n"
        f"{status}\n"
        f"{t['issue_title']}{issue['title']}{t['issue_title_end']}\n"
        f"{t['issue_body']}{issue['body']}{t['issue_body_end']}\n"
        f"{t['issue_end']}"
        f"{t['user_start']}\nSummarize this issue and suggest a fix.\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"**Summary**: {issue['title'].split(':')[-1].strip()}\n\n"
        f"**Suggested approach**: Based on the issue description, I recommend:\n"
        f"1. Investigate the root cause\n"
        f"2. Write a test case to reproduce\n"
        f"3. Implement the fix\n"
        f"4. Verify with the test case\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "issue"}


def generate_jupyter_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a Jupyter notebook execution sample."""
    cell = random.choice(JUPYTER_CELLS)
    
    if "error" in cell:
        output_section = f"{t['jupyter_error']}{cell['error']}{t['jupyter_error_end']}"
    elif cell.get("output"):
        output_section = f"{t['jupyter_output']}{cell['output']}{t['jupyter_output_end']}"
    else:
        output_section = t['empty_output']
    
    text = (
        f"{t['bos']}"
        f"{t['jupyter_start']}\n"
        f"{t['jupyter_code']}\n{cell['code']}\n{t['jupyter_code_end']}\n"
        f"{output_section}\n"
        f"{t['jupyter_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "jupyter"}


def generate_code_execution_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a code execution sample with result."""
    code_examples = [
        {"code": "print('Hello, World!')", "result": "Hello, World!"},
        {"code": "sum([1, 2, 3, 4, 5])", "result": "15"},
        {"code": "len('Python')", "result": "6"},
        {"code": "'hello'.upper()", "result": "'HELLO'"},
        {"code": "list(range(5))", "result": "[0, 1, 2, 3, 4]"},
    ]
    
    example = random.choice(code_examples)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nExecute this code and show the result:\n"
        f"{t['code_start']}{t['lang_python']}\n{example['code']}\n{t['code_end']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n{example['code']}\n{t['exec_end']}\n"
        f"{t['exec_result']}{example['result']}{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "execution"}



# Shell command execution templates
SHELL_COMMANDS = [
    {"cmd": "ls -la", "output": "total 48\ndrwxr-xr-x  5 user user 4096 Jan 24 10:00 .\ndrwxr-xr-x  3 user user 4096 Jan 24 09:00 ..\n-rw-r--r--  1 user user  256 Jan 24 10:00 README.md\ndrwxr-xr-x  2 user user 4096 Jan 24 10:00 src\n-rw-r--r--  1 user user  128 Jan 24 10:00 requirements.txt"},
    {"cmd": "pwd", "output": "/home/user/project"},
    {"cmd": "cat requirements.txt", "output": "numpy>=1.21.0\npandas>=1.3.0\nrequests>=2.26.0\npytest>=6.2.0"},
    {"cmd": "echo $PATH", "output": "/usr/local/bin:/usr/bin:/bin"},
    {"cmd": "which python", "output": "/usr/bin/python3"},
    {"cmd": "python --version", "output": "Python 3.10.12"},
    {"cmd": "pip list | head -5", "output": "Package    Version\n---------- -------\nnumpy      1.24.0\npandas     2.0.0\nrequests   2.28.0"},
    {"cmd": "git status", "output": "On branch main\nYour branch is up to date with 'origin/main'.\n\nnothing to commit, working tree clean"},
    {"cmd": "git log --oneline -3", "output": "a1b2c3d Fix bug in parser\ne4f5g6h Add new feature\ni7j8k9l Initial commit"},
    {"cmd": "find . -name '*.py' | head -5", "output": "./src/main.py\n./src/utils.py\n./tests/test_main.py"},
    {"cmd": "wc -l src/*.py", "output": "  45 src/main.py\n  32 src/utils.py\n  77 total"},
    {"cmd": "grep -r 'TODO' src/", "output": "src/main.py:12:    # TODO: Add error handling\nsrc/utils.py:28:    # TODO: Optimize this function"},
    {"cmd": "mkdir -p new_folder && ls", "output": "README.md  new_folder  requirements.txt  src"},
    {"cmd": "touch test.txt && echo 'created'", "output": "created"},
    {"cmd": "rm -f test.txt && echo 'deleted'", "output": "deleted"},
]

SHELL_ERRORS = [
    {"cmd": "cat nonexistent.txt", "error": "cat: nonexistent.txt: No such file or directory", "exit_code": 1},
    {"cmd": "cd /root", "error": "bash: cd: /root: Permission denied", "exit_code": 1},
    {"cmd": "pip install nonexistent-package-xyz", "error": "ERROR: Could not find a version that satisfies the requirement nonexistent-package-xyz", "exit_code": 1},
    {"cmd": "python syntax_error.py", "error": "  File \"syntax_error.py\", line 1\n    def broken(\n              ^\nSyntaxError: unexpected EOF while parsing", "exit_code": 1},
    {"cmd": "git push origin main", "error": "fatal: Authentication failed for 'https://github.com/user/repo.git'", "exit_code": 128},
]

SHELL_TIMEOUT = [
    {"cmd": "sleep 100"},
    {"cmd": "while true; do echo 'loop'; done"},
    {"cmd": "find / -name '*.log' 2>/dev/null"},
]

MULTI_STEP_SCENARIOS = [
    {
        "task": "Set up a Python virtual environment and install dependencies",
        "steps": [
            {"cmd": "python -m venv venv", "output": ""},
            {"cmd": "source venv/bin/activate", "output": ""},
            {"cmd": "pip install -r requirements.txt", "output": "Successfully installed numpy-1.24.0 pandas-2.0.0 requests-2.28.0"},
            {"cmd": "pip list", "output": "Package    Version\n---------- -------\nnumpy      1.24.0\npandas     2.0.0\nrequests   2.28.0"},
        ]
    },
    {
        "task": "Clone a repository and run tests",
        "steps": [
            {"cmd": "git clone https://github.com/example/repo.git", "output": "Cloning into 'repo'...\ndone."},
            {"cmd": "cd repo && ls", "output": "README.md  setup.py  src  tests"},
            {"cmd": "pip install -e .", "output": "Successfully installed repo-0.1.0"},
            {"cmd": "pytest tests/", "output": "===== 5 passed in 0.42s ====="},
        ]
    },
    {
        "task": "Create a new Python project structure",
        "steps": [
            {"cmd": "mkdir -p myproject/src myproject/tests", "output": ""},
            {"cmd": "touch myproject/src/__init__.py myproject/tests/__init__.py", "output": ""},
            {"cmd": "echo 'def main(): print(\"Hello\")' > myproject/src/main.py", "output": ""},
            {"cmd": "tree myproject", "output": "myproject\n├── src\n│   ├── __init__.py\n│   └── main.py\n└── tests\n    └── __init__.py"},
        ]
    },
    {
        "task": "Check system resources",
        "steps": [
            {"cmd": "df -h | head -5", "output": "Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       100G   45G   55G  45% /"},
            {"cmd": "free -h", "output": "              total        used        free\nMem:           16Gi       8.0Gi       8.0Gi"},
            {"cmd": "nproc", "output": "8"},
            {"cmd": "uname -a", "output": "Linux hostname 5.15.0-generic x86_64 GNU/Linux"},
        ]
    },
    {
        "task": "Debug a failing Python script",
        "steps": [
            {"cmd": "python buggy.py", "error": "Traceback (most recent call last):\n  File \"buggy.py\", line 3\n    result = x / y\nZeroDivisionError: division by zero"},
            {"cmd": "cat buggy.py", "output": "x = 10\ny = 0\nresult = x / y\nprint(result)"},
            {"cmd": "sed -i 's/y = 0/y = 2/' buggy.py", "output": ""},
            {"cmd": "python buggy.py", "output": "5.0"},
        ]
    },
]

PYTHON_SCRIPTS = [
    {
        "description": "Run a data processing script",
        "code": "import json\ndata = {'name': 'test', 'value': 42}\nprint(json.dumps(data, indent=2))",
        "output": "{\n  \"name\": \"test\",\n  \"value\": 42\n}"
    },
    {
        "description": "Calculate statistics",
        "code": "numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\nprint(f'Sum: {sum(numbers)}')\nprint(f'Avg: {sum(numbers)/len(numbers)}')\nprint(f'Max: {max(numbers)}')",
        "output": "Sum: 55\nAvg: 5.5\nMax: 10"
    },
    {
        "description": "File operations",
        "code": "with open('output.txt', 'w') as f:\n    f.write('Hello, World!')\nwith open('output.txt', 'r') as f:\n    print(f.read())",
        "output": "Hello, World!"
    },
    {
        "description": "HTTP request simulation",
        "code": "# Simulating API response\nresponse = {'status': 200, 'data': {'id': 1, 'name': 'User'}}\nprint(f\"Status: {response['status']}\")\nprint(f\"Data: {response['data']}\")",
        "output": "Status: 200\nData: {'id': 1, 'name': 'User'}"
    },
]


def generate_shell_execution_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a shell/terminal command execution sample."""
    example = random.choice(SHELL_COMMANDS)

    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nRun this command: {example['cmd']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {example['cmd']}\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{example['output']}\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "shell_execution"}


def generate_shell_error_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a shell command with error handling."""
    example = random.choice(SHELL_ERRORS)

    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nExecute: {example['cmd']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {example['cmd']}\n{t['exec_end']}\n"
        f"{t['exec_error']}\n{example['error']}\n{t['exec_error_end']}\n"
        f"The command failed with exit code {example['exit_code']}.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "shell_error"}


def generate_shell_timeout_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a shell command that times out."""
    example = random.choice(SHELL_TIMEOUT)

    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nRun: {example['cmd']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {example['cmd']}\n{t['exec_end']}\n"
        f"{t['exec_timeout']}\n"
        f"The command timed out after 30 seconds. I'll interrupt it and try a different approach.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "shell_timeout"}


def generate_multi_step_execution_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a multi-step execution scenario."""
    scenario = random.choice(MULTI_STEP_SCENARIOS)

    steps_text = ""
    for step in scenario["steps"]:
        steps_text += f"{t['exec_start']}\n$ {step['cmd']}\n{t['exec_end']}\n"
        if "error" in step:
            steps_text += f"{t['exec_error']}\n{step['error']}\n{t['exec_error_end']}\n"
        elif step.get("output"):
            steps_text += f"{t['exec_result']}\n{step['output']}\n{t['exec_result_end']}\n"
        else:
            steps_text += f"{t['exec_result']}(no output){t['exec_result_end']}\n"

    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{scenario['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {scenario['task'].lower()}. Let me execute the necessary commands:\n\n"
        f"{steps_text}"
        f"\nDone! All steps completed successfully.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "multi_step_execution"}


def generate_python_script_execution_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a Python script execution sample."""
    script = random.choice(PYTHON_SCRIPTS)

    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{script['description']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll write and execute a Python script for this:\n\n"
        f"{t['code_start']}{t['lang_python']}\n{script['code']}\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ python script.py\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{script['output']}\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "python_script_execution"}


DEBUG_SCENARIOS = [
    {
        "code": "def process(data):\n    return data['value'] * 2\n\nresult = process(None)",
        "error": "TypeError: 'NoneType' object is not subscriptable",
        "fix": "def process(data):\n    if data is None:\n        return 0\n    return data['value'] * 2",
        "explanation": "The issue is that `process(None)` is called with `None` instead of a dictionary. Added a None check."
    },
    {
        "code": "numbers = [1, 2, 3, 4, 5]\nprint(numbers[10])",
        "error": "IndexError: list index out of range",
        "fix": "numbers = [1, 2, 3, 4, 5]\nif len(numbers) > 10:\n    print(numbers[10])\nelse:\n    print('Index out of range')",
        "explanation": "The list only has 5 elements but we're trying to access index 10. Added bounds checking."
    },
    {
        "code": "def divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
        "error": "ZeroDivisionError: division by zero",
        "fix": "def divide(a, b):\n    if b == 0:\n        return float('inf')\n    return a / b",
        "explanation": "Division by zero is undefined. Added a check to handle zero divisor."
    },
    {
        "code": "import json\ndata = json.loads('{invalid json}')",
        "error": "json.decoder.JSONDecodeError: Expecting property name",
        "fix": "import json\ntry:\n    data = json.loads('{\"valid\": \"json\"}')\nexcept json.JSONDecodeError as e:\n    print(f'Invalid JSON: {e}')\n    data = {}",
        "explanation": "The JSON string was malformed. Added try/except to handle parse errors gracefully."
    },
    {
        "code": "text = 'hello'\nresult = text + 123",
        "error": "TypeError: can only concatenate str (not \"int\") to str",
        "fix": "text = 'hello'\nresult = text + str(123)",
        "explanation": "Cannot concatenate string and integer directly. Convert the integer to string first."
    },
    {
        "code": "d = {'a': 1, 'b': 2}\nvalue = d['c']",
        "error": "KeyError: 'c'",
        "fix": "d = {'a': 1, 'b': 2}\nvalue = d.get('c', 0)  # Returns 0 if key doesn't exist",
        "explanation": "Key 'c' doesn't exist in the dictionary. Use .get() with a default value."
    },
    {
        "code": "with open('nonexistent.txt') as f:\n    content = f.read()",
        "error": "FileNotFoundError: [Errno 2] No such file or directory: 'nonexistent.txt'",
        "fix": "import os\nif os.path.exists('nonexistent.txt'):\n    with open('nonexistent.txt') as f:\n        content = f.read()\nelse:\n    content = ''",
        "explanation": "The file doesn't exist. Added existence check before opening."
    },
    {
        "code": "class User:\n    def __init__(self, name):\n        self.name = name\n\nu = User()\nprint(u.name)",
        "error": "TypeError: __init__() missing 1 required positional argument: 'name'",
        "fix": "class User:\n    def __init__(self, name='Anonymous'):\n        self.name = name\n\nu = User()\nprint(u.name)",
        "explanation": "The constructor requires a 'name' argument. Added a default value."
    },
    {
        "code": "nums = [1, 2, 3]\nfor i in range(len(nums)):\n    if nums[i] == 2:\n        nums.remove(nums[i])\n    print(nums[i])",
        "error": "IndexError: list index out of range",
        "fix": "nums = [1, 2, 3]\nnums = [n for n in nums if n != 2]\nfor n in nums:\n    print(n)",
        "explanation": "Modifying a list while iterating causes index issues. Use list comprehension instead."
    },
    {
        "code": "import requests\nresponse = requests.get('https://invalid-url-12345.com')\ndata = response.json()",
        "error": "requests.exceptions.ConnectionError: Failed to establish connection",
        "fix": "import requests\ntry:\n    response = requests.get('https://api.example.com', timeout=5)\n    response.raise_for_status()\n    data = response.json()\nexcept requests.RequestException as e:\n    print(f'Request failed: {e}')\n    data = {}",
        "explanation": "Network requests can fail. Added timeout and error handling."
    },
]

def generate_interactive_debugging_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a diverse interactive debugging session."""
    scenario = random.choice(DEBUG_SCENARIOS)
    
    prompts = [
        "My Python script is crashing. Can you help debug it?",
        "I'm getting an error with this code. What's wrong?",
        "This code throws an exception. How do I fix it?",
        "Can you help me understand why this fails?",
        "Debug this code for me:",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(prompts)}\n"
        f"{t['code_start']}{t['lang_python']}\n{scenario['code']}\n{t['code_end']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"Let me run this and see the error:\n\n"
        f"{t['exec_start']}\n$ python debug.py\n{t['exec_end']}\n"
        f"{t['exec_error']}\n{scenario['error']}\n{t['exec_error_end']}\n\n"
        f"{scenario['explanation']}\n\nHere's the fixed code:\n\n"
        f"{t['code_start']}{t['lang_python']}\n{scenario['fix']}\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ python debug_fixed.py\n{t['exec_end']}\n"
        f"{t['exec_result']}\n(runs successfully)\n{t['exec_result_end']}\n\n"
        f"Fixed! The script now handles this case properly.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )

    return {"text": text, "type": "interactive_debugging"}


def generate_file_operation_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a file operation sample."""
    op = random.choice(FILE_OPERATIONS)
    
    if op["action"] == "add":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nCreate a new file with helper functions.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['add_file']}\n"
            f"{t['filepath_start']}{op['path']}{t['filepath_end']}\n"
            f"{t['file_content']}\n{op['content']}\n{t['file_content_end']}\n"
            f"{t['add_file_end']}\n"
            f"Created new file at {op['path']}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    elif op["action"] == "edit":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nDisable debug mode in the main file.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['edit_file']}\n"
            f"{t['filepath_start']}{op['path']}{t['filepath_end']}\n"
            f"{t['replace']}\n{op['old']}\n{t['replace_end']}\n"
            f"{t['file_content']}\n{op['new']}\n{t['file_content_end']}\n"
            f"{t['edit_file_end']}\n"
            f"Updated {op['path']}: changed '{op['old']}' to '{op['new']}'\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    elif op["action"] == "delete":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nRemove the deprecated module.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['delete_file']}\n"
            f"{t['filepath_start']}{op['path']}{t['filepath_end']}\n"
            f"{t['delete_file_end']}\n"
            f"Deleted file: {op['path']}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    else:  # rename
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nRename utils.py to be inside a utils folder.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['rename_file']}\n"
            f"{t['filepath_start']}{op['old_path']}{t['filepath_end']}\n"
            f"{t['file_sep']}\n"
            f"{t['filepath_start']}{op['new_path']}{t['filepath_end']}\n"
            f"{t['rename_file_end']}\n"
            f"Renamed {op['old_path']} to {op['new_path']}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    
    return {"text": text, "type": "file_operation", "action": op["action"]}


def generate_edit_with_line_numbers(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a precise edit operation with line numbers."""
    original_code = """def process_data(data):
    result = []
    for item in data:
        result.append(item)
    return result"""
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nAdd input validation at line 2 of this function:\n"
        f"{t['code_start']}{t['lang_python']}\n{original_code}\n{t['code_end']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['edit_file']}\n"
        f"{t['edit_range']}{t['line_num']}2{t['line_num_end']}{t['edit_range_end']}\n"
        f"{t['insert_after']}\n"
        f"    if data is None:\n"
        f"        return []\n"
        f"{t['edit_file_end']}\n"
        f"Added input validation after line 2.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "edit_lines"}


REPO_DATABASE = [
    {"name": "facebook/react", "stars": 220000, "desc": "A JavaScript library for building user interfaces", "lang": "JavaScript", "topics": ["frontend", "ui", "components"]},
    {"name": "microsoft/vscode", "stars": 155000, "desc": "Visual Studio Code - a lightweight but powerful source code editor", "lang": "TypeScript", "topics": ["editor", "ide", "development"]},
    {"name": "tensorflow/tensorflow", "stars": 180000, "desc": "An open-source machine learning framework", "lang": "Python/C++", "topics": ["ml", "deep-learning", "neural-networks"]},
    {"name": "torvalds/linux", "stars": 165000, "desc": "The Linux kernel source tree", "lang": "C", "topics": ["kernel", "os", "systems"]},
    {"name": "python/cpython", "stars": 58000, "desc": "The Python programming language interpreter", "lang": "Python/C", "topics": ["interpreter", "language", "runtime"]},
    {"name": "rust-lang/rust", "stars": 90000, "desc": "A systems programming language focused on safety and performance", "lang": "Rust", "topics": ["compiler", "language", "systems"]},
    {"name": "golang/go", "stars": 118000, "desc": "The Go programming language", "lang": "Go", "topics": ["compiler", "language", "concurrency"]},
    {"name": "nodejs/node", "stars": 102000, "desc": "Node.js JavaScript runtime", "lang": "JavaScript/C++", "topics": ["runtime", "server", "javascript"]},
    {"name": "kubernetes/kubernetes", "stars": 105000, "desc": "Production-grade container orchestration", "lang": "Go", "topics": ["containers", "orchestration", "cloud"]},
    {"name": "docker/docker-ce", "stars": 67000, "desc": "Docker Engine - container runtime", "lang": "Go", "topics": ["containers", "virtualization", "devops"]},
    {"name": "openai/whisper", "stars": 55000, "desc": "Robust speech recognition via large-scale weak supervision", "lang": "Python", "topics": ["asr", "speech", "ml"]},
    {"name": "huggingface/transformers", "stars": 120000, "desc": "State-of-the-art ML for PyTorch, TensorFlow, and JAX", "lang": "Python", "topics": ["nlp", "transformers", "ml"]},
    {"name": "langchain-ai/langchain", "stars": 75000, "desc": "Building applications with LLMs through composability", "lang": "Python", "topics": ["llm", "agents", "ai"]},
    {"name": "vercel/next.js", "stars": 118000, "desc": "The React Framework for the Web", "lang": "JavaScript", "topics": ["react", "ssr", "web"]},
    {"name": "sveltejs/svelte", "stars": 75000, "desc": "Cybernetically enhanced web apps", "lang": "JavaScript", "topics": ["frontend", "compiler", "web"]},
]

REPO_QUESTIONS = [
    ("What is this repository about?", lambda r: f"**{r['name'].split('/')[-1]}** is {r['desc'].lower()}. It's written primarily in {r['lang']} and has {r['stars']:,} stars on GitHub.\n\nKey topics: {', '.join(r['topics'])}"),
    ("What language is this project written in?", lambda r: f"This project ({r['name']}) is primarily written in **{r['lang']}**. It's a {r['desc'].lower()}."),
    ("How popular is this repository?", lambda r: f"**{r['name']}** has {r['stars']:,} stars on GitHub, making it one of the most popular {r['topics'][0]} projects. It's maintained by {r['name'].split('/')[0]}."),
    ("What are the main features?", lambda r: f"**{r['name'].split('/')[-1]}** focuses on {', '.join(r['topics'])}. {r['desc']}. The project is written in {r['lang']}."),
    ("Who maintains this project?", lambda r: f"This repository is maintained by **{r['name'].split('/')[0]}**. It's their {r['name'].split('/')[-1]} project with {r['stars']:,} stars."),
    ("Can you summarize this repo?", lambda r: f"**Summary of {r['name']}:**\n- Description: {r['desc']}\n- Language: {r['lang']}\n- Stars: {r['stars']:,}\n- Topics: {', '.join(r['topics'])}"),
]

def generate_repo_context_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a diverse sample with repository context."""
    repo = random.choice(REPO_DATABASE)
    question, answer_fn = random.choice(REPO_QUESTIONS)
    branch = random.choice(["main", "master", "develop", "dev"])
    
    text = (
        f"{t['bos']}"
        f"{t['reponame']}{repo['name']}{t['reponame_end']}"
        f"{t['gh_stars']}{repo['stars']}{t['gh_stars_end']}"
        f"{t['branch']}{branch}{t['branch_end']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n{answer_fn(repo)}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "repo_context"}


def generate_dataset(output_dir: str, samples_per_type: int = 2000):
    """Generate all agentic datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    t = SPECIAL_TOKENS
    
    generators = [
        ("fim_dataset.jsonl", generate_fim_sample, "Fill-In-The-Middle"),
        ("commit_dataset.jsonl", generate_commit_sample, "Git Commits"),
        ("diff_dataset.jsonl", generate_diff_sample, "Diffs"),
        ("issue_dataset.jsonl", generate_issue_sample, "GitHub Issues"),
        ("jupyter_dataset.jsonl", generate_jupyter_sample, "Jupyter Execution"),
        ("execution_dataset.jsonl", generate_code_execution_sample, "Code Execution"),
        ("shell_execution_dataset.jsonl", generate_shell_execution_sample, "Shell Execution"),
        ("shell_error_dataset.jsonl", generate_shell_error_sample, "Shell Errors"),
        ("shell_timeout_dataset.jsonl", generate_shell_timeout_sample, "Shell Timeouts"),
        ("multi_step_execution_dataset.jsonl", generate_multi_step_execution_sample, "Multi-Step Execution"),
        ("python_script_dataset.jsonl", generate_python_script_execution_sample, "Python Scripts"),
        ("debugging_dataset.jsonl", generate_interactive_debugging_sample, "Interactive Debugging"),
        ("file_ops_dataset.jsonl", generate_file_operation_sample, "File Operations"),
        ("edit_lines_dataset.jsonl", generate_edit_with_line_numbers, "Line Edits"),
        ("repo_context_dataset.jsonl", generate_repo_context_sample, "Repo Context"),
    ]
    
    for filename, generator, desc in generators:
        filepath = os.path.join(output_dir, filename)
        print(f"📝 Generating {desc} dataset ({samples_per_type} samples)...")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(samples_per_type):
                sample = generator(t)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"   ✅ Saved to {filepath}")
    
    print(f"\n✅ Generated {len(generators)} datasets with {samples_per_type} samples each")
    print(f"   Total: {len(generators) * samples_per_type} samples")


# Removed standalone execution - use as module instead
# # if __name__ == "__main__":
#     output_dir = os.path.join(os.path.dirname(__file__), "data")
    # generate_dataset(output_dir, samples_per_type=2000)
