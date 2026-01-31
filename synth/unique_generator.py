"""
High-Quality Unique Dataset Generator

This module generates unique samples for each dataset type by using parameterized
generation that combines random elements to ensure no duplicates.

Key principle: Instead of choosing from a small pool, we GENERATE unique
samples by combining random:
- Names, numbers, values
- Code snippets, functions, variables
- Commands, paths, outputs
- Questions, contexts, answers

This is the main dataset generation module for Xoron-Dev. It consolidates
all dataset types into a single generator with guaranteed uniqueness.

Usage:
    from synth.unique_generator import generate_all_datasets
    generate_all_datasets('./synth/data', samples_per_type=2000)
    
    # Or generate a single dataset type
    from synth.unique_generator import generate_unique_dataset, generate_unique_shell_execution
    from config.special_tokens import SPECIAL_TOKENS
    generate_unique_dataset(generate_unique_shell_execution, 'shell.jsonl', 1000)
"""

import os
import json
import random
import hashlib
from typing import Dict, List, Any, Set, Callable, Tuple

from config.special_tokens import SPECIAL_TOKENS

# =============================================================================
# RANDOM DATA POOLS FOR PARAMETERIZATION
# =============================================================================

NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
         "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tara",
         "Uma", "Victor", "Wendy", "Xavier", "Yuki", "Zara", "Ahmed", "Bella", "Carlos", "Diya"]

LANGUAGES = ["Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++", "C", "Ruby", 
             "PHP", "Swift", "Kotlin", "Scala", "Haskell", "Elixir", "Clojure", "R", "Julia"]

FRAMEWORKS = ["React", "Vue", "Angular", "Django", "Flask", "FastAPI", "Express", "Next.js",
              "Spring", "Rails", "Laravel", "Gin", "Echo", "Actix", "Rocket", "Phoenix"]

DATABASES = ["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Cassandra", "DynamoDB",
             "Elasticsearch", "Neo4j", "CouchDB", "MariaDB", "Oracle", "SQL Server"]

TOOLS = ["Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitLab CI", "GitHub Actions",
         "CircleCI", "Travis CI", "AWS", "GCP", "Azure", "Nginx", "Apache", "Caddy"]

PACKAGES = ["numpy", "pandas", "requests", "flask", "django", "fastapi", "sqlalchemy", "pytest",
            "tensorflow", "pytorch", "scikit-learn", "matplotlib", "seaborn", "celery", "redis",
            "boto3", "pillow", "opencv-python", "transformers", "langchain", "openai"]

APT_PACKAGES = ["curl", "wget", "git", "vim", "nano", "htop", "tree", "jq", "unzip", "zip",
                "build-essential", "gcc", "g++", "make", "cmake", "python3", "python3-pip",
                "nodejs", "npm", "nginx", "apache2", "postgresql", "mysql-server", "redis-server",
                "docker.io", "ffmpeg", "imagemagick", "tmux", "screen", "zsh", "fish"]

SHELL_COMMANDS = ["ls", "cd", "pwd", "cat", "grep", "find", "awk", "sed", "sort", "uniq",
                  "wc", "head", "tail", "cut", "tr", "xargs", "tee", "diff", "patch"]

FILE_EXTENSIONS = [".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp", ".c", ".rb", ".php",
                   ".json", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".md", ".txt"]

DIRECTORIES = ["src", "lib", "app", "api", "core", "utils", "helpers", "models", "views",
               "controllers", "services", "handlers", "middleware", "config", "tests"]

VERBS = ["create", "update", "delete", "fetch", "process", "validate", "transform", "parse",
         "serialize", "deserialize", "encode", "decode", "compress", "decompress", "encrypt"]

NOUNS = ["user", "item", "order", "product", "customer", "account", "session", "token",
         "message", "notification", "event", "task", "job", "request", "response", "data"]

ADJECTIVES = ["new", "old", "active", "inactive", "valid", "invalid", "pending", "completed",
              "failed", "successful", "primary", "secondary", "default", "custom"]

ERROR_TYPES = ["TypeError", "ValueError", "KeyError", "IndexError", "AttributeError",
               "FileNotFoundError", "ConnectionError", "TimeoutError", "PermissionError",
               "ImportError", "ModuleNotFoundError", "ZeroDivisionError", "RuntimeError"]

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
HTTP_CODES = [200, 201, 204, 400, 401, 403, 404, 500, 502, 503]

PORTS = [80, 443, 3000, 3001, 5000, 5432, 6379, 8000, 8080, 8443, 9000, 27017]

IP_PREFIXES = ["192.168.1", "10.0.0", "172.16.0", "192.168.0", "10.1.1"]

DOMAINS = ["example.com", "api.example.com", "dev.example.com", "staging.example.com",
           "test.local", "localhost", "myapp.io", "service.internal"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def random_ip():
    prefix = random.choice(IP_PREFIXES)
    return f"{prefix}.{random.randint(1, 254)}"

def random_port():
    return random.choice(PORTS) + random.randint(0, 100)

def random_version():
    return f"{random.randint(1, 20)}.{random.randint(0, 99)}.{random.randint(0, 99)}"

def random_hash():
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:12]

def random_func_name():
    return f"{random.choice(VERBS)}_{random.choice(NOUNS)}"

def random_var_name():
    return f"{random.choice(ADJECTIVES)}_{random.choice(NOUNS)}"

def random_class_name():
    return f"{random.choice(ADJECTIVES).title()}{random.choice(NOUNS).title()}"

def random_file_path():
    return f"{random.choice(DIRECTORIES)}/{random_func_name()}{random.choice(FILE_EXTENSIONS)}"

def random_url():
    return f"https://{random.choice(DOMAINS)}/api/v{random.randint(1,3)}/{random.choice(NOUNS)}s"

def random_number(min_val=1, max_val=1000):
    return random.randint(min_val, max_val)


# =============================================================================
# AVAILABLE TOOLS DEFINITIONS
# =============================================================================

# Tool definitions for different contexts
SHELL_TOOLS = [
    {"name": "execute_bash", "description": "Execute a bash command in the terminal", "parameters": {"command": "string"}},
]

CODE_EXECUTION_TOOLS = [
    {"name": "execute_bash", "description": "Execute a bash command in the terminal", "parameters": {"command": "string"}},
    {"name": "execute_ipython_cell", "description": "Run Python code in an IPython environment", "parameters": {"code": "string"}},
]

FILE_TOOLS = [
    {"name": "str_replace_editor", "description": "View, create and edit files", "parameters": {"command": "string", "path": "string", "old_str": "string", "new_str": "string"}},
    {"name": "execute_bash", "description": "Execute a bash command in the terminal", "parameters": {"command": "string"}},
]

FULL_TOOLS = [
    {"name": "execute_bash", "description": "Execute a bash command in the terminal", "parameters": {"command": "string"}},
    {"name": "execute_ipython_cell", "description": "Run Python code in an IPython environment", "parameters": {"code": "string"}},
    {"name": "str_replace_editor", "description": "View, create and edit files", "parameters": {"command": "string", "path": "string"}},
    {"name": "browser", "description": "Interact with a web browser", "parameters": {"code": "string"}},
    {"name": "fetch", "description": "Fetch a URL from the internet", "parameters": {"url": "string"}},
]


def format_available_tools(t: Dict[str, str], tools: List[Dict]) -> str:
    """Format available tools section for inclusion in prompts."""
    if not tools:
        return ""
    
    tools_json = json.dumps(tools, indent=2)
    return f"{t['available_tools_start']}\n{tools_json}\n{t['available_tools_end']}\n"


def maybe_include_tools(t: Dict[str, str], tools: List[Dict], probability: float = 0.5) -> str:
    """Randomly include available tools section based on probability."""
    if random.random() < probability:
        return format_available_tools(t, tools)
    return ""

# =============================================================================
# UNIQUE SAMPLE GENERATORS
# =============================================================================

def generate_unique_shell_execution(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique shell execution sample with available tools."""
    cmd_templates = [
        (f"ls -la {random.choice(DIRECTORIES)}/", f"total {random_number(10, 100)}\ndrwxr-xr-x {random_number(2,10)} user user {random_number(100, 9999)} Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)} ."),
        (f"cat {random_file_path()}", f"# {random_class_name()}\n\ndef {random_func_name()}():\n    pass"),
        (f"grep -r '{random_var_name()}' {random.choice(DIRECTORIES)}/", f"{random_file_path()}:{random_number(1,100)}:    {random_var_name()} = {random_number()}"),
        (f"find . -name '*.{random.choice(['py', 'js', 'ts', 'go', 'rs'])}' | wc -l", str(random_number(5, 500))),
        (f"wc -l {random_file_path()}", f"  {random_number(50, 2000)} {random_file_path()}"),
        (f"head -n {random_number(5, 20)} {random_file_path()}", f"#!/usr/bin/env python3\n# {random_class_name()}\n\nimport {random.choice(PACKAGES)}"),
        (f"tail -f /var/log/{random.choice(['syslog', 'auth.log', 'nginx/access.log', 'apache2/error.log'])}", f"Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)} server {random.choice(['INFO', 'WARN', 'ERROR'])}: {random_func_name()} completed"),
        (f"ps aux | grep {random.choice(['python', 'node', 'java', 'nginx', 'postgres'])}", f"user     {random_number(1000, 9999)}  {random.uniform(0.1, 5.0):.1f}  {random.uniform(0.1, 3.0):.1f} {random_number(10000, 99999)} {random_number(1000, 9999)} pts/0    S+   {random_number(10,23)}:{random_number(10,59)}   0:0{random_number(0,9)} {random.choice(['python', 'node'])} {random_file_path()}"),
        (f"df -h | head -5", f"Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       {random_number(50, 500)}G   {random_number(10, 200)}G   {random_number(50, 300)}G  {random_number(10, 90)}% /"),
        (f"free -h", f"              total        used        free\nMem:           {random_number(8, 64)}Gi       {random_number(2, 32)}Gi       {random_number(2, 32)}Gi"),
        (f"du -sh {random.choice(DIRECTORIES)}/", f"{random_number(1, 500)}M\t{random.choice(DIRECTORIES)}/"),
        (f"chmod {random.choice(['644', '755', '600', '700'])} {random_file_path()}", ""),
        (f"mkdir -p {random.choice(DIRECTORIES)}/{random.choice(DIRECTORIES)}/{random_var_name()}", ""),
        (f"cp {random_file_path()} {random_file_path()}.bak", ""),
        (f"mv {random_file_path()} {random_file_path()}", ""),
        (f"rm -f {random_file_path()}", ""),
        (f"touch {random_file_path()}", ""),
        (f"echo '{random_var_name()} = {random_number()}' >> {random_file_path()}", ""),
        (f"curl -s {random_url()} | jq '.{random.choice(NOUNS)}'", f'{{"id": {random_number()}, "name": "{random.choice(NAMES)}"}}'),
        (f"wget -q {random_url()}/{random_hash()}.tar.gz", ""),
    ]
    
    cmd, output = random.choice(cmd_templates)
    output_text = output if output else "(no output)"
    
    # Include available tools in ~50% of samples
    tools_section = maybe_include_tools(t, SHELL_TOOLS, probability=0.5)
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['user_start']}\nRun: {cmd}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{output_text}\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "shell_execution", "idx": idx}


def generate_unique_shell_error(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique shell error sample with available tools."""
    error_templates = [
        (f"cat {random_file_path()}", f"cat: {random_file_path()}: No such file or directory"),
        (f"cd {random.choice(DIRECTORIES)}/{random_hash()}", f"bash: cd: {random.choice(DIRECTORIES)}/{random_hash()}: No such file or directory"),
        (f"python {random_file_path()}", f"python: can't open file '{random_file_path()}': [Errno 2] No such file or directory"),
        (f"rm {random_file_path()}", f"rm: cannot remove '{random_file_path()}': No such file or directory"),
        (f"chmod 777 /etc/{random_var_name()}", f"chmod: changing permissions of '/etc/{random_var_name()}': Operation not permitted"),
        (f"sudo apt install {random_hash()}", f"E: Unable to locate package {random_hash()}"),
        (f"pip install {random_hash()}", f"ERROR: Could not find a version that satisfies the requirement {random_hash()}"),
        (f"npm install {random_hash()}", f"npm ERR! 404 Not Found - GET https://registry.npmjs.org/{random_hash()} - Not found"),
        (f"docker run {random_hash()}", f"Unable to find image '{random_hash()}:latest' locally\ndocker: Error response from daemon: pull access denied"),
        (f"git clone https://github.com/{random_hash()}/{random_hash()}.git", f"Cloning into '{random_hash()}'...\nfatal: repository 'https://github.com/{random_hash()}/{random_hash()}.git/' not found"),
        (f"ssh {random.choice(NAMES).lower()}@{random_ip()}", f"ssh: connect to host {random_ip()} port 22: Connection refused"),
        (f"curl {random_url()}/{random_hash()}", f"curl: (6) Could not resolve host: {random.choice(DOMAINS)}"),
        (f"psql -U {random.choice(NAMES).lower()} -d {random_var_name()}", f"psql: error: connection to server on socket \"/var/run/postgresql/.s.PGSQL.5432\" failed: FATAL:  database \"{random_var_name()}\" does not exist"),
        (f"mysql -u {random.choice(NAMES).lower()} -p {random_var_name()}", f"ERROR 1049 (42000): Unknown database '{random_var_name()}'"),
        (f"node {random_file_path()}", f"node:internal/modules/cjs/loader:1080\n  throw err;\n  ^\nError: Cannot find module '{random_file_path()}'"),
    ]
    
    cmd, error = random.choice(error_templates)
    tools_section = maybe_include_tools(t, SHELL_TOOLS, probability=0.5)
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['user_start']}\nExecute: {cmd}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n"
        f"{t['exec_error']}\n{error}\n{t['exec_error_end']}\n"
        f"The command failed. {random.choice(['Check the path exists.', 'Verify permissions.', 'Install the required package.', 'Check your credentials.'])}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "shell_error", "idx": idx}


def generate_unique_shell_timeout(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique shell timeout sample with available tools."""
    timeout_cmds = [
        f"sleep {random_number(60, 3600)}",
        f"find / -name '{random_var_name()}*' 2>/dev/null",
        f"tar -czvf backup_{random_hash()}.tar.gz /",
        f"rsync -avz /home/ {random.choice(NAMES).lower()}@{random_ip()}:/backup/",
        f"wget -r {random_url()}/",
        f"curl -o /dev/null {random_url()}/large_file_{random_hash()}.zip",
        f"dd if=/dev/zero of=test_{random_hash()}.img bs=1M count={random_number(1000, 10000)}",
        f"python -c 'while True: pass'",
        f"yes > /dev/null",
        f"ping {random_ip()}",
        f"tail -f /var/log/{random.choice(['syslog', 'messages', 'auth.log'])}",
        f"watch -n 1 'ps aux | head -20'",
        f"top -b -n {random_number(100, 1000)}",
        f"stress --cpu {random_number(2, 8)} --timeout {random_number(300, 3600)}",
        f"ab -n {random_number(10000, 100000)} -c {random_number(10, 100)} {random_url()}/",
    ]
    
    cmd = random.choice(timeout_cmds)
    timeout_secs = random.choice([30, 60, 120, 300])
    tools_section = maybe_include_tools(t, SHELL_TOOLS, probability=0.5)
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['user_start']}\nRun: {cmd}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n"
        f"{t['exec_timeout']}\n"
        f"The command timed out after {timeout_secs} seconds. "
        f"{random.choice(['I will interrupt it.', 'This is a long-running process.', 'Consider running in background.', 'Use nohup or screen for long tasks.'])}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "shell_timeout", "idx": idx}


def generate_unique_python_script(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique Python script execution sample with available tools."""
    script_templates = [
        # Data processing
        (f"import json\ndata = {{'name': '{random.choice(NAMES)}', 'id': {random_number()}}}\nprint(json.dumps(data, indent=2))",
         f'{{\n  "name": "{random.choice(NAMES)}",\n  "id": {random_number()}\n}}'),
        # Math calculation
        (f"import math\nresult = math.sqrt({random_number(100, 10000)})\nprint(f'Square root: {{result:.4f}}')",
         f"Square root: {random.uniform(10, 100):.4f}"),
        # List operations
        (f"numbers = {list(random.sample(range(1, 100), 5))}\nprint(f'Sum: {{sum(numbers)}}, Avg: {{sum(numbers)/len(numbers):.2f}}')",
         f"Sum: {random_number(50, 300)}, Avg: {random.uniform(20, 80):.2f}"),
        # File operations
        (f"with open('{random_file_path()}', 'w') as f:\n    f.write('{random_var_name()} = {random_number()}')\nprint('File written successfully')",
         "File written successfully"),
        # String manipulation
        (f"text = '{random.choice(NAMES)} is working on {random.choice(FRAMEWORKS)}'\nprint(text.upper())",
         f"{random.choice(NAMES).upper()} IS WORKING ON {random.choice(FRAMEWORKS).upper()}"),
        # Dictionary operations
        (f"users = {{{', '.join([f'\"{n}\": {random_number()}' for n in random.sample(NAMES, 3)])}}}\nprint(f'Total: {{sum(users.values())}}')",
         f"Total: {random_number(100, 500)}"),
        # Date/time
        (f"from datetime import datetime\nprint(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))",
         f"2024-{random_number(1,12):02d}-{random_number(1,28):02d} {random_number(0,23):02d}:{random_number(0,59):02d}:{random_number(0,59):02d}"),
        # Random generation
        (f"import random\nprint([random.randint(1, 100) for _ in range({random_number(3, 8)})])",
         str([random_number(1, 100) for _ in range(random_number(3, 8))])),
        # HTTP request simulation
        (f"# Simulating API call\nresponse = {{'status': {random.choice(HTTP_CODES)}, 'data': {{'id': {random_number()}}}}}\nprint(response)",
         f"{{'status': {random.choice(HTTP_CODES)}, 'data': {{'id': {random_number()}}}}}"),
        # Class definition
        (f"class {random_class_name()}:\n    def __init__(self):\n        self.value = {random_number()}\n\nobj = {random_class_name()}()\nprint(f'Value: {{obj.value}}')",
         f"Value: {random_number()}"),
    ]
    
    code, output = random.choice(script_templates)
    task = random.choice(["Run a data processing script", "Execute Python code", "Run this script", "Process data with Python"])
    tools_section = maybe_include_tools(t, CODE_EXECUTION_TOOLS, probability=0.5)
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['user_start']}\n{task}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll write and execute a Python script:\n\n"
        f"{t['code_start']}{t['lang_python']}\n{code}\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ python script.py\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{output}\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "python_script_execution", "idx": idx}


def generate_unique_jupyter(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique Jupyter notebook execution sample with available tools."""
    cell_templates = [
        # Data analysis
        (f"import pandas as pd\ndf = pd.DataFrame({{'name': {random.sample(NAMES, 3)}, 'value': {[random_number() for _ in range(3)]}}})\ndf.head()",
         f"   name  value\n0  {random.choice(NAMES)}    {random_number()}\n1  {random.choice(NAMES)}    {random_number()}\n2  {random.choice(NAMES)}    {random_number()}"),
        # NumPy operations
        (f"import numpy as np\narr = np.array({[random_number() for _ in range(5)]})\nprint(f'Mean: {{arr.mean():.2f}}, Std: {{arr.std():.2f}}')",
         f"Mean: {random.uniform(100, 500):.2f}, Std: {random.uniform(50, 200):.2f}"),
        # Plotting (output description)
        (f"import matplotlib.pyplot as plt\nplt.plot({[random_number() for _ in range(5)]})\nplt.title('{random_class_name()} Analysis')\nplt.show()",
         f"[Figure displayed: Line plot titled '{random_class_name()} Analysis']"),
        # Statistics
        (f"from statistics import mean, stdev\ndata = {[random_number() for _ in range(10)]}\nprint(f'Mean: {{mean(data):.2f}}, StdDev: {{stdev(data):.2f}}')",
         f"Mean: {random.uniform(200, 600):.2f}, StdDev: {random.uniform(50, 200):.2f}"),
        # Error case
        (f"result = {random_number()} / 0",
         None, "ZeroDivisionError: division by zero"),
        # Dictionary comprehension
        (f"squares = {{x: x**2 for x in range({random_number(5, 10)})}}\nprint(squares)",
         str({x: x**2 for x in range(random_number(5, 10))})),
        # List comprehension
        (f"evens = [x for x in range({random_number(10, 30)}) if x % 2 == 0]\nprint(evens)",
         str([x for x in range(random_number(10, 30)) if x % 2 == 0])),
        # Lambda function
        (f"multiply = lambda x, y: x * y\nprint(multiply({random_number(2, 20)}, {random_number(2, 20)}))",
         str(random_number(4, 400))),
    ]
    
    template = random.choice(cell_templates)
    code = template[0]
    tools_section = maybe_include_tools(t, CODE_EXECUTION_TOOLS, probability=0.5)
    
    if len(template) == 3:  # Error case
        output_section = f"{t['jupyter_error']}{template[2]}{t['jupyter_error_end']}"
    elif template[1]:
        output_section = f"{t['jupyter_output']}{template[1]}{t['jupyter_output_end']}"
    else:
        output_section = t['empty_output']
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['jupyter_start']}\n"
        f"{t['jupyter_code']}\n{code}\n{t['jupyter_code_end']}\n"
        f"{output_section}\n"
        f"{t['jupyter_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "jupyter", "idx": idx}


def generate_unique_multi_step(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique multi-step execution sample with available tools."""
    task_templates = [
        {
            "task": f"Create a new {random.choice(LANGUAGES)} project structure",
            "steps": [
                (f"mkdir -p {random_var_name()}/src {random_var_name()}/tests", ""),
                (f"touch {random_var_name()}/src/__init__.py {random_var_name()}/tests/__init__.py", ""),
                (f"echo 'def main(): print(\"Hello\")' > {random_var_name()}/src/main.py", ""),
                (f"tree {random_var_name()}", f"{random_var_name()}\n├── src\n│   ├── __init__.py\n│   └── main.py\n└── tests\n    └── __init__.py"),
            ]
        },
        {
            "task": "Check system resources",
            "steps": [
                ("df -h | head -5", f"Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       {random_number(50, 500)}G   {random_number(10, 200)}G   {random_number(50, 300)}G  {random_number(10, 90)}% /"),
                ("free -h", f"              total        used        free\nMem:           {random_number(8, 64)}Gi       {random_number(2, 32)}Gi       {random_number(2, 32)}Gi"),
                ("nproc", str(random_number(2, 32))),
                ("uname -a", f"Linux hostname {random_number(4,6)}.{random_number(0,20)}.0-generic x86_64 GNU/Linux"),
            ]
        },
        {
            "task": f"Set up a {random.choice(FRAMEWORKS)} application",
            "steps": [
                (f"mkdir {random_var_name()} && cd {random_var_name()}", ""),
                (f"npm init -y", f'{{"name": "{random_var_name()}", "version": "1.0.0"}}'),
                (f"npm install {random.choice(['express', 'fastify', 'koa', 'hapi'])}", f"added {random_number(50, 200)} packages"),
                ("ls -la", f"total {random_number(10, 50)}\ndrwxr-xr-x  node_modules\n-rw-r--r--  package.json"),
            ]
        },
        {
            "task": f"Deploy {random.choice(TOOLS)} configuration",
            "steps": [
                (f"cat > {random_var_name()}.yaml << 'EOF'\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {random_var_name()}\nEOF", ""),
                (f"kubectl apply -f {random_var_name()}.yaml", f"configmap/{random_var_name()} created"),
                (f"kubectl get configmap {random_var_name()}", f"NAME              DATA   AGE\n{random_var_name()}   1      {random_number(1, 60)}s"),
            ]
        },
        {
            "task": f"Set up {random.choice(DATABASES)} database",
            "steps": [
                (f"createdb {random_var_name()}", ""),
                (f"psql -d {random_var_name()} -c 'CREATE TABLE {random.choice(NOUNS)}s (id SERIAL PRIMARY KEY, name VARCHAR(100));'", "CREATE TABLE"),
                (f"psql -d {random_var_name()} -c '\\dt'", f"         List of relations\n Schema |    Name    | Type  | Owner\n--------+------------+-------+-------\n public | {random.choice(NOUNS)}s | table | user"),
            ]
        },
    ]
    
    template = random.choice(task_templates)
    tools_section = maybe_include_tools(t, FULL_TOOLS, probability=0.5)
    
    steps_text = ""
    for cmd, output in template["steps"]:
        output_text = output if output else "(no output)"
        steps_text += f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n{t['exec_result']}\n{output_text}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{tools_section}"
        f"{t['user_start']}\n{template['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {template['task'].lower()}. Let me execute the necessary commands:\n\n"
        f"{steps_text}\n"
        f"Done! All steps completed successfully.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "multi_step_execution", "idx": idx}


def generate_unique_file_ops(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique file operation sample."""
    op_type = random.choice(["add", "edit", "delete", "rename", "copy", "move"])
    
    path1 = random_file_path()
    path2 = random_file_path()
    content = f"def {random_func_name()}():\n    {random_var_name()} = {random_number()}\n    return {random_var_name()}"
    old_content = f"{random_var_name()} = {random_number()}"
    new_content = f"{random_var_name()} = {random_number()}"
    
    if op_type == "add":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nCreate a new file with {random.choice(['helper functions', 'utility code', 'configuration', 'constants'])}.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['add_file']}\n"
            f"{t['filepath_start']}{path1}{t['filepath_end']}\n"
            f"{t['file_content']}\n{content}\n{t['file_content_end']}\n"
            f"{t['add_file_end']}\n"
            f"Created new file at {path1}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    elif op_type == "edit":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nUpdate the {random.choice(['configuration', 'value', 'setting', 'constant'])} in {path1}.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['edit_file']}\n"
            f"{t['filepath_start']}{path1}{t['filepath_end']}\n"
            f"{t['replace']}\n{old_content}\n{t['replace_end']}\n"
            f"{t['file_content']}\n{new_content}\n{t['file_content_end']}\n"
            f"{t['edit_file_end']}\n"
            f"Updated {path1}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    elif op_type == "delete":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nRemove the deprecated file {path1}.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['delete_file']}\n"
            f"{t['filepath_start']}{path1}{t['filepath_end']}\n"
            f"{t['delete_file_end']}\n"
            f"Deleted {path1}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    elif op_type == "rename":
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\nRename {path1} to {path2}.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['exec_start']}\n$ mv {path1} {path2}\n{t['exec_end']}\n"
            f"{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
            f"Renamed {path1} to {path2}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    else:  # copy or move
        action = "cp" if op_type == "copy" else "mv"
        verb = "Copied" if op_type == "copy" else "Moved"
        text = (
            f"{t['bos']}"
            f"{t['user_start']}\n{verb.replace('ed', '')} {path1} to {path2}.\n{t['user_end']}"
            f"{t['assistant_start']}\n"
            f"{t['exec_start']}\n$ {action} {path1} {path2}\n{t['exec_end']}\n"
            f"{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
            f"{verb} {path1} to {path2}\n"
            f"{t['assistant_end']}"
            f"{t['eos']}"
        )
    
    return {"text": text, "type": "file_ops", "idx": idx}


def generate_unique_edit_lines(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique line editing sample."""
    path = random_file_path()
    start_line = random_number(1, 50)
    end_line = start_line + random_number(1, 10)
    
    var1 = random_var_name()
    var2 = random_var_name()
    val1 = random_number()
    val2 = random_number()
    multiplier = random_number(2, 10)
    
    old_code = f"    {var1} = {val1}\n    return {var1}"
    new_code = f"    {var2} = {val2}\n    # Updated logic\n    result = {var2} * {multiplier}\n    return result"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nUpdate lines {start_line}-{end_line} in {path} to improve the {random.choice(['logic', 'calculation', 'processing', 'handling'])}.\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['edit_file']}\n"
        f"{t['filepath_start']}{path}{t['filepath_end']}\n"
        f"Lines: {start_line}-{end_line}\n"
        f"{t['replace']}\n{old_code}\n{t['replace_end']}\n"
        f"{t['file_content']}\n{new_code}\n{t['file_content_end']}\n"
        f"{t['edit_file_end']}\n"
        f"Updated lines {start_line}-{end_line} in {path}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "edit_lines", "idx": idx}


def generate_unique_debugging(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique debugging sample."""
    error_type = random.choice(ERROR_TYPES)
    func_name = random_func_name()
    var_name = random_var_name()
    value = random_number()
    
    debug_scenarios = {
        "TypeError": {
            "code": f"def {func_name}(data):\n    return data['{var_name}'] * 2\n\nresult = {func_name}(None)",
            "error": f"TypeError: 'NoneType' object is not subscriptable",
            "fix": f"def {func_name}(data):\n    if data is None:\n        return 0\n    return data.get('{var_name}', 0) * 2",
            "explanation": f"The function received None instead of a dictionary. Added None check and used .get() for safe access."
        },
        "KeyError": {
            "code": f"data = {{'a': 1, 'b': 2}}\nvalue = data['{var_name}']",
            "error": f"KeyError: '{var_name}'",
            "fix": f"data = {{'a': 1, 'b': 2}}\nvalue = data.get('{var_name}', {value})  # Default value if key missing",
            "explanation": f"Key '{var_name}' doesn't exist. Use .get() with a default value."
        },
        "IndexError": {
            "code": f"items = [{', '.join(str(random_number()) for _ in range(3))}]\nprint(items[{random_number(10, 20)}])",
            "error": f"IndexError: list index out of range",
            "fix": f"items = [{', '.join(str(random_number()) for _ in range(3))}]\nidx = {random_number(10, 20)}\nif idx < len(items):\n    print(items[idx])\nelse:\n    print('Index out of range')",
            "explanation": "List index exceeds length. Added bounds checking."
        },
        "ValueError": {
            "code": f"number = int('{random.choice(NAMES)}')",
            "error": f"ValueError: invalid literal for int() with base 10: '{random.choice(NAMES)}'",
            "fix": f"text = '{random.choice(NAMES)}'\ntry:\n    number = int(text)\nexcept ValueError:\n    number = 0\n    print(f'Cannot convert {{text}} to int')",
            "explanation": "Cannot convert non-numeric string to int. Added try/except."
        },
        "ZeroDivisionError": {
            "code": f"def {func_name}(a, b):\n    return a / b\n\nresult = {func_name}({value}, 0)",
            "error": "ZeroDivisionError: division by zero",
            "fix": f"def {func_name}(a, b):\n    if b == 0:\n        return float('inf')\n    return a / b",
            "explanation": "Division by zero. Added check for zero divisor."
        },
        "FileNotFoundError": {
            "code": f"with open('{random_file_path()}') as f:\n    content = f.read()",
            "error": f"FileNotFoundError: [Errno 2] No such file or directory: '{random_file_path()}'",
            "fix": f"import os\npath = '{random_file_path()}'\nif os.path.exists(path):\n    with open(path) as f:\n        content = f.read()\nelse:\n    content = ''",
            "explanation": "File doesn't exist. Added existence check."
        },
        "AttributeError": {
            "code": f"{var_name} = None\nprint({var_name}.upper())",
            "error": f"AttributeError: 'NoneType' object has no attribute 'upper'",
            "fix": f"{var_name} = None\nif {var_name} is not None:\n    print({var_name}.upper())\nelse:\n    print('Value is None')",
            "explanation": "Called method on None. Added None check."
        },
        "ImportError": {
            "code": f"from {random_hash()} import {func_name}",
            "error": f"ModuleNotFoundError: No module named '{random_hash()}'",
            "fix": f"try:\n    from {random_hash()} import {func_name}\nexcept ImportError:\n    print('Module not installed. Run: pip install {random_hash()}')\n    {func_name} = lambda: None",
            "explanation": "Module not installed. Added try/except with helpful message."
        },
    }
    
    scenario = debug_scenarios.get(error_type, debug_scenarios["TypeError"])
    
    prompts = [
        "My Python script is crashing. Can you help debug it?",
        "I'm getting an error with this code. What's wrong?",
        "This code throws an exception. How do I fix it?",
        "Debug this code for me:",
        f"Getting {error_type}, help me fix it:",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(prompts)}\n"
        f"{t['code_start']}{t['lang_python']}\n{scenario['code']}\n{t['code_end']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"Let me analyze and fix this error:\n\n"
        f"{t['exec_start']}\n$ python debug.py\n{t['exec_end']}\n"
        f"{t['exec_error']}\n{scenario['error']}\n{t['exec_error_end']}\n\n"
        f"**Problem:** {scenario['explanation']}\n\n"
        f"**Fixed code:**\n"
        f"{t['code_start']}{t['lang_python']}\n{scenario['fix']}\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ python debug_fixed.py\n{t['exec_end']}\n"
        f"{t['exec_result']}\n(runs successfully)\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "interactive_debugging", "idx": idx}


def generate_unique_diff(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique diff sample."""
    func_name = random_func_name()
    var_name = random_var_name()
    class_name = random_class_name()
    
    diff_types = [
        {
            "context": [f"def {func_name}(data):", "    result = []", "    for item in data:", "        result.append(item)", "    return result"],
            "adds": [f"    if not data:", "        return []", f"        if item.{var_name} is not None:"],
            "explanation": f"Added input validation and None check for {var_name}."
        },
        {
            "context": [f"class {class_name}:", f"    def __init__(self, {var_name}):", f"        self.{var_name} = {var_name}"],
            "adds": [f"        self.created_at = datetime.now()", f"        self.id = uuid.uuid4()"],
            "explanation": f"Added timestamp and UUID to {class_name}."
        },
        {
            "context": [f"async def {func_name}(url):", "    response = await client.get(url)", "    return response.json()"],
            "adds": ["    try:", "    except Exception as e:", f"        logger.error(f'{func_name} failed: {{e}}')", "        return None"],
            "explanation": f"Added error handling to {func_name}."
        },
        {
            "context": [f"@app.route('/api/{random.choice(NOUNS)}s')", f"def get_{random.choice(NOUNS)}s():", f"    items = {class_name}.query.all()", "    return jsonify(items)"],
            "adds": ["@login_required", f"    page = request.args.get('page', 1, type=int)", f"    items = {class_name}.query.paginate(page=page, per_page=20)"],
            "explanation": "Added authentication and pagination."
        },
        {
            "context": [f"def {func_name}(config_path):", "    with open(config_path) as f:", "        return json.load(f)"],
            "adds": ["    if not os.path.exists(config_path):", "        return {}", f"        config = json.load(f)", "        return validate_config(config)"],
            "explanation": "Added file existence check and config validation."
        },
    ]
    
    template = random.choice(diff_types)
    
    diff_lines = []
    add_idx = 0
    for line in template["context"]:
        diff_lines.append(f"{t['diff_context']} {line}")
        if add_idx < len(template["adds"]) and random.random() > 0.3:
            diff_lines.append(f"{t['diff_add']} {template['adds'][add_idx]}")
            add_idx += 1
    
    while add_idx < len(template["adds"]):
        diff_lines.append(f"{t['diff_add']} {template['adds'][add_idx]}")
        add_idx += 1
    
    diff_text = f"{t['diff_start']}\n" + "\n".join(diff_lines) + f"\n{t['diff_end']}"
    
    prompts = [
        "Review this diff and explain the changes:",
        "What does this diff do?",
        "Explain the modifications in this patch:",
        "Summarize the changes:",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(prompts)}\n{diff_text}\n{t['user_end']}"
        f"{t['assistant_start']}\n{template['explanation']}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "diff", "idx": idx}


def generate_unique_commit(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique commit sample."""
    lang = random.choice(["python", "javascript", "typescript", "go", "rust"])
    func_name = random_func_name()
    var_name = random_var_name()
    
    commit_types = ["feat", "fix", "refactor", "docs", "test", "chore", "perf"]
    commit_type = random.choice(commit_types)
    
    code_templates = {
        "python": {
            "before": f"def {func_name}():\n    pass  # TODO: implement",
            "after": f"def {func_name}():\n    {var_name} = {random_number()}\n    return {var_name} * 2",
        },
        "javascript": {
            "before": f"function {func_name}() {{\n    // TODO: implement\n}}",
            "after": f"function {func_name}() {{\n    const {var_name} = {random_number()};\n    return {var_name} * 2;\n}}",
        },
        "typescript": {
            "before": f"function {func_name}(): number {{\n    // TODO: implement\n    return 0;\n}}",
            "after": f"function {func_name}(): number {{\n    const {var_name}: number = {random_number()};\n    return {var_name} * 2;\n}}",
        },
        "go": {
            "before": f"func {func_name.title().replace('_', '')}() int {{\n    // TODO: implement\n    return 0\n}}",
            "after": f"func {func_name.title().replace('_', '')}() int {{\n    {var_name} := {random_number()}\n    return {var_name} * 2\n}}",
        },
        "rust": {
            "before": f"fn {func_name}() -> i32 {{\n    // TODO: implement\n    0\n}}",
            "after": f"fn {func_name}() -> i32 {{\n    let {var_name} = {random_number()};\n    {var_name} * 2\n}}",
        },
    }
    
    template = code_templates[lang]
    
    commit_messages = [
        f"{commit_type}: implement {func_name} function",
        f"{commit_type}({random.choice(NOUNS)}): add {func_name} logic",
        f"{commit_type}: complete {func_name} implementation",
        f"{commit_type}: add {var_name} calculation to {func_name}",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['commit_before']}\n{template['before']}\n{t['commit_before_end']}"
        f"{t['commit_after']}\n{template['after']}\n{t['commit_after_end']}"
        f"{t['commit_msg']}{random.choice(commit_messages)}{t['commit_msg_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "commit", "language": lang, "idx": idx}


def generate_unique_fim(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique fill-in-the-middle sample."""
    lang = random.choice(["python", "javascript", "typescript", "go", "rust"])
    func_name = random_func_name()
    var_name = random_var_name()
    value = random_number()
    
    fim_templates = {
        "python": {
            "prefix": f"def {func_name}(items):\n    ",
            "middle": f"result = []\n    for item in items:\n        if item > {value}:\n            result.append(item)",
            "suffix": f"\n    return result\n\nprint({func_name}([{', '.join(str(random_number()) for _ in range(5))}]))",
        },
        "javascript": {
            "prefix": f"function {func_name}(items) {{\n    ",
            "middle": f"return items.filter(item => item > {value})",
            "suffix": f";\n}}\n\nconsole.log({func_name}([{', '.join(str(random_number()) for _ in range(5))}]));",
        },
        "typescript": {
            "prefix": f"function {func_name}(items: number[]): number[] {{\n    ",
            "middle": f"return items.filter((item: number) => item > {value})",
            "suffix": f";\n}}\n\nconsole.log({func_name}([{', '.join(str(random_number()) for _ in range(5))}]));",
        },
        "go": {
            "prefix": f"func {func_name.title().replace('_', '')}(items []int) []int {{\n    ",
            "middle": f"var result []int\n    for _, item := range items {{\n        if item > {value} {{\n            result = append(result, item)\n        }}\n    }}",
            "suffix": f"\n    return result\n}}",
        },
        "rust": {
            "prefix": f"fn {func_name}(items: Vec<i32>) -> Vec<i32> {{\n    ",
            "middle": f"items.into_iter().filter(|&x| x > {value}).collect()",
            "suffix": f"\n}}",
        },
    }
    
    template = fim_templates[lang]
    
    text = (
        f"{t['bos']}"
        f"{t['fim_prefix']}{template['prefix']}"
        f"{t['fim_suffix']}{template['suffix']}"
        f"{t['fim_middle']}{template['middle']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "fim", "language": lang, "idx": idx}


def generate_unique_issue(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique GitHub issue sample."""
    issue_types = ["bug", "feature", "performance", "security", "documentation"]
    issue_type = random.choice(issue_types)
    func_name = random_func_name()
    component = random.choice(NOUNS)
    
    issue_templates = {
        "bug": {
            "title": f"Bug: {func_name} crashes with {random.choice(ERROR_TYPES)}",
            "body": f"**Description**\nThe {func_name} function crashes when processing {component} data.\n\n**Steps to reproduce**\n1. Call {func_name}() with invalid input\n2. Observe the {random.choice(ERROR_TYPES)}\n\n**Expected behavior**\nShould handle invalid input gracefully.\n\n**Environment**\n- Python {random_version()}\n- OS: Ubuntu {random_number(18, 24)}.04",
        },
        "feature": {
            "title": f"Feature: Add {func_name} support for {component}",
            "body": f"**Description**\nIt would be useful to have {func_name} functionality for {component} management.\n\n**Use case**\nUsers need to {random.choice(VERBS)} {component}s efficiently.\n\n**Proposed solution**\nImplement a new {func_name} method in the {random_class_name()} class.",
        },
        "performance": {
            "title": f"Performance: {func_name} is slow with large {component} datasets",
            "body": f"**Description**\nThe {func_name} function takes {random_number(10, 60)} seconds for {random_number(10000, 100000)} {component}s.\n\n**Profiling results**\n- {random_number(60, 90)}% time spent in database queries\n- Memory usage: {random_number(500, 2000)}MB\n\n**Suggested optimization**\nAdd caching and batch processing.",
        },
        "security": {
            "title": f"Security: {func_name} vulnerable to {random.choice(['SQL injection', 'XSS', 'CSRF', 'path traversal'])}",
            "body": f"**Description**\nThe {func_name} endpoint doesn't properly sanitize {component} input.\n\n**Impact**\nAttackers could potentially access unauthorized data.\n\n**Recommendation**\nAdd input validation and parameterized queries.",
        },
        "documentation": {
            "title": f"Docs: Missing documentation for {func_name}",
            "body": f"**Description**\nThe {func_name} function lacks proper documentation.\n\n**Missing items**\n- Parameter descriptions\n- Return value documentation\n- Usage examples\n\n**Suggested content**\nAdd docstrings and update README.",
        },
    }
    
    template = issue_templates[issue_type]
    is_closed = random.choice([True, False])
    status = t['issue_closed'] if is_closed else t['issue_open']
    
    text = (
        f"{t['bos']}"
        f"{t['issue_start']}\n"
        f"{status}\n"
        f"{t['issue_title']}{template['title']}{t['issue_title_end']}\n"
        f"{t['issue_body']}{template['body']}{t['issue_body_end']}\n"
        f"{t['issue_end']}"
        f"{t['user_start']}\nSummarize this issue and suggest next steps.\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"**Summary**: This is a {issue_type} issue regarding {func_name}.\n\n"
        f"**Next steps**:\n"
        f"1. {'Review the fix' if is_closed else 'Investigate the root cause'}\n"
        f"2. {'Verify the solution' if is_closed else 'Create a test case'}\n"
        f"3. {'Close if verified' if is_closed else 'Implement and test the fix'}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "issue", "idx": idx}


def generate_unique_repo_context(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique repository context sample."""
    org = random.choice(["facebook", "google", "microsoft", "apache", "mozilla", "rust-lang", "golang", "python", "nodejs", "kubernetes"])
    repo = random.choice(["react", "tensorflow", "vscode", "kafka", "firefox", "rust", "go", "cpython", "node", "kubernetes"])
    stars = random_number(10000, 200000)
    lang = random.choice(LANGUAGES)
    branch = random.choice(["main", "master", "develop", "dev"])
    
    questions = [
        (f"What is this repository about?", f"**{repo}** is a {random.choice(['popular', 'widely-used', 'well-known'])} {lang} project maintained by {org}. It has {stars:,} stars on GitHub, indicating strong community interest."),
        (f"What language is this project written in?", f"This project ({org}/{repo}) is primarily written in **{lang}**. It's one of the most starred {lang} repositories."),
        (f"How popular is this repository?", f"**{org}/{repo}** has {stars:,} stars, making it one of the top repositories in its category. It's actively maintained with regular updates."),
        (f"Who maintains this project?", f"This repository is maintained by **{org}**. They have a team of {random_number(10, 100)} active contributors."),
        (f"What's the main branch?", f"The main development branch is **{branch}**. PRs should be submitted against this branch."),
        (f"How can I contribute?", f"To contribute to {org}/{repo}:\n1. Fork the repository\n2. Create a feature branch\n3. Make your changes\n4. Submit a PR to {branch}"),
    ]
    
    question, answer = random.choice(questions)
    
    text = (
        f"{t['bos']}"
        f"{t['reponame']}{org}/{repo}{t['reponame_end']}"
        f"{t['gh_stars']}{stars}{t['gh_stars_end']}"
        f"{t['branch']}{branch}{t['branch_end']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n{answer}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "repo_context", "idx": idx}


def generate_unique_execution(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique code execution sample."""
    lang = random.choice(["python", "javascript", "bash"])
    func_name = random_func_name()
    var_name = random_var_name()
    values = [random_number() for _ in range(random_number(3, 7))]
    
    if lang == "python":
        code = f"# {random_class_name()} calculation\n{var_name} = {values}\nresult = sum({var_name}) / len({var_name})\nprint(f'Average: {{result:.2f}}')"
        output = f"Average: {sum(values) / len(values):.2f}"
    elif lang == "javascript":
        code = f"// {random_class_name()} calculation\nconst {var_name} = {values};\nconst avg = {var_name}.reduce((a, b) => a + b) / {var_name}.length;\nconsole.log(`Average: ${{avg.toFixed(2)}}`);"
        output = f"Average: {sum(values) / len(values):.2f}"
    else:
        code = f"#!/bin/bash\n# {random_class_name()} script\n{var_name}=({' '.join(map(str, values))})\necho \"Count: ${{#{var_name}[@]}}\""
        output = f"Count: {len(values)}"
    
    lang_token = {"python": t['lang_python'], "javascript": t['lang_javascript'], "bash": t['lang_bash']}[lang]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nRun this {lang} code:\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['code_start']}{lang_token}\n{code}\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ {['python', 'node', 'bash'][['python', 'javascript', 'bash'].index(lang)]} script.{'py' if lang == 'python' else 'js' if lang == 'javascript' else 'sh'}\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{output}\n{t['exec_result_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "code_execution", "language": lang, "idx": idx}


# =============================================================================
# SYSTEM ADMIN GENERATORS
# =============================================================================

def generate_unique_apt_install(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique apt install sample."""
    packages = random.sample(APT_PACKAGES, random.randint(1, 5))
    pkg_str = " ".join(packages)
    version = random_version()
    
    tasks = [
        f"Install {packages[0]}",
        f"Set up {packages[0]} and related tools",
        f"Install {' and '.join(packages[:2]) if len(packages) > 1 else packages[0]}",
        f"Get {packages[0]} installed on this system",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(tasks)}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll install the requested packages:\n\n"
        f"{t['exec_start']}\n$ sudo apt update\n{t['exec_end']}\n"
        f"{t['exec_result']}\nHit:1 http://archive.ubuntu.com/ubuntu jammy InRelease\nReading package lists... Done\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ sudo apt install -y {pkg_str}\n{t['exec_end']}\n"
        f"{t['exec_result']}\nReading package lists... Done\nSetting up {packages[0]} ({version}) ...\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ {packages[0]} --version\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{packages[0]} {version}\n{t['exec_result_end']}\n\n"
        f"Done! {packages[0]} has been installed successfully.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "apt_install", "idx": idx}


def generate_unique_docker(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique Docker sample."""
    image = random.choice(["nginx", "redis", "postgres", "mysql", "mongo", "node", "python", "golang", "rust", "ubuntu"])
    tag = random.choice(["latest", "alpine", "slim", random_version()])
    container_name = f"{random_var_name()}_{random_number(100, 999)}"
    port_host = random_port()
    port_container = random.choice([80, 443, 3000, 5000, 5432, 6379, 8080, 27017])
    
    docker_cmds = [
        {
            "task": f"Run a {image} container",
            "steps": [
                (f"docker pull {image}:{tag}", f"{tag}: Pulling from library/{image}\nDigest: sha256:{random_hash()}\nStatus: Downloaded newer image for {image}:{tag}"),
                (f"docker run -d --name {container_name} -p {port_host}:{port_container} {image}:{tag}", random_hash()),
                (f"docker ps | grep {container_name}", f"{random_hash()[:12]}   {image}:{tag}   \"{image}\"   {random_number(1, 60)} seconds ago   Up {random_number(1, 60)} seconds   0.0.0.0:{port_host}->{port_container}/tcp   {container_name}"),
            ]
        },
        {
            "task": f"Build a Docker image",
            "steps": [
                (f"cat > Dockerfile << 'EOF'\nFROM {image}:{tag}\nWORKDIR /app\nCOPY . .\nCMD [\"{image}\", \"--version\"]\nEOF", ""),
                (f"docker build -t {container_name}:latest .", f"Step 1/4 : FROM {image}:{tag}\n ---> {random_hash()[:12]}\nSuccessfully built {random_hash()[:12]}\nSuccessfully tagged {container_name}:latest"),
                (f"docker images | grep {container_name}", f"{container_name}   latest    {random_hash()[:12]}   {random_number(1, 60)} seconds ago   {random_number(50, 500)}MB"),
            ]
        },
        {
            "task": f"Check Docker container logs",
            "steps": [
                (f"docker logs {container_name}", f"[{random_number(2020, 2024)}-{random_number(1,12):02d}-{random_number(1,28):02d}] Starting {image}...\n[INFO] Listening on port {port_container}"),
                (f"docker stats {container_name} --no-stream", f"CONTAINER ID   NAME              CPU %     MEM USAGE / LIMIT\n{random_hash()[:12]}   {container_name}   {random.uniform(0.1, 5.0):.2f}%    {random_number(10, 500)}MiB / {random_number(1, 16)}GiB"),
            ]
        },
        {
            "task": f"Stop and remove Docker container",
            "steps": [
                (f"docker stop {container_name}", container_name),
                (f"docker rm {container_name}", container_name),
                (f"docker ps -a | grep {container_name} || echo 'Container removed'", "Container removed"),
            ]
        },
    ]
    
    template = random.choice(docker_cmds)
    
    steps_text = ""
    for cmd, output in template["steps"]:
        output_text = output if output else "(no output)"
        steps_text += f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n{t['exec_result']}\n{output_text}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{template['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {template['task'].lower()}:\n\n"
        f"{steps_text}\n"
        f"Done!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "docker", "idx": idx}


def generate_unique_database_setup(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique database setup sample."""
    db_type = random.choice(["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite"])
    db_name = random_var_name()
    user_name = random.choice(NAMES).lower()
    table_name = random.choice(NOUNS) + "s"
    
    db_configs = {
        "PostgreSQL": {
            "install": "sudo apt install -y postgresql postgresql-contrib",
            "start": "sudo systemctl start postgresql",
            "create_db": f"sudo -u postgres createdb {db_name}",
            "create_user": f"sudo -u postgres psql -c \"CREATE USER {user_name} WITH PASSWORD 'password123';\"",
            "grant": f"sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user_name};\"",
            "verify": f"psql -U {user_name} -d {db_name} -c '\\dt'",
            "version": f"psql --version",
            "version_output": f"psql (PostgreSQL) {random_number(12, 16)}.{random_number(0, 9)}",
        },
        "MySQL": {
            "install": "sudo apt install -y mysql-server",
            "start": "sudo systemctl start mysql",
            "create_db": f"sudo mysql -e \"CREATE DATABASE {db_name};\"",
            "create_user": f"sudo mysql -e \"CREATE USER '{user_name}'@'localhost' IDENTIFIED BY 'password123';\"",
            "grant": f"sudo mysql -e \"GRANT ALL PRIVILEGES ON {db_name}.* TO '{user_name}'@'localhost';\"",
            "verify": f"mysql -u {user_name} -p {db_name} -e 'SHOW TABLES;'",
            "version": "mysql --version",
            "version_output": f"mysql  Ver {random_number(8, 9)}.{random_number(0, 35)} for Linux on x86_64",
        },
        "MongoDB": {
            "install": "sudo apt install -y mongodb",
            "start": "sudo systemctl start mongodb",
            "create_db": f"mongosh --eval 'use {db_name}'",
            "create_user": f"mongosh {db_name} --eval 'db.createUser({{user: \"{user_name}\", pwd: \"password123\", roles: [\"readWrite\"]}})'",
            "grant": "",
            "verify": f"mongosh {db_name} --eval 'db.getCollectionNames()'",
            "version": "mongosh --version",
            "version_output": f"{random_number(1, 2)}.{random_number(0, 9)}.{random_number(0, 9)}",
        },
        "Redis": {
            "install": "sudo apt install -y redis-server",
            "start": "sudo systemctl start redis-server",
            "create_db": "redis-cli CONFIG SET databases 32",
            "create_user": f"redis-cli ACL SETUSER {user_name} on >{random_hash()} ~* +@all",
            "grant": "",
            "verify": "redis-cli PING",
            "version": "redis-server --version",
            "version_output": f"Redis server v={random_number(6, 7)}.{random_number(0, 9)}.{random_number(0, 9)}",
        },
        "SQLite": {
            "install": "sudo apt install -y sqlite3",
            "start": "",
            "create_db": f"sqlite3 {db_name}.db 'CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, name TEXT);'",
            "create_user": "",
            "grant": "",
            "verify": f"sqlite3 {db_name}.db '.tables'",
            "version": "sqlite3 --version",
            "version_output": f"3.{random_number(30, 45)}.{random_number(0, 9)}",
        },
    }
    
    config = db_configs[db_type]
    
    steps_text = ""
    steps_text += f"{t['exec_start']}\n$ {config['install']}\n{t['exec_end']}\n{t['exec_result']}\nSetting up {db_type.lower()}...\n{t['exec_result_end']}\n"
    
    if config['start']:
        steps_text += f"{t['exec_start']}\n$ {config['start']}\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    
    steps_text += f"{t['exec_start']}\n$ {config['create_db']}\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    
    if config['create_user']:
        steps_text += f"{t['exec_start']}\n$ {config['create_user']}\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    
    steps_text += f"{t['exec_start']}\n$ {config['version']}\n{t['exec_end']}\n{t['exec_result']}\n{config['version_output']}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nInstall and configure {db_type}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you set up {db_type}:\n\n"
        f"{steps_text}\n"
        f"Done! {db_type} is now installed and configured.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "database_setup", "idx": idx}


def generate_unique_webserver(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique webserver setup sample."""
    server = random.choice(["Nginx", "Apache", "Caddy"])
    # Generate unique domain
    domain = f"{random_var_name()}.{random.choice(['com', 'io', 'dev', 'app', 'net', 'org'])}"
    port = random_number(80, 9000)
    
    server_configs = {
        "Nginx": {
            "install": "sudo apt install -y nginx",
            "config": f"server {{\n    listen {port};\n    server_name {domain};\n    root /var/www/{domain};\n    index index.html;\n}}",
            "config_path": f"/etc/nginx/sites-available/{domain}",
            "enable": f"sudo ln -s /etc/nginx/sites-available/{domain} /etc/nginx/sites-enabled/",
            "test": "sudo nginx -t",
            "restart": "sudo systemctl restart nginx",
        },
        "Apache": {
            "install": "sudo apt install -y apache2",
            "config": f"<VirtualHost *:{port}>\n    ServerName {domain}\n    DocumentRoot /var/www/{domain}\n</VirtualHost>",
            "config_path": f"/etc/apache2/sites-available/{domain}.conf",
            "enable": f"sudo a2ensite {domain}.conf",
            "test": "sudo apache2ctl configtest",
            "restart": "sudo systemctl restart apache2",
        },
        "Caddy": {
            "install": "sudo apt install -y caddy",
            "config": f"{domain} {{\n    root * /var/www/{domain}\n    file_server\n}}",
            "config_path": "/etc/caddy/Caddyfile",
            "enable": "",
            "test": "caddy validate --config /etc/caddy/Caddyfile",
            "restart": "sudo systemctl restart caddy",
        },
    }
    
    config = server_configs[server]
    
    steps_text = ""
    steps_text += f"{t['exec_start']}\n$ {config['install']}\n{t['exec_end']}\n{t['exec_result']}\nSetting up {server.lower()}...\n{t['exec_result_end']}\n"
    steps_text += f"{t['exec_start']}\n$ cat > {config['config_path']} << 'EOF'\n{config['config']}\nEOF\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    
    if config['enable']:
        steps_text += f"{t['exec_start']}\n$ {config['enable']}\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    
    steps_text += f"{t['exec_start']}\n$ {config['test']}\n{t['exec_end']}\n{t['exec_result']}\nSyntax OK\n{t['exec_result_end']}\n"
    steps_text += f"{t['exec_start']}\n$ {config['restart']}\n{t['exec_end']}\n{t['exec_result']}\n(no output)\n{t['exec_result_end']}\n"
    steps_text += f"{t['exec_start']}\n$ curl -I http://localhost:{port}\n{t['exec_end']}\n{t['exec_result']}\nHTTP/1.1 200 OK\nServer: {server}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nSet up {server} for {domain}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll configure {server} for {domain}:\n\n"
        f"{steps_text}\n"
        f"Done! {server} is now serving {domain}.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "webserver_setup", "idx": idx}


def generate_unique_ssh_setup(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique SSH setup sample."""
    user = random.choice(NAMES).lower()
    host = random_ip()
    key_type = random.choice(["ed25519", "rsa", "ecdsa"])
    
    ssh_tasks = [
        {
            "task": "Generate SSH key pair",
            "steps": [
                (f"ssh-keygen -t {key_type} -C '{user}@example.com' -f ~/.ssh/id_{key_type} -N ''", f"Generating public/private {key_type} key pair.\nYour identification has been saved in /home/{user}/.ssh/id_{key_type}"),
                (f"chmod 600 ~/.ssh/id_{key_type}", ""),
                (f"cat ~/.ssh/id_{key_type}.pub", f"ssh-{key_type} AAAA{random_hash()}{random_hash()} {user}@example.com"),
            ]
        },
        {
            "task": f"Copy SSH key to {host}",
            "steps": [
                (f"ssh-copy-id -i ~/.ssh/id_{key_type}.pub {user}@{host}", f"/usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s)\nNumber of key(s) added: 1"),
                (f"ssh {user}@{host} 'hostname'", f"server-{random_number(1, 100)}"),
            ]
        },
        {
            "task": "Configure SSH client",
            "steps": [
                (f"cat >> ~/.ssh/config << 'EOF'\nHost {random_var_name()}\n    HostName {host}\n    User {user}\n    IdentityFile ~/.ssh/id_{key_type}\nEOF", ""),
                ("chmod 600 ~/.ssh/config", ""),
                (f"ssh {random_var_name()} 'uptime'", f" {random_number(10, 23)}:{random_number(10, 59)}:{random_number(10, 59)} up {random_number(1, 100)} days"),
            ]
        },
        {
            "task": "Set up SSH tunnel",
            "steps": [
                (f"ssh -L {random_port()}:localhost:{random.choice([5432, 3306, 6379, 27017])} -N -f {user}@{host}", ""),
                (f"ss -tlnp | grep {random_port()}", f"LISTEN  0  128  127.0.0.1:{random_port()}  0.0.0.0:*"),
            ]
        },
    ]
    
    template = random.choice(ssh_tasks)
    
    steps_text = ""
    for cmd, output in template["steps"]:
        output_text = output if output else "(no output)"
        steps_text += f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n{t['exec_result']}\n{output_text}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{template['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {template['task'].lower()}:\n\n"
        f"{steps_text}\n"
        f"Done!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "ssh_setup", "idx": idx}


def generate_unique_monitoring(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique system monitoring sample."""
    monitoring_tasks = [
        {
            "task": "Check CPU and memory usage",
            "steps": [
                ("top -bn1 | head -15", f"top - {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)} up {random_number(1,100)} days\nTasks: {random_number(100,300)} total\n%Cpu(s): {random.uniform(1, 50):.1f} us, {random.uniform(0.1, 10):.1f} sy\nMiB Mem: {random_number(8000, 64000)} total, {random_number(1000, 30000)} free"),
                ("free -h", f"              total        used        free\nMem:           {random_number(8, 64)}Gi       {random_number(2, 32)}Gi       {random_number(2, 32)}Gi"),
                (f"vmstat 1 3", f"procs -----------memory---------- ---swap--\n r  b   swpd   free   buff  cache\n {random_number(0,5)}  {random_number(0,2)}      0 {random_number(100000, 9999999)} {random_number(10000, 999999)} {random_number(100000, 9999999)}"),
            ]
        },
        {
            "task": "Monitor disk usage",
            "steps": [
                ("df -h", f"Filesystem      Size  Used Avail Use% Mounted on\n/dev/sda1       {random_number(50, 500)}G   {random_number(10, 200)}G   {random_number(50, 300)}G  {random_number(10, 90)}% /"),
                (f"du -sh /var/log/*", f"{random_number(1, 100)}M\t/var/log/syslog\n{random_number(1, 50)}M\t/var/log/auth.log"),
                ("iostat -x 1 3", f"Device            r/s     w/s     rkB/s     wkB/s\nsda              {random.uniform(0.1, 100):.2f}    {random.uniform(0.1, 100):.2f}    {random.uniform(1, 1000):.2f}    {random.uniform(1, 1000):.2f}"),
            ]
        },
        {
            "task": "Check network connections",
            "steps": [
                ("netstat -tuln | head -20", f"Active Internet connections (only servers)\nProto Recv-Q Send-Q Local Address           Foreign Address         State\ntcp        0      0 0.0.0.0:{random.choice([22, 80, 443, 3000])}              0.0.0.0:*               LISTEN"),
                ("ss -s", f"Total: {random_number(100, 500)}\nTCP:   {random_number(10, 100)} (estab {random_number(1, 50)}, closed {random_number(0, 20)})"),
                (f"iftop -t -s 3 2>/dev/null || echo 'Install: apt install iftop'", f"   {random_ip()} => {random_ip()}  {random.uniform(0.1, 10):.2f}Mb"),
            ]
        },
        {
            "task": "View system logs",
            "steps": [
                ("journalctl -xe --no-pager | tail -20", f"Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)} server systemd[1]: Started {random_class_name()} Service."),
                ("tail -f /var/log/syslog | head -10", f"Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)} server kernel: [{random_number(1000, 99999)}.{random_number(100, 999)}] {random_func_name()}: completed"),
                ("dmesg | tail -10", f"[{random_number(1000, 99999)}.{random_number(100, 999)}] {random.choice(['eth0', 'wlan0', 'enp0s3'])}: link up"),
            ]
        },
        {
            "task": "Check running processes",
            "steps": [
                (f"ps aux --sort=-%mem | head -10", f"USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n{random.choice(NAMES).lower()}     {random_number(1000, 9999)}  {random.uniform(0.1, 10):.1f}  {random.uniform(0.1, 5):.1f} {random_number(100000, 999999)} {random_number(10000, 99999)} ?        Ssl  {random_number(10,23)}:{random_number(10,59)}   {random_number(0,10)}:{random_number(10,59)} {random.choice(['python', 'node', 'java', 'nginx'])}"),
                (f"pgrep -a {random.choice(['python', 'node', 'nginx', 'postgres'])}", f"{random_number(1000, 9999)} {random.choice(['python', 'node', 'nginx', 'postgres'])} {random_file_path()}"),
                ("uptime", f" {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)} up {random_number(1, 365)} days, {random_number(0, 23)}:{random_number(10,59)}, {random_number(1, 10)} users, load average: {random.uniform(0.1, 4):.2f}, {random.uniform(0.1, 4):.2f}, {random.uniform(0.1, 4):.2f}"),
            ]
        },
    ]
    
    template = random.choice(monitoring_tasks)
    
    steps_text = ""
    for cmd, output in template["steps"]:
        steps_text += f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n{t['exec_result']}\n{output}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{template['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {template['task'].lower()}:\n\n"
        f"{steps_text}\n"
        f"Done!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "monitoring", "idx": idx}


def generate_unique_download(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique download sample."""
    filename = f"{random_var_name()}.{random.choice(['tar.gz', 'zip', 'deb', 'rpm', 'AppImage', 'sh'])}"
    url = f"{random_url()}/{filename}"
    size = f"{random_number(1, 500)}M"
    
    download_tasks = [
        {
            "task": f"Download {filename}",
            "steps": [
                (f"wget -O {filename} {url}", f"--{random_number(2020, 2024)}-{random_number(1,12):02d}-{random_number(1,28):02d} {random_number(10,23)}:{random_number(10,59)}:{random_number(10,59)}--  {url}\nLength: {random_number(1000000, 500000000)} ({size}) [application/octet-stream]\nSaving to: '{filename}'\n\n{filename}          100%[===================>] {size}  {random_number(1, 50)}MB/s    in {random.uniform(0.1, 30):.1f}s"),
                (f"ls -lh {filename}", f"-rw-r--r-- 1 user user {size} Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)} {filename}"),
            ]
        },
        {
            "task": f"Download with curl",
            "steps": [
                (f"curl -L -o {filename} {url}", f"  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current\n                                 Dload  Upload   Total   Spent    Left  Speed\n100 {size}  100 {size}    0     0  {random_number(1, 50)}M      0  0:00:{random_number(1, 30):02d}  0:00:{random_number(1, 30):02d} --:--:-- {random_number(1, 50)}M"),
                (f"file {filename}", f"{filename}: {random.choice(['gzip compressed data', 'Zip archive data', 'ELF 64-bit LSB executable', 'Bourne-Again shell script'])}"),
            ]
        },
        {
            "task": f"Download and extract",
            "steps": [
                (f"wget {url}", f"Saving to: '{filename}'"),
                (f"tar -xzf {filename}" if filename.endswith('.tar.gz') else f"unzip {filename}", f"{'x ' + random_var_name() + '/' if filename.endswith('.tar.gz') else 'Archive:  ' + filename}"),
                ("ls -la", f"total {random_number(10, 100)}\ndrwxr-xr-x  {random_number(2, 10)} user user {random_number(100, 9999)} Jan {random_number(1,28)} {random_number(10,23)}:{random_number(10,59)} {random_var_name()}"),
            ]
        },
    ]
    
    template = random.choice(download_tasks)
    
    steps_text = ""
    for cmd, output in template["steps"]:
        steps_text += f"{t['exec_start']}\n$ {cmd}\n{t['exec_end']}\n{t['exec_result']}\n{output}\n{t['exec_result_end']}\n"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{template['task']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you {template['task'].lower()}:\n\n"
        f"{steps_text}\n"
        f"Done!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "download", "idx": idx}


def generate_unique_language_setup(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique language environment setup sample."""
    lang = random.choice(LANGUAGES)
    version = random_version()
    
    lang_configs = {
        "Python": ("python3", "pip3", f"python3 --version", f"Python {version}"),
        "JavaScript": ("node", "npm", "node --version", f"v{version}"),
        "TypeScript": ("tsc", "npm", "tsc --version", f"Version {version}"),
        "Rust": ("rustc", "cargo", "rustc --version", f"rustc {version}"),
        "Go": ("go", "go", "go version", f"go version go{version} linux/amd64"),
        "Java": ("java", "mvn", "java --version", f"openjdk {version}"),
        "Ruby": ("ruby", "gem", "ruby --version", f"ruby {version}"),
        "PHP": ("php", "composer", "php --version", f"PHP {version}"),
    }
    
    if lang not in lang_configs:
        lang = "Python"
    
    cmd, pkg_mgr, ver_cmd, ver_output = lang_configs[lang]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nSet up {lang} development environment\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you set up {lang}:\n\n"
        f"{t['exec_start']}\n$ sudo apt install -y {cmd.lower()} {pkg_mgr}\n{t['exec_end']}\n"
        f"{t['exec_result']}\nSetting up {cmd.lower()}...\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ {ver_cmd}\n{t['exec_end']}\n"
        f"{t['exec_result']}\n{ver_output}\n{t['exec_result_end']}\n\n"
        f"Let me verify with a test:\n\n"
        f"{t['code_start']}{t.get('lang_' + lang.lower(), t['lang_python'])}\nprint('Hello, {lang}!')\n{t['code_end']}\n\n"
        f"{t['exec_start']}\n$ {cmd.lower()} hello.{'py' if lang == 'Python' else 'js' if lang in ['JavaScript', 'TypeScript'] else 'go' if lang == 'Go' else 'rb' if lang == 'Ruby' else 'java'}\n{t['exec_end']}\n"
        f"{t['exec_result']}\nHello, {lang}!\n{t['exec_result_end']}\n\n"
        f"{lang} is now installed and ready to use!\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "language_setup", "language": lang, "idx": idx}


def generate_unique_desktop_setup(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique desktop environment setup sample."""
    desktop = random.choice(["GNOME", "KDE Plasma", "XFCE", "LXDE", "Cinnamon", "MATE"])
    display = random.choice(["VNC", "X11", "Wayland", "RDP"])
    resolution = random.choice(["1920x1080", "2560x1440", "3840x2160", "1280x720"])
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nSet up {desktop} desktop with {display}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"I'll help you set up {desktop} with {display}:\n\n"
        f"{t['exec_start']}\n$ sudo apt install -y {desktop.lower().replace(' ', '-')}-desktop\n{t['exec_end']}\n"
        f"{t['exec_result']}\nSetting up {desktop.lower()}...\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ sudo apt install -y {display.lower()}-server\n{t['exec_end']}\n"
        f"{t['exec_result']}\nSetting up {display.lower()}-server...\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ {display.lower()}server :{random_number(1, 5)} -geometry {resolution} -depth 24\n{t['exec_end']}\n"
        f"{t['exec_result']}\nNew '{desktop} desktop' at :{random_number(1, 5)} on machine hostname\n{t['exec_result_end']}\n"
        f"{t['exec_start']}\n$ {display.lower()}server -list\n{t['exec_end']}\n"
        f"{t['exec_result']}\nX DISPLAY #\tRFB PORT #\tPROCESS ID\n:{random_number(1, 5)}\t\t{random_number(5900, 5910)}\t\t{random_number(10000, 99999)}\n{t['exec_result_end']}\n\n"
        f"Done! {desktop} desktop is now running with {display} at {resolution}.\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "desktop_setup", "idx": idx}


# =============================================================================
# DOCUMENT GENERATORS
# =============================================================================

def generate_unique_document(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique document handling sample."""
    doc_types = ["txt", "md", "json", "yaml", "csv", "log"]
    doc_type = random.choice(doc_types)
    
    topic = random.choice([
        f"{random_class_name()} implementation guide",
        f"{random_func_name()} API documentation",
        f"{random.choice(LANGUAGES)} best practices",
        f"{random.choice(FRAMEWORKS)} configuration",
        f"{random.choice(DATABASES)} setup guide",
        f"System {random_var_name()} report",
        f"Project {random_var_name()} specification",
        f"Meeting notes for {random_class_name()}",
        f"Release notes v{random_version()}",
        f"Error log analysis for {random_func_name()}",
    ])
    
    filename = f"{random_var_name()}.{doc_type}"
    
    if doc_type == "txt":
        content = f"""# {topic}

## Overview
This document describes the {random_func_name()} functionality implemented in {random.choice(LANGUAGES)}.
Version: {random_version()}
Author: {random.choice(NAMES)}
Date: {random_number(2020, 2025)}-{random_number(1,12):02d}-{random_number(1,28):02d}

## Details
The {random_var_name()} component handles {random.choice(['data processing', 'user authentication', 'API requests', 'file management', 'database operations'])}.
It supports {random_number(2, 10)} concurrent operations and has a throughput of {random_number(100, 10000)} requests/second.

## Configuration
- max_connections: {random_number(10, 1000)}
- timeout_ms: {random_number(100, 5000)}
- retry_count: {random_number(1, 5)}
"""
    elif doc_type == "md":
        content = f"""# {topic}

## Introduction
This is the documentation for **{random_class_name()}** module.

### Features
- {random_func_name()}: Handles {random.choice(['data', 'requests', 'events', 'messages'])}
- {random_func_name()}: Processes {random.choice(['input', 'output', 'streams', 'batches'])}
- {random_func_name()}: Manages {random.choice(['state', 'cache', 'connections', 'resources'])}

### Usage
```{random.choice(['python', 'javascript', 'bash'])}
{random_var_name()} = {random_class_name()}()
{random_var_name()}.{random_func_name()}()
```

### Performance
| Metric | Value |
|--------|-------|
| Latency | {random_number(1, 100)}ms |
| Throughput | {random_number(100, 10000)}/s |
"""
    elif doc_type == "json":
        content = f"""{{
  "name": "{random_class_name()}",
  "version": "{random_version()}",
  "description": "{topic}",
  "config": {{
    "{random_var_name()}": {random_number(1, 100)},
    "{random_var_name()}": "{random.choice(['enabled', 'disabled', 'auto'])}",
    "{random_var_name()}": {random.choice(['true', 'false'])}
  }},
  "dependencies": [
    "{random.choice(PACKAGES)}",
    "{random.choice(PACKAGES)}"
  ]
}}"""
    elif doc_type == "yaml":
        content = f"""# {topic}
name: {random_class_name()}
version: {random_version()}

config:
  {random_var_name()}: {random_number(1, 100)}
  {random_var_name()}: {random.choice(['enabled', 'disabled'])}
  {random_var_name()}: {random.choice(['true', 'false'])}

services:
  - name: {random_func_name()}
    port: {random_port()}
    replicas: {random_number(1, 5)}
"""
    elif doc_type == "csv":
        headers = [random_var_name() for _ in range(4)]
        rows = []
        for _ in range(5):
            rows.append(",".join([str(random_number(1, 1000)) for _ in range(4)]))
        content = ",".join(headers) + "\n" + "\n".join(rows)
    else:  # log
        log_levels = ["INFO", "WARN", "ERROR", "DEBUG"]
        lines = []
        for i in range(10):
            ts = f"{random_number(2024, 2025)}-{random_number(1,12):02d}-{random_number(1,28):02d} {random_number(0,23):02d}:{random_number(0,59):02d}:{random_number(0,59):02d}"
            level = random.choice(log_levels)
            msg = f"{random_func_name()} {random.choice(['completed', 'started', 'failed', 'retrying'])} for {random_var_name()}"
            lines.append(f"[{ts}] [{level}] {msg}")
        content = "\n".join(lines)
    
    tasks = [
        ("Summarize this document", f"This document describes {topic}. Key points include configuration settings and implementation details."),
        ("What is the main topic?", f"The main topic is {topic}."),
        ("Extract the key information", f"Key information: {topic}, version {random_version()}, with {random_number(2, 5)} main components."),
        ("Explain this document", f"This is a {doc_type} file containing {topic}. It provides configuration and documentation."),
    ]
    
    task, response = random.choice(tasks)
    
    text = (
        f"{t['bos']}"
        f"{t['filename_start']}{filename}{t['filename_end']}"
        f"{t['doc_start']}\n{content}\n{t['doc_end']}"
        f"{t['user_start']}\n{task}\n{t['user_end']}"
        f"{t['assistant_start']}\n{response}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "document", "file_type": doc_type, "idx": idx}


# =============================================================================
# ANTI-HALLUCINATION GENERATORS
# =============================================================================

def generate_unique_idk(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique 'I don't know' sample."""
    # Dynamic unknowable questions
    question_templates = [
        f"What will {random.choice(NAMES)} do tomorrow at {random_number(1, 12)}:{random_number(10, 59):02d}?",
        f"What is the exact number of {random.choice(['stars', 'atoms', 'grains of sand', 'fish', 'birds'])} in {random.choice(['the universe', 'the ocean', 'the world', 'this room'])}?",
        f"What will the {random.choice(['stock market', 'weather', 'Bitcoin price', 'lottery numbers'])} be on {random_number(2025, 2030)}-{random_number(1,12):02d}-{random_number(1,28):02d}?",
        f"What is {random.choice(NAMES)}'s {random.choice(['password', 'bank balance', 'phone number', 'social security number', 'private thoughts'])}?",
        f"What did {random.choice(NAMES)} dream about {random_number(1, 30)} days ago?",
        f"What is the {random.choice(['cure', 'solution', 'answer'])} to {random.choice(['all diseases', 'world hunger', 'climate change', 'aging'])}?",
        f"What will happen to {random.choice(NAMES)} in {random_number(10, 100)} years?",
        f"What is the meaning of life for {random.choice(NAMES)} specifically?",
        f"What number between {random_number(1, 100)} and {random_number(1000, 10000)} am I thinking of?",
        f"What will {random.choice(['Google', 'Apple', 'Microsoft', 'Amazon'])}'s stock price be in {random_number(1, 10)} years?",
        f"What is {random.choice(NAMES)}'s exact location right now?",
        f"What will be invented in {random_number(2030, 2100)}?",
        f"What is the last digit of pi?",
        f"What will {random.choice(NAMES)} eat for dinner on {random_number(2025, 2030)}-{random_number(1,12):02d}-{random_number(1,28):02d}?",
        f"How many {random.choice(['thoughts', 'heartbeats', 'breaths'])} has {random.choice(NAMES)} had in their lifetime?",
    ]
    
    question = random.choice(question_templates)
    
    responses = [
        "I don't have access to that information.",
        "I cannot predict future events.",
        "This is beyond my knowledge and capabilities.",
        "I don't know the answer to this question.",
        "I cannot access personal or private information.",
        "This is unknowable or unpredictable.",
        "I'm not able to provide this information.",
        "This falls outside what I can reliably answer.",
    ]
    
    explanations = [
        "This involves predicting the future, which is inherently uncertain.",
        "I don't have access to personal or private information.",
        "This requires real-time data that I don't have.",
        "The information needed to answer this doesn't exist or isn't accessible.",
        "This involves randomness that cannot be predicted.",
        "This question asks about subjective experiences I cannot know.",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['beyond_knowledge']}\n"
        f"{random.choice(responses)}\n\n"
        f"{t['note_start']}{random.choice(explanations)}{t['note_end']}\n"
        f"{t['confidence_low']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "idk", "idx": idx}


def generate_unique_fact_check(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique fact-checking sample."""
    # Generate dynamic fact-check scenarios
    facts = [
        (f"The speed of light is {random_number(200000, 400000)} km/s.", random_number(200000, 400000) == 299792, "The speed of light is approximately 299,792 km/s."),
        (f"{random.choice(LANGUAGES)} was created in {random_number(1970, 2020)}.", False, f"The creation date varies by language. Python was 1991, JavaScript 1995, etc."),
        (f"The human body has {random_number(100, 300)} bones.", random_number(100, 300) == 206, "Adults have 206 bones, though babies have about 270."),
        (f"Water boils at {random_number(80, 120)}°C at sea level.", random_number(80, 120) == 100, "Water boils at 100°C (212°F) at sea level."),
        (f"The Earth is {random.uniform(4.0, 5.0):.2f} billion years old.", True, "The Earth is approximately 4.54 billion years old."),
        (f"{random.choice(['TCP', 'UDP', 'HTTP', 'FTP'])} is a {random.choice(['reliable', 'unreliable', 'stateless', 'stateful'])} protocol.", random.choice([True, False]), "Protocol characteristics vary. TCP is reliable, UDP is unreliable."),
        (f"Git was created by {random.choice(['Linus Torvalds', 'Bill Gates', 'Steve Jobs', 'Mark Zuckerberg'])}.", random.choice([True, False]), "Git was created by Linus Torvalds in 2005."),
        (f"The first computer virus appeared in {random_number(1970, 2000)}.", random_number(1970, 2000) == 1986, "The first PC virus (Brain) appeared in 1986."),
        (f"JavaScript and Java are {random.choice(['the same', 'completely different', 'related'])} languages.", random.choice([True, False]), "JavaScript and Java are completely different languages."),
        (f"RAM is {random.choice(['volatile', 'non-volatile', 'permanent'])} memory.", random.choice([True, False]), "RAM is volatile memory - it loses data when power is off."),
    ]
    
    claim, is_true, correction = random.choice(facts)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nIs this true? \"{claim}\"\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t.get('plan_start', '')}\nLet me verify this claim step by step.\n{t.get('plan_end', '')}\n"
        f"{t.get('critique_start', '')}\nAnalyzing the factual accuracy...\n{t.get('critique_end', '')}\n"
        f"{t['fact_check']}\n"
        f"{'This is correct.' if is_true else 'This is incorrect.'} {correction}\n"
        f"{t['confidence_high'] if is_true else t['confidence_medium']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "fact_check", "idx": idx}


def generate_unique_grounded_response(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique grounded response sample."""
    # Dynamic context-question-answer pairs
    topics = [
        {
            "context": f"According to the {random.choice(['official documentation', 'research paper', 'technical specification'])}, {random.choice(LANGUAGES)} version {random_version()} introduced {random_func_name()} functionality.",
            "question": f"What feature was added in {random.choice(LANGUAGES)} {random_version()}?",
            "answer": f"The {random_func_name()} functionality was introduced.",
            "source": f"{random.choice(LANGUAGES)} documentation"
        },
        {
            "context": f"The {random.choice(DATABASES)} documentation states that the default port is {random_port()} and supports up to {random_number(100, 10000)} concurrent connections.",
            "question": f"What is the default port for {random.choice(DATABASES)}?",
            "answer": f"The default port is {random_port()}.",
            "source": f"{random.choice(DATABASES)} documentation"
        },
        {
            "context": f"Research from {random.choice(['MIT', 'Stanford', 'Google', 'Microsoft'])} shows that {random_func_name()} algorithms can improve performance by {random_number(10, 90)}%.",
            "question": f"How much can {random_func_name()} improve performance?",
            "answer": f"Research shows improvements of up to {random_number(10, 90)}%.",
            "source": f"{random.choice(['MIT', 'Stanford', 'Google', 'Microsoft'])} research"
        },
        {
            "context": f"The {random.choice(FRAMEWORKS)} framework requires {random.choice(LANGUAGES)} version {random_version()} or higher.",
            "question": f"What version of {random.choice(LANGUAGES)} does {random.choice(FRAMEWORKS)} require?",
            "answer": f"Version {random_version()} or higher is required.",
            "source": f"{random.choice(FRAMEWORKS)} documentation"
        },
    ]
    
    topic = random.choice(topics)
    
    text = (
        f"{t['bos']}"
        f"{t['context_start']}\n{topic['context']}\n{t['context_end']}"
        f"{t['user_start']}\n{topic['question']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['grounded']}{topic['answer']}\n"
        f"{t['cite_start']}{t['source_start']}{topic['source']}{t['source_end']}{t['cite_end']}\n"
        f"{t['confidence_high']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "grounded_response", "idx": idx}


def generate_unique_self_correction(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique self-correction sample."""
    corrections = [
        {
            "wrong": f"The speed of light is {random_number(200000, 400000)} miles per second.",
            "correction": f"I need to correct myself. The speed of light is approximately 186,000 miles per second (or 299,792 km/s)."
        },
        {
            "wrong": f"{random.choice(LANGUAGES)} was created in {random_number(1980, 2010)}.",
            "correction": f"Let me correct that. The actual creation date may differ - Python was 1991, JavaScript 1995, Go 2009, Rust 2010."
        },
        {
            "wrong": f"HTTP uses port {random_number(70, 90)} by default.",
            "correction": f"I should correct this. HTTP uses port 80 by default, and HTTPS uses port 443."
        },
        {
            "wrong": f"Git was created by {random.choice(['Bill Gates', 'Steve Jobs', 'Mark Zuckerberg'])}.",
            "correction": "That's incorrect. Git was created by Linus Torvalds in 2005 for Linux kernel development."
        },
        {
            "wrong": f"The human body has {random_number(150, 250)} bones.",
            "correction": "I need to clarify: adults have 206 bones, though babies are born with about 270 bones that fuse over time."
        },
        {
            "wrong": f"RAM stores data {random.choice(['permanently', 'forever', 'even when powered off'])}.",
            "correction": "That's wrong. RAM is volatile memory - it loses all data when power is turned off. SSDs and HDDs provide permanent storage."
        },
        {
            "wrong": f"{random.choice(DATABASES)} is a {random.choice(['relational', 'NoSQL', 'graph'])} database.",
            "correction": f"Let me correct that. Database types vary: PostgreSQL/MySQL are relational, MongoDB is NoSQL, Neo4j is graph."
        },
        {
            "wrong": f"TCP is {random.choice(['unreliable', 'connectionless', 'faster than UDP'])}.",
            "correction": "I should clarify: TCP is reliable and connection-oriented. UDP is unreliable but faster for some use cases."
        },
    ]
    
    example = random.choice(corrections)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nPlease review and correct if needed: \"{example['wrong']}\"\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['self_correct']}\n"
        f"{t['correction_start']}\n{example['correction']}\n{t['correction_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "self_correction", "idx": idx}


def generate_unique_confidence(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique confidence level sample."""
    examples = [
        # High confidence - factual
        {"question": f"What is {random_number(2, 20)} + {random_number(2, 20)}?", "answer": str(random_number(4, 40)), "confidence": "high", "explanation": "Basic arithmetic."},
        {"question": f"What is the capital of {random.choice(['France', 'Germany', 'Japan', 'Brazil', 'Australia'])}?", "answer": random.choice(["Paris", "Berlin", "Tokyo", "Brasília", "Canberra"]), "confidence": "high", "explanation": "Well-established fact."},
        {"question": f"Is {random.choice(LANGUAGES)} a programming language?", "answer": "Yes", "confidence": "high", "explanation": "Definitional fact."},
        # Medium confidence - contextual
        {"question": f"Is {random.choice(FRAMEWORKS)} good for {random_var_name()} projects?", "answer": "It depends on your specific requirements and team expertise.", "confidence": "medium", "explanation": "Context-dependent."},
        {"question": f"Should I use {random.choice(DATABASES)} or {random.choice(DATABASES)}?", "answer": "Both have merits. Consider your data model and scaling needs.", "confidence": "medium", "explanation": "Trade-offs involved."},
        {"question": f"Is {random.choice(['remote work', 'agile', 'microservices'])} better?", "answer": "It depends on the organization and context.", "confidence": "medium", "explanation": "Subjective and situational."},
        # Low confidence - speculative
        {"question": f"Will {random.choice(LANGUAGES)} be popular in {random_number(2040, 2100)}?", "answer": "Impossible to predict with certainty.", "confidence": "low", "explanation": "Future prediction."},
        {"question": f"What will {random.choice(['AI', 'quantum computing'])} achieve by {random_number(2030, 2050)}?", "answer": "Highly speculative - technology evolves unpredictably.", "confidence": "low", "explanation": "Long-term forecast."},
    ]
    
    example = random.choice(examples)
    
    conf_token = {
        "high": t['confidence_high'],
        "medium": t['confidence_medium'],
        "low": t['confidence_low'],
    }[example['confidence']]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{example['question']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{conf_token}\n"
        f"{example['answer']}\n"
        f"{t['note_start']}Confidence: {example['confidence']} - {example['explanation']}{t['note_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "confidence_level", "idx": idx}


def generate_unique_citation(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique citation sample."""
    citations = [
        {"claim": f"The {random.choice(LANGUAGES)} language has {random_number(1, 50)} million users.", "source": f"{random.choice(LANGUAGES)} Foundation", "quote": f"Our community has grown to {random_number(1, 50)} million developers."},
        {"claim": f"{random.choice(DATABASES)} can handle {random_number(10000, 1000000)} transactions per second.", "source": f"{random.choice(DATABASES)} benchmarks", "quote": f"Performance tests show {random_number(10000, 1000000)} TPS under load."},
        {"claim": f"The {random.choice(FRAMEWORKS)} framework reduces development time by {random_number(20, 80)}%.", "source": f"{random.choice(FRAMEWORKS)} case studies", "quote": f"Teams report {random_number(20, 80)}% faster development cycles."},
        {"claim": f"Cloud computing market will reach ${random_number(500, 2000)} billion by {random_number(2025, 2030)}.", "source": "Gartner Research", "quote": f"Market projections indicate ${random_number(500, 2000)}B by {random_number(2025, 2030)}."},
        {"claim": f"AI adoption has increased by {random_number(50, 300)}% since {random_number(2018, 2022)}.", "source": "McKinsey Report", "quote": f"Enterprise AI adoption grew {random_number(50, 300)}% in recent years."},
        {"claim": f"Open source software powers {random_number(70, 99)}% of the internet.", "source": "Linux Foundation", "quote": f"Open source runs on {random_number(70, 99)}% of web servers."},
        {"claim": f"Remote work productivity increased by {random_number(10, 50)}% during the pandemic.", "source": "Stanford Research", "quote": f"Studies show {random_number(10, 50)}% productivity gains for remote workers."},
        {"claim": f"Cybersecurity spending will exceed ${random_number(100, 500)} billion by {random_number(2025, 2030)}.", "source": "Cybersecurity Ventures", "quote": f"Global spending projected at ${random_number(100, 500)}B."},
    ]
    
    cite = random.choice(citations)
    
    questions = [
        f"What do we know about {cite['claim'].split()[1].lower()} {cite['claim'].split()[2].lower()}?",
        f"Can you cite a source for: {cite['claim']}",
        f"Is there evidence that {cite['claim'].lower()}",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(questions)}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{cite['claim']}\n\n"
        f"{t['cite_start']}\n"
        f"{t['source_start']}{cite['source']}{t['source_end']}\n"
        f"{t['quote_start']}{cite['quote']}{t['quote_end']}\n"
        f"{t['cite_end']}\n"
        f"{t['confidence_high']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "citation", "idx": idx}


def generate_unique_retrieval_grounded(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique RAG-style retrieval-grounded sample."""
    docs = [
        {
            "query": f"How does {random_func_name()} work in {random.choice(LANGUAGES)}?",
            "retrieved": f"The {random_func_name()} function in {random.choice(LANGUAGES)} processes data by iterating through collections and applying transformations. It supports both synchronous and asynchronous operations.",
            "answer": f"The {random_func_name()} function processes data through iteration and transformation, supporting both sync and async modes."
        },
        {
            "query": f"What is the best practice for {random_var_name()} in {random.choice(FRAMEWORKS)}?",
            "retrieved": f"Best practices for {random_var_name()} include: 1) Use dependency injection, 2) Follow single responsibility principle, 3) Write unit tests, 4) Document public APIs.",
            "answer": f"Best practices include dependency injection, single responsibility, unit testing, and API documentation."
        },
        {
            "query": f"How do I configure {random.choice(DATABASES)} for {random_var_name()}?",
            "retrieved": f"To configure {random.choice(DATABASES)}: Set connection pool size to {random_number(10, 100)}, enable query caching, configure replication for high availability, and set appropriate timeouts.",
            "answer": f"Configure with pool size {random_number(10, 100)}, enable caching, set up replication, and configure timeouts."
        },
        {
            "query": f"What causes {random.choice(ERROR_TYPES)} in {random.choice(LANGUAGES)}?",
            "retrieved": f"{random.choice(ERROR_TYPES)} typically occurs when accessing invalid data, null references, or type mismatches. Common fixes include null checks, type validation, and proper error handling.",
            "answer": f"This error occurs from invalid data access, null references, or type mismatches. Fix with null checks and validation."
        },
    ]
    
    doc = random.choice(docs)
    
    text = (
        f"{t['bos']}"
        f"{t['retrieved_start']}\n{doc['retrieved']}\n{t['retrieved_end']}"
        f"{t['user_start']}\n{doc['query']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['grounded']}\n"
        f"{doc['answer']}\n"
        f"{t['confidence_high']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "retrieval_grounded", "idx": idx}


def generate_unique_knowledge_cutoff(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique knowledge cutoff awareness sample."""
    # More dynamic question generation
    date = f"{random_number(2025, 2030)}-{random_number(1,12):02d}-{random_number(1,28):02d}"
    time = f"{random_number(0, 23):02d}:{random_number(0, 59):02d}"
    city = random.choice(['New York', 'London', 'Tokyo', 'Sydney', 'Paris', 'Berlin', 'Singapore', 'Dubai', 'Toronto', 'Mumbai'])
    company = random.choice(['Google', 'Apple', 'Microsoft', 'Amazon', 'Meta', 'Tesla', 'Netflix', 'Nvidia', 'OpenAI', 'Anthropic'])
    person = random.choice(['Elon Musk', 'Tim Cook', 'Satya Nadella', 'Mark Zuckerberg', 'Jeff Bezos', 'Sam Altman', 'Sundar Pichai'])
    asset = random.choice(['Bitcoin', 'Ethereum', 'Apple stock', 'gold', 'silver', 'oil', 'S&P 500', 'NASDAQ', 'Tesla stock'])
    event = random.choice(['Super Bowl', 'World Cup', 'Olympics', 'election', 'Grammy Awards', 'Oscars', 'World Series', 'Champions League'])
    
    question_templates = [
        f"What happened in the news on {date}?",
        f"What is the current price of {asset}?",
        f"Who won the {event} in {random_number(2025, 2030)}?",
        f"What is the latest version of {random.choice(LANGUAGES)} as of {date}?",
        f"What did {person} announce on {date}?",
        f"What is the weather in {city} right now at {time}?",
        f"What are the current {random.choice(['COVID', 'flu', 'unemployment', 'inflation'])} statistics for {random_number(2025, 2030)}?",
        f"What movies came out on {date}?",
        f"What is {company}'s stock price today ({date})?",
        f"What happened at the {random.choice(['tech conference', 'summit', 'event', 'meeting'])} on {date}?",
        f"What is {person}'s net worth as of {date}?",
        f"What is the population of {city} in {random_number(2025, 2030)}?",
        f"What is the latest news about {company}?",
        f"What is the score of the {random.choice(['game', 'match', 'tournament'])} happening right now?",
        f"What is trending on {random.choice(['Twitter', 'TikTok', 'Instagram', 'YouTube'])} today?",
        f"What is the exchange rate between {random.choice(['USD', 'EUR', 'GBP'])} and {random.choice(['JPY', 'CNY', 'INR'])} today?",
        f"What is the current temperature in {city}?",
        f"What did {company} release on {date}?",
        f"Who is the current CEO of {company}?",
        f"What is the latest {random.choice(LANGUAGES)} framework released in {random_number(2025, 2030)}?",
    ]
    
    question = random.choice(question_templates)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['knowledge_cutoff']}\n"
        f"I don't have access to real-time information or events after my knowledge cutoff date. "
        f"For current information, please check reliable news sources or official websites.\n"
        f"{t['confidence_low']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "knowledge_cutoff", "idx": idx}


def generate_unique_uncertainty(t: Dict[str, str], idx: int) -> Dict[str, Any]:
    """Generate unique uncertainty expression sample."""
    # Generate dynamic uncertain topics
    topic_templates = [
        (f"What will {random.choice(NAMES)}'s career look like in {random_number(5, 50)} years?", "Predicting individual futures is impossible - too many variables.", random_number(70, 95)),
        (f"Will {random.choice(LANGUAGES)} still be popular in {random_number(2030, 2100)}?", "Technology trends are unpredictable. Languages rise and fall based on many factors.", random_number(60, 85)),
        (f"What will the price of {random.choice(['Bitcoin', 'Ethereum', 'gold', 'oil'])} be in {random_number(1, 10)} years?", "Financial markets are inherently unpredictable. No one can reliably forecast prices.", random_number(80, 95)),
        (f"Will {random.choice(COMPANIES) if 'COMPANIES' in dir() else random.choice(['Google', 'Apple', 'Microsoft', 'Amazon'])} exist in {random_number(50, 200)} years?", "Corporate longevity is uncertain. Most companies don't survive that long.", random_number(65, 85)),
        (f"What causes {random.choice(['consciousness', 'dreams', 'creativity', 'intuition', 'emotions'])}?", "This remains one of the biggest mysteries in neuroscience and philosophy.", random_number(70, 90)),
        (f"Is there intelligent life on {random.choice(['Mars', 'Europa', 'Titan', 'Enceladus', 'exoplanets'])}?", "We have no definitive evidence yet. The search continues.", random_number(55, 75)),
        (f"What will {random.choice(['AI', 'quantum computing', 'fusion power', 'space travel'])} look like in {random_number(20, 100)} years?", "Long-term technological predictions are highly speculative.", random_number(75, 95)),
        (f"Will {random.choice(['climate change', 'pandemics', 'asteroid impacts', 'AI'])} cause human extinction?", "Existential risks are difficult to quantify. Experts disagree significantly.", random_number(60, 85)),
        (f"What is the best {random.choice(['programming language', 'framework', 'database', 'cloud provider'])} for {random_var_name()}?", "This depends on specific requirements, team expertise, and constraints.", random_number(50, 70)),
        (f"Should I invest in {random.choice(['stocks', 'crypto', 'real estate', 'bonds'])} right now?", "I cannot provide financial advice. Markets are unpredictable.", random_number(80, 95)),
        (f"What will happen to {random.choice(['democracy', 'capitalism', 'the internet', 'social media'])} in the future?", "Social and political systems evolve unpredictably.", random_number(70, 90)),
        (f"Is {random.choice(['string theory', 'multiverse theory', 'simulation hypothesis'])} correct?", "These are theoretical frameworks without definitive experimental confirmation.", random_number(75, 95)),
        (f"What is the cure for {random.choice(['cancer', 'Alzheimers', 'aging', 'all diseases'])}?", "Medical breakthroughs are unpredictable. Research is ongoing.", random_number(65, 85)),
        (f"Will {random.choice(['remote work', 'AI assistants', 'self-driving cars', 'VR'])} become universal by {random_number(2030, 2050)}?", "Adoption rates depend on many social and economic factors.", random_number(55, 75)),
        (f"What number am I thinking of between {random_number(1, 100)} and {random_number(100, 1000)}?", "I cannot read minds or access your thoughts.", random_number(95, 99)),
    ]
    
    topic, response, uncertainty = random.choice(topic_templates)
    
    # Add variation to phrasing
    prefixes = ["", "Can you tell me ", "I'm curious - ", "Quick question: ", "I wonder "]
    suffixes = ["", " I really want to know.", " This has been on my mind.", " What do you think?"]
    
    topic = random.choice(prefixes) + topic + random.choice(suffixes)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{topic}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t.get('uncertainty_score', '')}{uncertainty}{t.get('uncertainty_score_end', '')}\n"
        f"{t.get('uncertain', '')}\n"
        f"{t.get('speculative', '')}{response}\n"
        f"{t.get('uncertain_end', '')}\n"
        f"{t['confidence_low']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "uncertainty_expression", "idx": idx}


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_unique_dataset(generator_func, output_path: str, num_samples: int = 2000):
    """Generate a dataset with guaranteed unique samples."""
    t = SPECIAL_TOKENS
    samples = []
    seen_hashes = set()
    attempts = 0
    max_attempts = num_samples * 10  # Allow retries for uniqueness
    
    while len(samples) < num_samples and attempts < max_attempts:
        sample = generator_func(t, len(samples))
        sample_hash = hashlib.md5(sample["text"].encode()).hexdigest()
        
        if sample_hash not in seen_hashes:
            seen_hashes.add(sample_hash)
            del sample["idx"]  # Remove idx from final output
            samples.append(sample)
        
        attempts += 1
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return len(samples), len(seen_hashes)


def generate_all_datasets(output_dir: str, samples_per_type: int = 2000):
    """Generate all datasets with unique samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    generators = [
        # Agentic/Code datasets
        ("shell_execution_dataset.jsonl", generate_unique_shell_execution, "Shell Execution"),
        ("shell_error_dataset.jsonl", generate_unique_shell_error, "Shell Errors"),
        ("shell_timeout_dataset.jsonl", generate_unique_shell_timeout, "Shell Timeouts"),
        ("python_script_dataset.jsonl", generate_unique_python_script, "Python Scripts"),
        ("jupyter_dataset.jsonl", generate_unique_jupyter, "Jupyter Notebooks"),
        ("multi_step_execution_dataset.jsonl", generate_unique_multi_step, "Multi-Step Execution"),
        ("file_ops_dataset.jsonl", generate_unique_file_ops, "File Operations"),
        ("edit_lines_dataset.jsonl", generate_unique_edit_lines, "Line Edits"),
        ("debugging_dataset.jsonl", generate_unique_debugging, "Debugging"),
        ("diff_dataset.jsonl", generate_unique_diff, "Diffs"),
        ("commit_dataset.jsonl", generate_unique_commit, "Commits"),
        ("fim_dataset.jsonl", generate_unique_fim, "Fill-in-Middle"),
        ("issue_dataset.jsonl", generate_unique_issue, "GitHub Issues"),
        ("repo_context_dataset.jsonl", generate_unique_repo_context, "Repo Context"),
        ("execution_dataset.jsonl", generate_unique_execution, "Code Execution"),
        # System Admin datasets
        ("apt_install_dataset.jsonl", generate_unique_apt_install, "APT Install"),
        ("docker_dataset.jsonl", generate_unique_docker, "Docker"),
        ("database_setup_dataset.jsonl", generate_unique_database_setup, "Database Setup"),
        ("webserver_setup_dataset.jsonl", generate_unique_webserver, "Web Server Setup"),
        ("ssh_setup_dataset.jsonl", generate_unique_ssh_setup, "SSH Setup"),
        ("monitoring_dataset.jsonl", generate_unique_monitoring, "System Monitoring"),
        ("download_dataset.jsonl", generate_unique_download, "Downloads"),
        ("language_setup_dataset.jsonl", generate_unique_language_setup, "Language Setup"),
        ("desktop_setup_dataset.jsonl", generate_unique_desktop_setup, "Desktop Setup"),
        # Document datasets
        ("document_dataset.jsonl", generate_unique_document, "Documents"),
        # Anti-hallucination datasets
        ("uncertainty_dataset.jsonl", generate_unique_uncertainty, "Uncertainty Expression"),
        ("idk_dataset.jsonl", generate_unique_idk, "I Don't Know"),
        ("fact_check_dataset.jsonl", generate_unique_fact_check, "Fact Checking"),
        ("grounded_response_dataset.jsonl", generate_unique_grounded_response, "Grounded Responses"),
        ("self_correction_dataset.jsonl", generate_unique_self_correction, "Self Correction"),
        ("confidence_level_dataset.jsonl", generate_unique_confidence, "Confidence Levels"),
        ("citation_dataset.jsonl", generate_unique_citation, "Citations"),
        ("retrieval_grounded_dataset.jsonl", generate_unique_retrieval_grounded, "Retrieval Grounded"),
        ("knowledge_cutoff_dataset.jsonl", generate_unique_knowledge_cutoff, "Knowledge Cutoff"),
    ]
    
    total_samples = 0
    total_unique = 0
    
    for filename, generator, desc in generators:
        filepath = os.path.join(output_dir, filename)
        print(f"📝 Generating {desc} dataset ({samples_per_type} unique samples)...")
        
        num_samples, num_unique = generate_unique_dataset(generator, filepath, samples_per_type)
        total_samples += num_samples
        total_unique += num_unique
        
        print(f"   ✅ {num_samples} samples ({num_unique} unique) saved to {filepath}")
    
    print(f"\n✅ Generated {len(generators)} datasets")
    print(f"   Total: {total_samples} samples ({total_unique} unique)")
    
    return total_samples, total_unique


# Export all generator functions for external use
__all__ = [
    'generate_all_datasets',
    'generate_unique_dataset',
    'generate_unique_shell_execution',
    'generate_unique_shell_error',
    'generate_unique_shell_timeout',
    'generate_unique_python_script',
    'generate_unique_jupyter',
    'generate_unique_multi_step',
    'generate_unique_file_ops',
    'generate_unique_edit_lines',
    'generate_unique_debugging',
    'generate_unique_diff',
    'generate_unique_commit',
    'generate_unique_fim',
    'generate_unique_issue',
    'generate_unique_repo_context',
    'generate_unique_execution',
    'generate_unique_apt_install',
    'generate_unique_docker',
    'generate_unique_database_setup',
    'generate_unique_webserver',
    'generate_unique_ssh_setup',
    'generate_unique_monitoring',
    'generate_unique_download',
    'generate_unique_language_setup',
    'generate_unique_desktop_setup',
    'generate_unique_document',
    'generate_unique_uncertainty',
    'generate_unique_idk',
    'generate_unique_fact_check',
    'generate_unique_grounded_response',
    'generate_unique_self_correction',
    'generate_unique_confidence',
    'generate_unique_citation',
    'generate_unique_retrieval_grounded',
    'generate_unique_knowledge_cutoff',
]
