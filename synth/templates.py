"""Templates for chain-of-thought reasoning generation."""

from typing import Dict, List

# Math problem templates with varying difficulty
MATH_TEMPLATES = {
    "arithmetic": [
        {"question": "What is {a} + {b}?", "vars": {"a": (1, 1000), "b": (1, 1000)}, "op": "add"},
        {"question": "Calculate {a} - {b}.", "vars": {"a": (100, 10000), "b": (1, 100)}, "op": "sub"},
        {"question": "What is {a} × {b}?", "vars": {"a": (2, 100), "b": (2, 100)}, "op": "mul"},
        {"question": "Divide {a} by {b}.", "vars": {"a": (10, 1000), "b": (2, 20)}, "op": "div"},
        {"question": "What is {a}% of {b}?", "vars": {"a": (1, 100), "b": (10, 1000)}, "op": "percent"},
    ],
    "word_problems": [
        {
            "question": "A store has {a} apples. If {b} customers each buy {c} apples, how many apples are left?",
            "vars": {"a": (50, 500), "b": (2, 20), "c": (1, 5)},
            "op": "store_apples"
        },
        {
            "question": "A train travels at {a} km/h for {b} hours. How far does it travel?",
            "vars": {"a": (40, 200), "b": (1, 10)},
            "op": "distance"
        },
        {
            "question": "If {a} workers can complete a job in {b} days, how many days would it take {c} workers?",
            "vars": {"a": (2, 10), "b": (5, 30), "c": (1, 20)},
            "op": "workers"
        },
        {
            "question": "A rectangle has length {a} cm and width {b} cm. What is its area?",
            "vars": {"a": (5, 50), "b": (3, 30)},
            "op": "area"
        },
        {
            "question": "John has ${a}. He spends ${b} on lunch and ${c} on a book. How much does he have left?",
            "vars": {"a": (50, 200), "b": (5, 30), "c": (10, 40)},
            "op": "money_left"
        },
    ],
    "algebra": [
        {
            "question": "Solve for x: {a}x + {b} = {c}",
            "vars": {"a": (2, 10), "b": (1, 20), "c": (10, 100)},
            "op": "linear_eq"
        },
        {
            "question": "If 2x + {a} = {b}, what is the value of x?",
            "vars": {"a": (1, 20), "b": (10, 50)},
            "op": "simple_linear"
        },
        {
            "question": "Find the value of {a}² + {b}².",
            "vars": {"a": (2, 20), "b": (2, 20)},
            "op": "sum_squares"
        },
    ],
    "sequences": [
        {
            "question": "What is the next number in the sequence: {seq}?",
            "vars": {"start": (1, 10), "step": (2, 10), "length": 5},
            "op": "arithmetic_seq"
        },
        {
            "question": "Find the sum of the first {n} positive integers.",
            "vars": {"n": (5, 50)},
            "op": "sum_n"
        },
    ],
}

# Logic puzzle templates
LOGIC_TEMPLATES = {
    "deduction": [
        {
            "question": "All {A} are {B}. All {B} are {C}. Is it true that all {A} are {C}?",
            "categories": [
                ("dogs", "mammals", "animals"),
                ("roses", "flowers", "plants"),
                ("squares", "rectangles", "shapes"),
                ("students", "learners", "people"),
                ("cars", "vehicles", "machines"),
            ],
            "answer": True,
            "type": "syllogism"
        },
        {
            "question": "Some {A} are {B}. All {B} are {C}. Can we conclude that some {A} are {C}?",
            "categories": [
                ("birds", "pets", "animals"),
                ("books", "bestsellers", "publications"),
                ("athletes", "professionals", "workers"),
            ],
            "answer": True,
            "type": "partial_syllogism"
        },
    ],
    "ordering": [
        {
            "question": "{A} is taller than {B}. {B} is taller than {C}. {D} is shorter than {C}. Who is the tallest?",
            "names": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank"],
            "type": "height_order"
        },
        {
            "question": "In a race, {A} finished before {B}. {C} finished after {B} but before {D}. Who finished second?",
            "names": ["Alex", "Ben", "Chris", "Dan", "Eve"],
            "type": "race_order"
        },
    ],
    "truth_tables": [
        {
            "question": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "answer": "Cannot be determined (affirming the consequent fallacy)",
            "type": "conditional"
        },
        {
            "question": "If A implies B, and B is false, what can we conclude about A?",
            "answer": "A must be false (modus tollens)",
            "type": "modus_tollens"
        },
    ],
}

# Code debugging templates
CODE_TEMPLATES = {
    "python_bugs": [
        {
            "code": '''def sum_list(numbers):
    total = 0
    for i in range(len(numbers)):
        total += numbers[i + 1]
    return total''',
            "bug": "Index out of range - should be numbers[i] not numbers[i + 1]",
            "language": "python"
        },
        {
            "code": '''def factorial(n):
    if n == 0:
        return 0
    return n * factorial(n - 1)''',
            "bug": "Base case should return 1, not 0",
            "language": "python"
        },
        {
            "code": '''def find_max(lst):
    max_val = 0
    for num in lst:
        if num > max_val:
            max_val = num
    return max_val''',
            "bug": "Initial max_val should be float('-inf') or lst[0] to handle negative numbers",
            "language": "python"
        },
        {
            "code": '''def reverse_string(s):
    reversed = ""
    for i in range(len(s), 0, -1):
        reversed += s[i]
    return reversed''',
            "bug": "Index out of range - should be s[i-1] or range should be (len(s)-1, -1, -1)",
            "language": "python"
        },
        {
            "code": '''def is_palindrome(s):
    return s == s.reverse()''',
            "bug": "Strings don't have .reverse() method - should use s[::-1]",
            "language": "python"
        },
    ],
    "javascript_bugs": [
        {
            "code": '''function sumArray(arr) {
    let sum;
    for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
    }
    return sum;
}''',
            "bug": "sum is undefined, should be initialized to 0",
            "language": "javascript"
        },
        {
            "code": '''function findIndex(arr, target) {
    for (let i = 0; i <= arr.length; i++) {
        if (arr[i] === target) return i;
    }
    return -1;
}''',
            "bug": "Loop condition should be i < arr.length (not <=) to avoid undefined access",
            "language": "javascript"
        },
    ],
}

# General reasoning templates
REASONING_TEMPLATES = {
    "cause_effect": [
        {
            "scenario": "The plants in the garden are wilting.",
            "observations": ["It hasn't rained in two weeks", "The soil is dry", "The leaves are turning brown"],
            "conclusion": "The plants need water due to drought conditions"
        },
        {
            "scenario": "The car won't start.",
            "observations": ["The dashboard lights don't turn on", "No sound when turning the key", "The headlights were left on overnight"],
            "conclusion": "The battery is dead from being drained overnight"
        },
        {
            "scenario": "Sales have decreased this quarter.",
            "observations": ["A new competitor entered the market", "Prices were increased by 15%", "Marketing budget was cut"],
            "conclusion": "Multiple factors including competition and pricing changes led to decreased sales"
        },
    ],
    "comparison": [
        {
            "question": "Compare the advantages and disadvantages of working from home vs. working in an office.",
            "aspects": ["productivity", "work-life balance", "collaboration", "costs", "flexibility"]
        },
        {
            "question": "What are the pros and cons of electric vehicles compared to gasoline vehicles?",
            "aspects": ["environmental impact", "cost", "range", "maintenance", "infrastructure"]
        },
    ],
    "problem_solving": [
        {
            "problem": "A company needs to reduce costs by 20% without laying off employees.",
            "constraints": ["No layoffs", "Maintain product quality", "Keep customer satisfaction high"],
            "approaches": ["Reduce overhead", "Optimize processes", "Renegotiate contracts", "Automate tasks"]
        },
        {
            "problem": "A student is struggling to balance work, school, and personal life.",
            "constraints": ["Must maintain grades", "Need income from work", "Want social life"],
            "approaches": ["Time management", "Prioritization", "Reduce commitments", "Seek support"]
        },
    ],
}

# Science and factual reasoning
SCIENCE_TEMPLATES = {
    "physics": [
        {
            "question": "A ball is dropped from a height of {h} meters. How long does it take to hit the ground? (g = 10 m/s²)",
            "vars": {"h": (5, 100)},
            "formula": "t = sqrt(2h/g)"
        },
        {
            "question": "What is the kinetic energy of an object with mass {m} kg moving at {v} m/s?",
            "vars": {"m": (1, 100), "v": (1, 50)},
            "formula": "KE = 0.5 * m * v²"
        },
    ],
    "chemistry": [
        {
            "question": "Balance the equation: H₂ + O₂ → H₂O",
            "answer": "2H₂ + O₂ → 2H₂O",
            "type": "balancing"
        },
        {
            "question": "What is the molecular weight of H₂O?",
            "answer": "18 g/mol (2×1 + 16)",
            "type": "molecular_weight"
        },
    ],
    "biology": [
        {
            "question": "Explain the process of photosynthesis.",
            "key_points": ["Light energy", "CO₂ + H₂O", "Glucose + O₂", "Chlorophyll"],
            "type": "process"
        },
    ],
}

# Multi-step problem templates
MULTI_STEP_TEMPLATES = {
    "planning": [
        {
            "goal": "Plan a birthday party for 20 people with a budget of $500",
            "steps": ["Set date and venue", "Create guest list", "Plan menu", "Arrange decorations", "Send invitations"],
            "constraints": ["Budget limit", "Dietary restrictions", "Space requirements"]
        },
        {
            "goal": "Prepare for a job interview",
            "steps": ["Research company", "Review job description", "Prepare answers", "Plan outfit", "Practice"],
            "constraints": ["Time available", "Interview format", "Required skills"]
        },
    ],
    "debugging_process": [
        {
            "issue": "Website is loading slowly",
            "diagnostic_steps": [
                "Check network connection",
                "Analyze page load times",
                "Review server logs",
                "Check database queries",
                "Examine resource sizes"
            ],
            "potential_causes": ["Large images", "Slow database", "Network issues", "Unoptimized code"]
        },
    ],
}

# Question types for variety
QUESTION_TYPES = [
    "explain",
    "compare",
    "analyze",
    "solve",
    "debug",
    "plan",
    "evaluate",
    "predict",
    "summarize",
    "classify",
]

# Difficulty levels
DIFFICULTY_LEVELS = {
    "easy": {"steps": (2, 3), "complexity": "simple"},
    "medium": {"steps": (3, 5), "complexity": "moderate"},
    "hard": {"steps": (5, 8), "complexity": "complex"},
}
