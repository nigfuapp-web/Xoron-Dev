"""
High-Quality Chain-of-Thought Dataset Generator

Generates diverse, high-quality synthetic reasoning examples with:
- Rich planning and analysis phases
- Self-critique and error checking
- Uncertainty quantification
- Structured reasoning with proper token usage
- Varied question phrasings and contexts
"""

import json
import os
import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

from config.special_tokens import SPECIAL_TOKENS


@dataclass
class ReasoningExample:
    """A single chain-of-thought reasoning example."""
    question: str
    thinking: str
    answer: str
    category: str
    difficulty: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Question phrasing variations for diversity
QUESTION_PHRASINGS = {
    "calculate": [
        "Calculate: {expr}",
        "What is {expr}?",
        "Find the value of {expr}",
        "Compute {expr}",
        "Evaluate: {expr}",
        "Solve: {expr}",
        "Work out {expr}",
    ],
    "solve_equation": [
        "Solve for x: {eq}",
        "Find x in the equation: {eq}",
        "What value of x satisfies {eq}?",
        "Determine x: {eq}",
        "Solve the equation {eq}",
    ],
    "word_problem": [
        "{problem}",
        "Here's a problem: {problem}",
        "Consider this scenario: {problem}",
        "Problem: {problem}",
    ],
    "logic": [
        "{premises} {question}",
        "Given: {premises}\nQuestion: {question}",
        "Consider the following: {premises}\n{question}",
    ],
    "code_debug": [
        "Find and fix the bug in this code:\n{code}",
        "This code has a bug. Identify and correct it:\n{code}",
        "Debug the following code:\n{code}",
        "What's wrong with this code and how do you fix it?\n{code}",
    ],
}

# Diverse names for word problems
NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Maya", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yuki", "Zara", "Ahmed", "Bella", "Carlos", "Diya", "Elena", "Fatima",
]

# Diverse items for shopping problems
SHOPPING_ITEMS = [
    ("apples", "apple", "$"), ("oranges", "orange", "$"), ("books", "book", "$"),
    ("pencils", "pencil", "$"), ("notebooks", "notebook", "$"), ("shirts", "shirt", "$"),
    ("toys", "toy", "$"), ("pens", "pen", "$"), ("markers", "marker", "$"),
    ("erasers", "eraser", "$"), ("folders", "folder", "$"), ("calculators", "calculator", "$"),
    ("headphones", "pair of headphones", "$"), ("batteries", "pack of batteries", "$"),
    ("water bottles", "water bottle", "$"), ("snacks", "snack pack", "$"),
]

# Vehicles for distance problems
VEHICLES = [
    ("car", "drives", "km/h"), ("train", "travels", "km/h"), ("bicycle", "rides", "km/h"),
    ("bus", "travels", "km/h"), ("airplane", "flies", "km/h"), ("boat", "sails", "km/h"),
    ("motorcycle", "rides", "km/h"), ("truck", "drives", "km/h"),
]


class ChainOfThoughtGenerator:
    """
    Generates high-quality synthetic chain-of-thought reasoning examples.
    
    Features:
    - Diverse question phrasings to avoid repetition
    - Rich reasoning with planning, analysis, and critique phases
    - Proper use of all reasoning tokens
    - Uncertainty quantification where appropriate
    - Self-correction examples
    - Varied difficulty levels with meaningful differences
    """
    
    def __init__(self, tokens: Dict[str, str] = None, seed: int = 42):
        self.tokens = tokens or SPECIAL_TOKENS
        self.rng = random.Random(seed)
        self.t = self.tokens
        
        # Track used combinations to ensure diversity
        self._used_hashes = set()
        
    def _get_unique_seed(self) -> int:
        """Get a unique seed for generating diverse examples."""
        return self.rng.randint(0, 2**31)
    
    def _phrase_question(self, category: str, **kwargs) -> str:
        """Get a varied phrasing for a question."""
        phrasings = QUESTION_PHRASINGS.get(category, ["{problem}"])
        template = self.rng.choice(phrasings)
        return template.format(**kwargs)
        
    def _wrap_think(self, content: str) -> str:
        """Wrap content in think tags."""
        return f"{self.t['think_start']}\n{content}\n{self.t['think_end']}"
    
    def _wrap_observation(self, content: str) -> str:
        """Wrap content in observation tags."""
        return f"{self.t['observation_start']}{content}{self.t['observation_end']}"
    
    def _wrap_note(self, content: str) -> str:
        """Wrap content in note tags."""
        return f"{self.t['note_start']}{content}{self.t['note_end']}"
    
    def _wrap_step(self, content: str) -> str:
        """Wrap content in step tags."""
        return f"{self.t['step_start']}{content}{self.t['step_end']}"
    
    def _wrap_reflection(self, content: str) -> str:
        """Wrap content in reflection tags."""
        return f"{self.t['reflection_start']}{content}{self.t['reflection_end']}"
    
    def _wrap_hypothesis(self, content: str) -> str:
        """Wrap content in hypothesis tags."""
        return f"{self.t['hypothesis_start']}{content}{self.t['hypothesis_end']}"
    
    def _wrap_conclusion(self, content: str) -> str:
        """Wrap content in conclusion tags."""
        return f"{self.t['conclusion_start']}{content}{self.t['conclusion_end']}"
    
    # === NEW ENHANCED REASONING METHODS ===
    
    def _wrap_plan(self, steps: List[str]) -> str:
        """Wrap planning steps in plan tags."""
        plan_steps = "\n".join([f"{self.t.get('plan_step', '')}{step}{self.t.get('plan_step_end', '')}" 
                               for step in steps])
        return f"{self.t.get('plan_start', '')}\n{plan_steps}\n{self.t.get('plan_end', '')}"
    
    def _wrap_critique(self, content: str, has_error: bool = False) -> str:
        """Wrap self-critique in critique tags."""
        error_marker = self.t.get('error_found', '') if has_error else self.t.get('no_error', '')
        return f"{self.t.get('critique_start', '')}\n{error_marker}{content}\n{self.t.get('critique_end', '')}"
    
    def _wrap_analysis(self, content: str) -> str:
        """Wrap analysis in analysis tags."""
        return f"{self.t.get('analysis_start', '')}{content}{self.t.get('analysis_end', '')}"
    
    def _wrap_decision(self, options: List[str], chosen_idx: int, reasoning: str) -> str:
        """Wrap decision making with options."""
        parts = [self.t.get('decision_start', '')]
        for i, opt in enumerate(options):
            marker = self.t.get('chosen', '') if i == chosen_idx else self.t.get('rejected', '')
            parts.append(f"{self.t.get('option_start', '')}{marker}{opt}{self.t.get('option_end', '')}")
        parts.append(f"{self.t.get('because', '')}{reasoning}")
        parts.append(self.t.get('decision_end', ''))
        return "\n".join(parts)
    
    def _add_uncertainty_score(self, content: str, score: int) -> str:
        """Add uncertainty score (0-100) to content."""
        score = max(0, min(100, score))
        return f"{self.t.get('uncertainty_score', '')}{score}{self.t.get('uncertainty_score_end', '')}{content}"
    
    def _add_confidence(self, content: str, level: str = 'high') -> str:
        """Add confidence level to content."""
        level_key = f"confidence_{level}"
        if level_key in self.t:
            return f"{self.t[level_key]}{content}"
        return content
    
    def _format_conversation(self, question: str, thinking: str, answer: str) -> str:
        """Format as a full conversation with thinking."""
        return (
            f"{self.t['user_start']}\n{question}\n{self.t['user_end']}\n"
            f"{self.t['assistant_start']}\n"
            f"{self._wrap_think(thinking)}\n"
            f"{answer}\n"
            f"{self.t['assistant_end']}"
        )
    
    # ==================== MATH GENERATORS ====================
    
    def generate_arithmetic(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate high-quality arithmetic problems with rich reasoning."""
        
        # Different problem types based on difficulty
        if difficulty == "easy":
            return self._generate_arithmetic_easy()
        elif difficulty == "medium":
            return self._generate_arithmetic_medium()
        else:
            return self._generate_arithmetic_hard()
    
    def _generate_arithmetic_easy(self) -> ReasoningExample:
        """Easy: Single operation with small numbers, clear steps."""
        ops = [("+", "addition", "add"), ("-", "subtraction", "subtract")]
        op_sym, op_name, op_verb = self.rng.choice(ops)
        
        a = self.rng.randint(5, 99)
        b = self.rng.randint(5, 99)
        
        if op_sym == "-" and a < b:
            a, b = b, a
        
        result = a + b if op_sym == "+" else a - b
        
        # Varied question phrasing
        question = self._phrase_question("calculate", expr=f"{a} {op_sym} {b}")
        
        # Rich thinking with planning
        thinking_parts = [
            self._wrap_plan([
                f"Identify the operation: {op_name}",
                f"Apply the operation to {a} and {b}",
                "Verify the result"
            ]),
            self._wrap_observation(f"This is a basic {op_name} problem: {a} {op_sym} {b}"),
            self._wrap_step(f"Performing {op_name}: {a} {op_sym} {b} = {result}"),
        ]
        
        # Add verification for subtraction
        if op_sym == "-":
            thinking_parts.append(self._wrap_reflection(f"Verification: {result} + {b} = {result + b} = {a} ✓"))
        else:
            thinking_parts.append(self._wrap_reflection(f"Verification: {result} - {b} = {result - b} = {a} ✓"))
        
        thinking_parts.append(self._wrap_conclusion(f"The answer is {result}"))
        
        # Varied answer phrasing
        answer_templates = [
            f"The result is **{result}**.",
            f"**{result}**",
            f"{a} {op_sym} {b} = **{result}**",
            f"The answer is **{result}**.",
        ]
        answer = self.rng.choice(answer_templates)
        
        return ReasoningExample(
            question=question,
            thinking="\n".join(thinking_parts),
            answer=answer,
            category="math_arithmetic",
            difficulty="easy",
            metadata={"operation": op_name, "operands": [a, b], "result": result}
        )
    
    def _generate_arithmetic_medium(self) -> ReasoningExample:
        """Medium: Multiple operations or larger numbers with detailed breakdown."""
        problem_type = self.rng.choice(["multi_op", "large_mult", "division", "mixed"])
        
        if problem_type == "multi_op":
            # Chain of operations: a + b - c or a × b + c
            a = self.rng.randint(10, 100)
            b = self.rng.randint(10, 100)
            c = self.rng.randint(5, 50)
            
            if self.rng.random() < 0.5:
                expr = f"{a} + {b} - {c}"
                result = a + b - c
                steps = [
                    f"First, add {a} + {b} = {a + b}",
                    f"Then, subtract {c}: {a + b} - {c} = {result}"
                ]
            else:
                expr = f"{a} × {b} + {c}"
                result = a * b + c
                steps = [
                    f"First, multiply {a} × {b} = {a * b}",
                    f"Then, add {c}: {a * b} + {c} = {result}"
                ]
        
        elif problem_type == "large_mult":
            a = self.rng.randint(12, 99)
            b = self.rng.randint(12, 99)
            result = a * b
            expr = f"{a} × {b}"
            
            # Break down multiplication
            a_tens, a_ones = a // 10 * 10, a % 10
            steps = [
                f"Break down {a} = {a_tens} + {a_ones}",
                f"{a_tens} × {b} = {a_tens * b}",
                f"{a_ones} × {b} = {a_ones * b}",
                f"Sum: {a_tens * b} + {a_ones * b} = {result}"
            ]
        
        elif problem_type == "division":
            divisor = self.rng.randint(3, 15)
            quotient = self.rng.randint(10, 100)
            dividend = divisor * quotient
            result = quotient
            expr = f"{dividend} ÷ {divisor}"
            steps = [
                f"Divide {dividend} by {divisor}",
                f"How many times does {divisor} go into {dividend}?",
                f"{divisor} × {quotient} = {dividend}, so the answer is {quotient}"
            ]
        
        else:  # mixed
            a = self.rng.randint(100, 999)
            b = self.rng.randint(100, 999)
            op = self.rng.choice(["+", "-"])
            if op == "-" and a < b:
                a, b = b, a
            result = a + b if op == "+" else a - b
            expr = f"{a} {op} {b}"
            steps = [
                f"Align the numbers by place value",
                f"{'Add' if op == '+' else 'Subtract'} ones: {a % 10} {op} {b % 10}",
                f"{'Add' if op == '+' else 'Subtract'} tens: {(a // 10) % 10} {op} {(b // 10) % 10}",
                f"{'Add' if op == '+' else 'Subtract'} hundreds: {a // 100} {op} {b // 100}",
                f"Result: {result}"
            ]
        
        question = self._phrase_question("calculate", expr=expr)
        
        thinking_parts = [
            self._wrap_plan([f"Analyze: {expr}", "Break down into steps", "Calculate", "Verify"]),
            self._wrap_analysis(f"Expression: {expr}"),
        ]
        
        for step in steps:
            thinking_parts.append(self._wrap_step(step))
        
        thinking_parts.append(self._wrap_critique("Checking the calculation...", has_error=False))
        thinking_parts.append(self._wrap_conclusion(f"The answer is {result}"))
        
        return ReasoningExample(
            question=question,
            thinking="\n".join(thinking_parts),
            answer=f"**{result}**",
            category="math_arithmetic",
            difficulty="medium",
            metadata={"expression": expr, "result": result}
        )
    
    def _generate_arithmetic_hard(self) -> ReasoningExample:
        """Hard: Complex expressions, order of operations, or estimation."""
        problem_type = self.rng.choice(["order_of_ops", "estimation", "percentage", "fraction"])
        
        if problem_type == "order_of_ops":
            a, b, c, d = [self.rng.randint(2, 20) for _ in range(4)]
            expr = f"{a} + {b} × {c} - {d}"
            result = a + b * c - d
            
            thinking_parts = [
                self._wrap_plan([
                    "Identify order of operations (PEMDAS/BODMAS)",
                    "Multiplication before addition/subtraction",
                    "Calculate step by step"
                ]),
                self._wrap_observation(f"Expression: {expr}"),
                self._wrap_note("Order of operations: Parentheses, Exponents, Multiplication/Division, Addition/Subtraction"),
                self._wrap_step(f"First, multiplication: {b} × {c} = {b * c}"),
                self._wrap_step(f"Rewrite: {a} + {b * c} - {d}"),
                self._wrap_step(f"Left to right: {a} + {b * c} = {a + b * c}"),
                self._wrap_step(f"Finally: {a + b * c} - {d} = {result}"),
                self._wrap_critique("Verified order of operations was followed correctly", has_error=False),
                self._wrap_conclusion(f"The answer is {result}")
            ]
        
        elif problem_type == "estimation":
            a = self.rng.randint(100, 9999)
            b = self.rng.randint(100, 9999)
            result = a + b
            
            # Round to nearest hundred for estimation
            a_est = round(a, -2)
            b_est = round(b, -2)
            est_result = a_est + b_est
            
            expr = f"{a} + {b}"
            question = f"Estimate {a} + {b}, then calculate the exact answer."
            
            thinking_parts = [
                self._wrap_plan(["Round numbers for estimation", "Calculate estimate", "Calculate exact", "Compare"]),
                self._wrap_step(f"Round {a} to nearest hundred: {a_est}"),
                self._wrap_step(f"Round {b} to nearest hundred: {b_est}"),
                self._wrap_step(f"Estimate: {a_est} + {b_est} = {est_result}"),
                self._wrap_step(f"Exact calculation: {a} + {b} = {result}"),
                self._wrap_reflection(f"Estimate {est_result} is close to exact {result} (difference: {abs(result - est_result)})"),
                self._wrap_conclusion(f"Estimate: {est_result}, Exact: {result}")
            ]
            
            return ReasoningExample(
                question=question,
                thinking="\n".join(thinking_parts),
                answer=f"Estimate: **{est_result}**, Exact: **{result}**",
                category="math_arithmetic",
                difficulty="hard",
                metadata={"estimate": est_result, "exact": result}
            )
        
        elif problem_type == "percentage":
            base = self.rng.choice([50, 100, 150, 200, 250, 500, 1000])
            percent = self.rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
            result = base * percent // 100
            
            question = f"What is {percent}% of {base}?"
            
            thinking_parts = [
                self._wrap_plan(["Convert percentage to decimal", "Multiply by base", "Simplify"]),
                self._wrap_observation(f"Finding {percent}% of {base}"),
                self._wrap_step(f"Convert: {percent}% = {percent}/100 = {percent/100}"),
                self._wrap_step(f"Multiply: {base} × {percent/100} = {result}"),
                self._wrap_reflection(f"Sanity check: {percent}% is {'less than half' if percent < 50 else 'half or more'} of {base}, and {result} {'is' if (result < base/2) == (percent < 50) else 'is not'} {'less than' if percent < 50 else 'at least'} {base//2} ✓"),
                self._wrap_conclusion(f"{percent}% of {base} is {result}")
            ]
            
            return ReasoningExample(
                question=question,
                thinking="\n".join(thinking_parts),
                answer=f"**{result}**",
                category="math_arithmetic",
                difficulty="hard",
                metadata={"base": base, "percent": percent, "result": result}
            )
        
        else:  # fraction
            # Simple fraction addition
            d1, d2 = self.rng.choice([(2, 4), (3, 6), (4, 8), (2, 3), (3, 4), (4, 5)])
            n1 = self.rng.randint(1, d1 - 1)
            n2 = self.rng.randint(1, d2 - 1)
            
            # Find LCD
            lcd = (d1 * d2) // math.gcd(d1, d2)
            n1_new = n1 * (lcd // d1)
            n2_new = n2 * (lcd // d2)
            result_n = n1_new + n2_new
            result_d = lcd
            
            # Simplify
            gcd = math.gcd(result_n, result_d)
            result_n //= gcd
            result_d //= gcd
            
            question = f"Add the fractions: {n1}/{d1} + {n2}/{d2}"
            
            thinking_parts = [
                self._wrap_plan(["Find common denominator", "Convert fractions", "Add numerators", "Simplify"]),
                self._wrap_observation(f"Adding {n1}/{d1} + {n2}/{d2}"),
                self._wrap_step(f"LCD of {d1} and {d2} is {lcd}"),
                self._wrap_step(f"Convert: {n1}/{d1} = {n1_new}/{lcd}"),
                self._wrap_step(f"Convert: {n2}/{d2} = {n2_new}/{lcd}"),
                self._wrap_step(f"Add: {n1_new}/{lcd} + {n2_new}/{lcd} = {n1_new + n2_new}/{lcd}"),
            ]
            
            if gcd > 1:
                thinking_parts.append(self._wrap_step(f"Simplify: {n1_new + n2_new}/{lcd} = {result_n}/{result_d}"))
            
            thinking_parts.append(self._wrap_conclusion(f"The sum is {result_n}/{result_d}"))
            
            return ReasoningExample(
                question=question,
                thinking="\n".join(thinking_parts),
                answer=f"**{result_n}/{result_d}**",
                category="math_arithmetic",
                difficulty="hard",
                metadata={"fractions": [f"{n1}/{d1}", f"{n2}/{d2}"], "result": f"{result_n}/{result_d}"}
            )
        
        question = self._phrase_question("calculate", expr=expr)
        
        return ReasoningExample(
            question=question,
            thinking="\n".join(thinking_parts),
            answer=f"**{result}**",
            category="math_arithmetic",
            difficulty="hard",
            metadata={"expression": expr, "result": result}
        )
    
    def generate_word_problem(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate word problems with detailed reasoning."""
        templates = [
            self._word_problem_shopping,
            self._word_problem_distance,
            self._word_problem_workers,
            self._word_problem_mixture,
            self._word_problem_age,
        ]
        
        generator = self.rng.choice(templates)
        return generator(difficulty)
    
    def _word_problem_shopping(self, difficulty: str) -> ReasoningExample:
        """Shopping-related word problem."""
        items = ["apples", "oranges", "books", "pencils", "notebooks", "shirts", "toys"]
        item = self.rng.choice(items)
        
        if difficulty == "easy":
            quantity = self.rng.randint(5, 20)
            price = self.rng.randint(1, 10)
        elif difficulty == "medium":
            quantity = self.rng.randint(10, 50)
            price = self.rng.randint(5, 25)
        else:
            quantity = self.rng.randint(20, 100)
            price = self.rng.randint(10, 50)
        
        total = quantity * price
        
        question = f"Sarah wants to buy {quantity} {item}. Each {item[:-1] if item.endswith('s') else item} costs ${price}. How much will she spend in total?"
        
        thinking_parts = [
            self._wrap_observation(f"Sarah is buying {quantity} {item} at ${price} each"),
            self._wrap_note("This is a multiplication problem: quantity × price = total"),
            self._wrap_step(f"Calculate: {quantity} × ${price}"),
            self._wrap_step(f"= ${quantity * price}"),
            self._wrap_reflection("Let me verify: if each item is ${price} and she buys {quantity}, the total should be ${total}"),
            self._wrap_conclusion(f"Sarah will spend ${total}")
        ]
        
        thinking = "\n".join(thinking_parts)
        answer = f"Sarah will spend **${total}** in total."
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_word_problem",
            difficulty=difficulty
        )
    
    def _word_problem_distance(self, difficulty: str) -> ReasoningExample:
        """Distance/speed/time word problem."""
        vehicles = ["car", "train", "bicycle", "bus", "airplane"]
        vehicle = self.rng.choice(vehicles)
        
        if difficulty == "easy":
            speed = self.rng.randint(20, 60)
            time = self.rng.randint(1, 5)
        elif difficulty == "medium":
            speed = self.rng.randint(40, 120)
            time = self.rng.randint(2, 8)
        else:
            speed = self.rng.randint(60, 200)
            time = self.rng.choice([1.5, 2.5, 3.5, 4.5])
        
        distance = speed * time
        
        question = f"A {vehicle} travels at {speed} km/h for {time} hours. How far does it travel?"
        
        thinking_parts = [
            self._wrap_observation(f"Given: speed = {speed} km/h, time = {time} hours"),
            self._wrap_note("Using the formula: distance = speed × time"),
            self._wrap_step(f"distance = {speed} km/h × {time} h"),
            self._wrap_step(f"distance = {distance} km"),
            self._wrap_conclusion(f"The {vehicle} travels {distance} km")
        ]
        
        thinking = "\n".join(thinking_parts)
        answer = f"The {vehicle} travels **{distance} km**."
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_word_problem",
            difficulty=difficulty
        )
    
    def _word_problem_workers(self, difficulty: str) -> ReasoningExample:
        """Work rate problem."""
        if difficulty == "easy":
            workers1 = self.rng.randint(2, 5)
            days1 = self.rng.randint(4, 10)
            workers2 = self.rng.randint(2, 5)
        elif difficulty == "medium":
            workers1 = self.rng.randint(3, 8)
            days1 = self.rng.randint(6, 15)
            workers2 = self.rng.randint(4, 12)
        else:
            workers1 = self.rng.randint(5, 15)
            days1 = self.rng.randint(10, 30)
            workers2 = self.rng.randint(8, 20)
        
        # Total work = workers × days
        total_work = workers1 * days1
        days2 = total_work / workers2
        
        question = f"If {workers1} workers can complete a job in {days1} days, how many days would it take {workers2} workers to complete the same job?"
        
        thinking_parts = [
            self._wrap_observation(f"{workers1} workers complete the job in {days1} days"),
            self._wrap_note("The total amount of work is constant"),
            self._wrap_step(f"Total work = {workers1} workers × {days1} days = {total_work} worker-days"),
            self._wrap_step(f"With {workers2} workers: days = {total_work} ÷ {workers2}"),
            self._wrap_step(f"days = {days2:.1f}"),
            self._wrap_reflection(f"Verification: {workers2} × {days2:.1f} = {workers2 * days2:.1f} worker-days ✓"),
            self._wrap_conclusion(f"It would take {days2:.1f} days")
        ]
        
        thinking = "\n".join(thinking_parts)
        answer = f"It would take **{days2:.1f} days** for {workers2} workers to complete the job."
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_word_problem",
            difficulty=difficulty
        )
    
    def _word_problem_mixture(self, difficulty: str) -> ReasoningExample:
        """Mixture/percentage problem."""
        if difficulty == "easy":
            total = self.rng.randint(50, 100)
            percent = self.rng.randint(10, 50)
        elif difficulty == "medium":
            total = self.rng.randint(100, 500)
            percent = self.rng.randint(15, 75)
        else:
            total = self.rng.randint(200, 1000)
            percent = self.rng.randint(5, 95)
        
        result = total * percent / 100
        
        question = f"What is {percent}% of {total}?"
        
        thinking_parts = [
            self._wrap_observation(f"Need to find {percent}% of {total}"),
            self._wrap_note("Percentage formula: (percent/100) × total"),
            self._wrap_step(f"= ({percent}/100) × {total}"),
            self._wrap_step(f"= {percent/100} × {total}"),
            self._wrap_step(f"= {result}"),
            self._wrap_conclusion(f"{percent}% of {total} is {result}")
        ]
        
        thinking = "\n".join(thinking_parts)
        answer = f"**{result}** is {percent}% of {total}."
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_word_problem",
            difficulty=difficulty
        )
    
    def _word_problem_age(self, difficulty: str) -> ReasoningExample:
        """Age-related word problem."""
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        name1, name2 = self.rng.sample(names, 2)
        
        if difficulty == "easy":
            age1 = self.rng.randint(10, 30)
            diff = self.rng.randint(2, 10)
        elif difficulty == "medium":
            age1 = self.rng.randint(15, 45)
            diff = self.rng.randint(5, 20)
        else:
            age1 = self.rng.randint(20, 60)
            diff = self.rng.randint(10, 30)
        
        age2 = age1 + diff
        total = age1 + age2
        
        question = f"{name1} is {diff} years older than {name2}. If {name2} is {age1} years old, what is the sum of their ages?"
        
        thinking_parts = [
            self._wrap_observation(f"{name2}'s age = {age1} years"),
            self._wrap_observation(f"{name1} is {diff} years older than {name2}"),
            self._wrap_step(f"{name1}'s age = {age1} + {diff} = {age2} years"),
            self._wrap_step(f"Sum of ages = {age1} + {age2}"),
            self._wrap_step(f"= {total} years"),
            self._wrap_conclusion(f"The sum of their ages is {total} years")
        ]
        
        thinking = "\n".join(thinking_parts)
        answer = f"The sum of their ages is **{total} years**."
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_word_problem",
            difficulty=difficulty
        )
    
    def generate_algebra(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate algebra problems."""
        if difficulty == "easy":
            # Simple: ax + b = c
            a = self.rng.randint(2, 5)
            x = self.rng.randint(1, 10)
            b = self.rng.randint(1, 20)
            c = a * x + b
            
            question = f"Solve for x: {a}x + {b} = {c}"
            
            thinking_parts = [
                self._wrap_observation(f"Linear equation: {a}x + {b} = {c}"),
                self._wrap_step(f"Subtract {b} from both sides: {a}x = {c} - {b}"),
                self._wrap_step(f"{a}x = {c - b}"),
                self._wrap_step(f"Divide both sides by {a}: x = {c - b} / {a}"),
                self._wrap_step(f"x = {x}"),
                self._wrap_reflection(f"Check: {a}({x}) + {b} = {a*x} + {b} = {c} ✓"),
                self._wrap_conclusion(f"x = {x}")
            ]
            
        elif difficulty == "medium":
            # ax + b = cx + d
            a = self.rng.randint(3, 8)
            c = self.rng.randint(1, a - 1)
            x = self.rng.randint(2, 15)
            b = self.rng.randint(1, 30)
            d = a * x + b - c * x
            
            question = f"Solve for x: {a}x + {b} = {c}x + {d}"
            
            thinking_parts = [
                self._wrap_observation(f"Equation with x on both sides: {a}x + {b} = {c}x + {d}"),
                self._wrap_step(f"Move x terms to left: {a}x - {c}x + {b} = {d}"),
                self._wrap_step(f"Combine like terms: {a-c}x + {b} = {d}"),
                self._wrap_step(f"Subtract {b}: {a-c}x = {d - b}"),
                self._wrap_step(f"Divide by {a-c}: x = {(d-b)/(a-c)}"),
                self._wrap_reflection(f"Verify: {a}({x}) + {b} = {a*x + b}, {c}({x}) + {d} = {c*x + d} ✓"),
                self._wrap_conclusion(f"x = {x}")
            ]
            
        else:
            # Quadratic: x² + bx + c = 0 (factorable)
            r1 = self.rng.randint(-10, 10)
            r2 = self.rng.randint(-10, 10)
            b = -(r1 + r2)
            c = r1 * r2
            
            b_str = f"+ {b}" if b >= 0 else f"- {abs(b)}"
            c_str = f"+ {c}" if c >= 0 else f"- {abs(c)}"
            
            question = f"Solve: x² {b_str}x {c_str} = 0"
            
            thinking_parts = [
                self._wrap_observation(f"Quadratic equation: x² {b_str}x {c_str} = 0"),
                self._wrap_note("Looking for factors of the form (x - r₁)(x - r₂)"),
                self._wrap_step(f"Need two numbers that multiply to {c} and add to {b}"),
                self._wrap_hypothesis(f"Try {-r1} and {-r2}: product = {(-r1)*(-r2)}, sum = {-r1-r2}"),
                self._wrap_step(f"Factored form: (x - {r1})(x - {r2}) = 0"),
                self._wrap_step(f"Solutions: x = {r1} or x = {r2}"),
                self._wrap_conclusion(f"x = {r1} or x = {r2}")
            ]
            
            answer = f"The solutions are **x = {r1}** and **x = {r2}**."
            
            return ReasoningExample(
                question=question,
                thinking="\n".join(thinking_parts),
                answer=answer,
                category="math_algebra",
                difficulty=difficulty
            )
        
        thinking = "\n".join(thinking_parts)
        answer = f"**x = {x}**"
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="math_algebra",
            difficulty=difficulty
        )
    
    # ==================== LOGIC GENERATORS ====================
    
    def generate_logic_deduction(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate logical deduction problems."""
        categories = [
            ("dogs", "mammals", "animals"),
            ("roses", "flowers", "plants"),
            ("squares", "rectangles", "quadrilaterals"),
            ("sparrows", "birds", "animals"),
            ("oak trees", "trees", "plants"),
            ("salmon", "fish", "animals"),
            ("Python", "programming languages", "languages"),
        ]
        
        A, B, C = self.rng.choice(categories)
        
        if difficulty == "easy":
            question = f"All {A} are {B}. All {B} are {C}. Are all {A} also {C}?"
            
            thinking_parts = [
                self._wrap_observation(f"Premise 1: All {A} are {B}"),
                self._wrap_observation(f"Premise 2: All {B} are {C}"),
                self._wrap_step(f"If something is a {A.rstrip('s')}, it must be a {B.rstrip('s')}"),
                self._wrap_step(f"If something is a {B.rstrip('s')}, it must be a {C.rstrip('s')}"),
                self._wrap_note("This is a transitive relationship"),
                self._wrap_conclusion(f"Therefore, all {A} are {C}")
            ]
            
            answer = f"Yes, all {A} are {C}. This follows from the transitive property of the 'is a' relationship."
            
        elif difficulty == "medium":
            question = f"Some {A} are {B}. All {B} are {C}. Can we conclude that some {A} are {C}?"
            
            thinking_parts = [
                self._wrap_observation(f"Premise 1: Some {A} are {B} (not all, just some)"),
                self._wrap_observation(f"Premise 2: All {B} are {C}"),
                self._wrap_step(f"The {A} that are {B} must also be {C} (from premise 2)"),
                self._wrap_note("Since at least some {A} are {B}, those same {A} are also {C}"),
                self._wrap_conclusion(f"Yes, we can conclude that some {A} are {C}")
            ]
            
            answer = f"Yes, some {A} are {C}. The {A} that belong to the {B} category must also be {C}."
            
        else:
            question = f"No {A} are {B}. Some {B} are {C}. What can we conclude about {A} and {C}?"
            
            thinking_parts = [
                self._wrap_observation(f"Premise 1: No {A} are {B} (complete exclusion)"),
                self._wrap_observation(f"Premise 2: Some {B} are {C}"),
                self._wrap_hypothesis("Can we conclude anything about {A} and {C}?"),
                self._wrap_step(f"The {B} that are {C} are definitely not {A}"),
                self._wrap_step(f"But there might be {C} that are not {B}"),
                self._wrap_reflection("We cannot determine the relationship between {A} and {C} from these premises alone"),
                self._wrap_conclusion("No definitive conclusion can be drawn about the relationship between {A} and {C}")
            ]
            
            answer = f"We cannot draw a definitive conclusion about the relationship between {A} and {C}. Some {C} might be {A}, or none might be."
        
        thinking = "\n".join(thinking_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="logic_deduction",
            difficulty=difficulty
        )
    
    def generate_logic_ordering(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate ordering/ranking logic problems."""
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        
        if difficulty == "easy":
            selected = self.rng.sample(names, 3)
            order = selected.copy()
            self.rng.shuffle(order)
            
            question = f"{order[0]} is taller than {order[1]}. {order[1]} is taller than {order[2]}. Who is the shortest?"
            
            thinking_parts = [
                self._wrap_observation(f"{order[0]} > {order[1]} (in height)"),
                self._wrap_observation(f"{order[1]} > {order[2]} (in height)"),
                self._wrap_step(f"Combining: {order[0]} > {order[1]} > {order[2]}"),
                self._wrap_conclusion(f"{order[2]} is the shortest")
            ]
            
            answer = f"**{order[2]}** is the shortest."
            
        elif difficulty == "medium":
            selected = self.rng.sample(names, 4)
            
            question = f"In a race: {selected[0]} finished before {selected[1]}. {selected[2]} finished after {selected[1]} but before {selected[3]}. Who finished second?"
            
            thinking_parts = [
                self._wrap_observation(f"{selected[0]} finished before {selected[1]}"),
                self._wrap_observation(f"{selected[2]} finished after {selected[1]} but before {selected[3]}"),
                self._wrap_step(f"From first clue: {selected[0]} → {selected[1]}"),
                self._wrap_step(f"From second clue: {selected[1]} → {selected[2]} → {selected[3]}"),
                self._wrap_step(f"Combined order: {selected[0]} → {selected[1]} → {selected[2]} → {selected[3]}"),
                self._wrap_conclusion(f"{selected[1]} finished second")
            ]
            
            answer = f"**{selected[1]}** finished second."
            
        else:
            selected = self.rng.sample(names, 5)
            
            question = (f"{selected[0]} is older than {selected[1]}. "
                       f"{selected[2]} is younger than {selected[1]} but older than {selected[3]}. "
                       f"{selected[4]} is the oldest. "
                       f"Rank everyone from oldest to youngest.")
            
            thinking_parts = [
                self._wrap_observation(f"{selected[0]} > {selected[1]} (age)"),
                self._wrap_observation(f"{selected[1]} > {selected[2]} > {selected[3]} (age)"),
                self._wrap_observation(f"{selected[4]} is the oldest"),
                self._wrap_step(f"Partial order: {selected[0]} > {selected[1]} > {selected[2]} > {selected[3]}"),
                self._wrap_step(f"{selected[4]} is above everyone"),
                self._wrap_hypothesis(f"Where does {selected[4]} fit with {selected[0]}?"),
                self._wrap_note(f"Since {selected[4]} is THE oldest, they're above {selected[0]}"),
                self._wrap_conclusion(f"Order: {selected[4]} > {selected[0]} > {selected[1]} > {selected[2]} > {selected[3]}")
            ]
            
            answer = f"From oldest to youngest: **{selected[4]} > {selected[0]} > {selected[1]} > {selected[2]} > {selected[3]}**"
        
        thinking = "\n".join(thinking_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="logic_ordering",
            difficulty=difficulty
        )
    
    # ==================== CODE DEBUGGING GENERATORS ====================
    
    def generate_code_debug(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate code debugging problems."""
        bugs = self._get_code_bugs(difficulty)
        bug = self.rng.choice(bugs)
        
        question = f"Find and fix the bug in this {bug['language']} code:\n\n```{bug['language']}\n{bug['code']}\n```"
        
        thinking_parts = [
            self._wrap_observation(f"Analyzing {bug['language']} code for bugs"),
            self._wrap_step("Reading through the code line by line"),
        ]
        
        for hint in bug.get('hints', []):
            thinking_parts.append(self._wrap_note(hint))
        
        thinking_parts.extend([
            self._wrap_hypothesis(f"The issue might be: {bug['bug']}"),
            self._wrap_reflection("Let me verify this is the actual problem"),
            self._wrap_conclusion(f"Bug found: {bug['bug']}")
        ])
        
        thinking = "\n".join(thinking_parts)
        answer = f"**Bug:** {bug['bug']}\n\n**Fixed code:**\n```{bug['language']}\n{bug['fixed']}\n```"
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="code_debug",
            difficulty=difficulty
        )
    
    def _get_code_bugs(self, difficulty: str) -> List[Dict]:
        """Get code bugs based on difficulty."""
        easy_bugs = [
            {
                "language": "python",
                "code": "def add(a, b):\n    return a - b",
                "bug": "Wrong operator: using subtraction instead of addition",
                "fixed": "def add(a, b):\n    return a + b",
                "hints": ["Function is named 'add'", "But the operation performed is subtraction"]
            },
            {
                "language": "python",
                "code": "def greet(name):\n    print('Hello, ' + Name)",
                "bug": "Variable name case mismatch: 'Name' should be 'name'",
                "fixed": "def greet(name):\n    print('Hello, ' + name)",
                "hints": ["Parameter is 'name' (lowercase)", "But 'Name' (uppercase) is used"]
            },
            {
                "language": "javascript",
                "code": "function multiply(a, b) {\n    return a * c;\n}",
                "bug": "Wrong variable: 'c' is undefined, should be 'b'",
                "fixed": "function multiply(a, b) {\n    return a * b;\n}",
                "hints": ["Parameters are 'a' and 'b'", "But 'c' is used which doesn't exist"]
            },
        ]
        
        medium_bugs = [
            {
                "language": "python",
                "code": "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)",
                "bug": "Base case returns 0 instead of 1",
                "fixed": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
                "hints": ["Factorial of 0 should be 1", "Current base case returns 0"]
            },
            {
                "language": "python",
                "code": "def find_max(lst):\n    max_val = 0\n    for num in lst:\n        if num > max_val:\n            max_val = num\n    return max_val",
                "bug": "Initial max_val = 0 fails for lists with all negative numbers",
                "fixed": "def find_max(lst):\n    max_val = float('-inf')\n    for num in lst:\n        if num > max_val:\n            max_val = num\n    return max_val",
                "hints": ["What if all numbers are negative?", "Initial value of 0 would be returned incorrectly"]
            },
            {
                "language": "python",
                "code": "def reverse_list(lst):\n    for i in range(len(lst)):\n        lst[i], lst[len(lst)-i] = lst[len(lst)-i], lst[i]\n    return lst",
                "bug": "Index out of range: should be len(lst)-1-i, and only iterate half the list",
                "fixed": "def reverse_list(lst):\n    for i in range(len(lst)//2):\n        lst[i], lst[len(lst)-1-i] = lst[len(lst)-1-i], lst[i]\n    return lst",
                "hints": ["lst[len(lst)] is out of bounds", "Also swapping twice reverses back"]
            },
        ]
        
        hard_bugs = [
            {
                "language": "python",
                "code": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1",
                "bug": "Infinite loop: left = mid should be left = mid + 1",
                "fixed": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid\n    return -1",
                "hints": ["When arr[mid] < target, we need to exclude mid", "left = mid can cause infinite loop"]
            },
            {
                "language": "python",
                "code": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result",
                "bug": "Missing remaining elements: need to add leftover elements from both arrays",
                "fixed": "def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
                "hints": ["What happens when one array is exhausted?", "Remaining elements are not added"]
            },
        ]
        
        if difficulty == "easy":
            return easy_bugs
        elif difficulty == "medium":
            return easy_bugs + medium_bugs
        else:
            return easy_bugs + medium_bugs + hard_bugs
    
    # ==================== GENERAL REASONING GENERATORS ====================
    
    def generate_cause_effect(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate cause-and-effect reasoning problems."""
        scenarios = [
            {
                "situation": "The plants in the garden are wilting",
                "observations": ["It hasn't rained in two weeks", "The soil feels dry", "Leaves are turning yellow"],
                "cause": "lack of water/drought",
                "solution": "Water the plants regularly"
            },
            {
                "situation": "The car won't start",
                "observations": ["Dashboard lights don't turn on", "No sound when turning key", "Headlights were left on overnight"],
                "cause": "dead battery",
                "solution": "Jump-start or replace the battery"
            },
            {
                "situation": "The website is loading slowly",
                "observations": ["Images take long to load", "Server response time is high", "Many users are online"],
                "cause": "server overload or unoptimized resources",
                "solution": "Optimize images, add caching, or scale servers"
            },
            {
                "situation": "Students are performing poorly on tests",
                "observations": ["Homework completion is low", "Class attendance has dropped", "Students seem distracted"],
                "cause": "lack of engagement or external factors affecting focus",
                "solution": "Investigate causes, adjust teaching methods, provide support"
            },
            {
                "situation": "The room feels cold despite the heater being on",
                "observations": ["Windows are old and drafty", "Thermostat shows correct temperature", "Cold air near windows"],
                "cause": "heat loss through poorly insulated windows",
                "solution": "Seal windows or upgrade to double-pane glass"
            },
        ]
        
        scenario = self.rng.choice(scenarios)
        
        if difficulty == "easy":
            obs_count = 2
        elif difficulty == "medium":
            obs_count = 3
        else:
            obs_count = len(scenario["observations"])
        
        observations = scenario["observations"][:obs_count]
        
        question = f"Situation: {scenario['situation']}\n\nObservations:\n" + "\n".join(f"- {obs}" for obs in observations) + "\n\nWhat is the likely cause and what should be done?"
        
        thinking_parts = [
            self._wrap_observation(f"Main issue: {scenario['situation']}"),
        ]
        
        for obs in observations:
            thinking_parts.append(self._wrap_observation(obs))
        
        thinking_parts.extend([
            self._wrap_note("Looking for patterns in the observations"),
            self._wrap_hypothesis(f"The cause might be: {scenario['cause']}"),
            self._wrap_reflection("This hypothesis explains all the observations"),
            self._wrap_conclusion(f"Cause: {scenario['cause']}. Solution: {scenario['solution']}")
        ])
        
        thinking = "\n".join(thinking_parts)
        answer = f"**Likely cause:** {scenario['cause']}\n\n**Recommended action:** {scenario['solution']}"
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="reasoning_cause_effect",
            difficulty=difficulty
        )
    
    def generate_comparison(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate comparison/analysis problems."""
        comparisons = [
            {
                "topic": "working from home vs. working in an office",
                "aspects": {
                    "productivity": ("Can be higher with fewer distractions", "Easier collaboration but more interruptions"),
                    "work-life balance": ("Better flexibility", "Clearer boundaries"),
                    "costs": ("Save on commute", "Company provides equipment"),
                    "social": ("Can feel isolated", "Regular social interaction"),
                }
            },
            {
                "topic": "electric vehicles vs. gasoline vehicles",
                "aspects": {
                    "environment": ("Zero direct emissions", "Produces CO2"),
                    "cost": ("Higher upfront, lower running", "Lower upfront, higher fuel costs"),
                    "range": ("Limited, improving", "Longer range, quick refuel"),
                    "maintenance": ("Fewer moving parts", "More complex maintenance"),
                }
            },
            {
                "topic": "online learning vs. traditional classroom",
                "aspects": {
                    "flexibility": ("Learn anytime, anywhere", "Fixed schedule"),
                    "interaction": ("Limited direct interaction", "Face-to-face engagement"),
                    "cost": ("Often cheaper", "Higher tuition and fees"),
                    "resources": ("Digital resources", "Physical labs and libraries"),
                }
            },
        ]
        
        comp = self.rng.choice(comparisons)
        
        if difficulty == "easy":
            aspects = dict(list(comp["aspects"].items())[:2])
        elif difficulty == "medium":
            aspects = dict(list(comp["aspects"].items())[:3])
        else:
            aspects = comp["aspects"]
        
        question = f"Compare the advantages and disadvantages of {comp['topic']}."
        
        thinking_parts = [
            self._wrap_observation(f"Comparing: {comp['topic']}"),
            self._wrap_note("Will analyze multiple aspects"),
        ]
        
        for aspect, (pro1, pro2) in aspects.items():
            thinking_parts.append(self._wrap_step(f"Analyzing {aspect}:"))
            thinking_parts.append(self._wrap_note(f"Option 1: {pro1}"))
            thinking_parts.append(self._wrap_note(f"Option 2: {pro2}"))
        
        thinking_parts.append(self._wrap_conclusion("Both options have trade-offs depending on priorities"))
        
        thinking = "\n".join(thinking_parts)
        
        answer_parts = [f"**Comparison of {comp['topic']}:**\n"]
        for aspect, (pro1, pro2) in aspects.items():
            answer_parts.append(f"**{aspect.title()}:**")
            answer_parts.append(f"- Option 1: {pro1}")
            answer_parts.append(f"- Option 2: {pro2}\n")
        
        answer = "\n".join(answer_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="reasoning_comparison",
            difficulty=difficulty
        )
    
    # ==================== MULTI-STEP PLANNING GENERATORS ====================
    
    def generate_planning(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate multi-step planning problems."""
        plans = [
            {
                "goal": "organize a birthday party for 20 people",
                "budget": "$500",
                "steps": [
                    "Set a date and time",
                    "Choose a venue (home or rented space)",
                    "Create guest list and send invitations",
                    "Plan the menu and order food",
                    "Arrange decorations and entertainment",
                    "Prepare party favors"
                ],
                "considerations": ["dietary restrictions", "space requirements", "entertainment for all ages"]
            },
            {
                "goal": "prepare for a job interview",
                "budget": None,
                "steps": [
                    "Research the company thoroughly",
                    "Review the job description",
                    "Prepare answers to common questions",
                    "Prepare questions to ask",
                    "Choose appropriate attire",
                    "Plan your route and timing"
                ],
                "considerations": ["company culture", "role requirements", "your unique strengths"]
            },
            {
                "goal": "learn a new programming language in 3 months",
                "budget": None,
                "steps": [
                    "Choose learning resources (books, courses, tutorials)",
                    "Set up development environment",
                    "Learn basic syntax and concepts",
                    "Practice with small projects",
                    "Build a portfolio project",
                    "Join community and seek feedback"
                ],
                "considerations": ["time available", "learning style", "practical applications"]
            },
            {
                "goal": "reduce monthly expenses by 20%",
                "budget": "Current spending",
                "steps": [
                    "Track all current expenses",
                    "Categorize spending",
                    "Identify non-essential expenses",
                    "Find alternatives for necessary expenses",
                    "Set up a budget",
                    "Monitor and adjust"
                ],
                "considerations": ["fixed vs variable costs", "quality of life", "long-term savings"]
            },
        ]
        
        plan = self.rng.choice(plans)
        
        if difficulty == "easy":
            steps = plan["steps"][:3]
            considerations = plan["considerations"][:1]
        elif difficulty == "medium":
            steps = plan["steps"][:5]
            considerations = plan["considerations"][:2]
        else:
            steps = plan["steps"]
            considerations = plan["considerations"]
        
        budget_str = f" with a budget of {plan['budget']}" if plan['budget'] else ""
        question = f"Create a plan to {plan['goal']}{budget_str}."
        
        thinking_parts = [
            self._wrap_observation(f"Goal: {plan['goal']}"),
        ]
        
        if plan['budget']:
            thinking_parts.append(self._wrap_note(f"Budget constraint: {plan['budget']}"))
        
        thinking_parts.append(self._wrap_step("Breaking down into actionable steps:"))
        
        for i, step in enumerate(steps, 1):
            thinking_parts.append(self._wrap_step(f"Step {i}: {step}"))
        
        thinking_parts.append(self._wrap_reflection("Considering important factors:"))
        for cons in considerations:
            thinking_parts.append(self._wrap_note(f"Must consider: {cons}"))
        
        thinking_parts.append(self._wrap_conclusion("Plan is comprehensive and actionable"))
        
        thinking = "\n".join(thinking_parts)
        
        answer_parts = [f"**Plan to {plan['goal']}:**\n"]
        for i, step in enumerate(steps, 1):
            answer_parts.append(f"{i}. {step}")
        answer_parts.append(f"\n**Key considerations:** {', '.join(considerations)}")
        
        answer = "\n".join(answer_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="reasoning_planning",
            difficulty=difficulty
        )
    
    # ==================== SCIENCE GENERATORS ====================
    
    def generate_science(self, difficulty: str = "medium") -> ReasoningExample:
        """Generate science reasoning problems."""
        generators = [
            self._science_physics,
            self._science_chemistry,
            self._science_biology,
        ]
        
        generator = self.rng.choice(generators)
        return generator(difficulty)
    
    def _science_physics(self, difficulty: str) -> ReasoningExample:
        """Physics problem."""
        if difficulty == "easy":
            # Simple speed calculation
            speed = self.rng.randint(10, 50)
            time = self.rng.randint(1, 5)
            distance = speed * time
            
            question = f"A car travels at {speed} m/s for {time} seconds. How far does it travel?"
            
            thinking_parts = [
                self._wrap_observation(f"Speed = {speed} m/s, Time = {time} s"),
                self._wrap_note("Using formula: distance = speed × time"),
                self._wrap_step(f"distance = {speed} × {time}"),
                self._wrap_step(f"distance = {distance} m"),
                self._wrap_conclusion(f"The car travels {distance} meters")
            ]
            
            answer = f"The car travels **{distance} meters**."
            
        elif difficulty == "medium":
            # Free fall
            height = self.rng.randint(20, 100)
            g = 10  # simplified
            time = math.sqrt(2 * height / g)
            
            question = f"A ball is dropped from a height of {height} meters. How long does it take to hit the ground? (Use g = 10 m/s²)"
            
            thinking_parts = [
                self._wrap_observation(f"Height h = {height} m, g = 10 m/s²"),
                self._wrap_note("For free fall: h = ½gt²"),
                self._wrap_step("Solving for t: t = √(2h/g)"),
                self._wrap_step(f"t = √(2 × {height} / 10)"),
                self._wrap_step(f"t = √({2 * height / 10})"),
                self._wrap_step(f"t ≈ {time:.2f} seconds"),
                self._wrap_conclusion(f"It takes approximately {time:.2f} seconds")
            ]
            
            answer = f"It takes approximately **{time:.2f} seconds** to hit the ground."
            
        else:
            # Kinetic energy
            mass = self.rng.randint(5, 50)
            velocity = self.rng.randint(5, 30)
            ke = 0.5 * mass * velocity ** 2
            
            question = f"Calculate the kinetic energy of an object with mass {mass} kg moving at {velocity} m/s."
            
            thinking_parts = [
                self._wrap_observation(f"Mass m = {mass} kg, Velocity v = {velocity} m/s"),
                self._wrap_note("Kinetic energy formula: KE = ½mv²"),
                self._wrap_step(f"KE = ½ × {mass} × {velocity}²"),
                self._wrap_step(f"KE = ½ × {mass} × {velocity ** 2}"),
                self._wrap_step(f"KE = {0.5 * mass} × {velocity ** 2}"),
                self._wrap_step(f"KE = {ke} J"),
                self._wrap_conclusion(f"The kinetic energy is {ke} Joules")
            ]
            
            answer = f"The kinetic energy is **{ke} Joules**."
        
        thinking = "\n".join(thinking_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="science_physics",
            difficulty=difficulty
        )
    
    def _science_chemistry(self, difficulty: str) -> ReasoningExample:
        """Chemistry problem."""
        if difficulty == "easy":
            question = "What is the molecular formula for water, and what atoms does it contain?"
            
            thinking_parts = [
                self._wrap_observation("Water is a common compound"),
                self._wrap_note("Water consists of hydrogen and oxygen"),
                self._wrap_step("Each water molecule has 2 hydrogen atoms"),
                self._wrap_step("Each water molecule has 1 oxygen atom"),
                self._wrap_conclusion("Formula: H₂O (2 hydrogen, 1 oxygen)")
            ]
            
            answer = "Water has the molecular formula **H₂O**, containing 2 hydrogen atoms and 1 oxygen atom."
            
        elif difficulty == "medium":
            question = "Balance the chemical equation: H₂ + O₂ → H₂O"
            
            thinking_parts = [
                self._wrap_observation("Unbalanced equation: H₂ + O₂ → H₂O"),
                self._wrap_step("Count atoms on each side:"),
                self._wrap_note("Left: 2 H, 2 O"),
                self._wrap_note("Right: 2 H, 1 O"),
                self._wrap_step("Oxygen is unbalanced (2 vs 1)"),
                self._wrap_step("Put coefficient 2 before H₂O: H₂ + O₂ → 2H₂O"),
                self._wrap_note("Now right side: 4 H, 2 O"),
                self._wrap_step("Balance hydrogen with coefficient 2: 2H₂ + O₂ → 2H₂O"),
                self._wrap_reflection("Check: Left: 4 H, 2 O. Right: 4 H, 2 O ✓"),
                self._wrap_conclusion("Balanced: 2H₂ + O₂ → 2H₂O")
            ]
            
            answer = "**Balanced equation:** 2H₂ + O₂ → 2H₂O"
            
        else:
            question = "Calculate the molecular weight of glucose (C₆H₁₂O₆). (C=12, H=1, O=16)"
            
            thinking_parts = [
                self._wrap_observation("Glucose formula: C₆H₁₂O₆"),
                self._wrap_note("Atomic weights: C=12, H=1, O=16"),
                self._wrap_step("Carbon contribution: 6 × 12 = 72"),
                self._wrap_step("Hydrogen contribution: 12 × 1 = 12"),
                self._wrap_step("Oxygen contribution: 6 × 16 = 96"),
                self._wrap_step("Total: 72 + 12 + 96 = 180"),
                self._wrap_conclusion("Molecular weight of glucose = 180 g/mol")
            ]
            
            answer = "The molecular weight of glucose (C₆H₁₂O₆) is **180 g/mol**."
        
        thinking = "\n".join(thinking_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="science_chemistry",
            difficulty=difficulty
        )
    
    def _science_biology(self, difficulty: str) -> ReasoningExample:
        """Biology problem."""
        if difficulty == "easy":
            question = "What is the main function of red blood cells?"
            
            thinking_parts = [
                self._wrap_observation("Red blood cells are found in blood"),
                self._wrap_note("They contain hemoglobin"),
                self._wrap_step("Hemoglobin binds to oxygen"),
                self._wrap_step("Red blood cells transport oxygen from lungs to tissues"),
                self._wrap_conclusion("Main function: oxygen transport")
            ]
            
            answer = "The main function of red blood cells is to **transport oxygen** from the lungs to body tissues using hemoglobin."
            
        elif difficulty == "medium":
            question = "Explain the basic process of photosynthesis."
            
            thinking_parts = [
                self._wrap_observation("Photosynthesis occurs in plants"),
                self._wrap_note("Takes place in chloroplasts"),
                self._wrap_step("Inputs: CO₂ (from air) + H₂O (from soil) + light energy"),
                self._wrap_step("Chlorophyll absorbs light energy"),
                self._wrap_step("Energy used to convert CO₂ and H₂O"),
                self._wrap_step("Outputs: Glucose (C₆H₁₂O₆) + O₂"),
                self._wrap_conclusion("6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂")
            ]
            
            answer = "**Photosynthesis** is the process where plants convert carbon dioxide and water into glucose and oxygen using light energy. The equation is: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂"
            
        else:
            question = "Describe the stages of mitosis and their key events."
            
            thinking_parts = [
                self._wrap_observation("Mitosis is cell division for growth/repair"),
                self._wrap_note("Results in two identical daughter cells"),
                self._wrap_step("1. Prophase: Chromosomes condense, nuclear envelope breaks down"),
                self._wrap_step("2. Metaphase: Chromosomes align at cell equator"),
                self._wrap_step("3. Anaphase: Sister chromatids separate, move to poles"),
                self._wrap_step("4. Telophase: Nuclear envelopes reform, chromosomes decondense"),
                self._wrap_note("Cytokinesis follows: cytoplasm divides"),
                self._wrap_conclusion("Four stages: Prophase → Metaphase → Anaphase → Telophase")
            ]
            
            answer = """**Stages of Mitosis:**
1. **Prophase:** Chromosomes condense, nuclear envelope breaks down
2. **Metaphase:** Chromosomes align at the cell's equator
3. **Anaphase:** Sister chromatids separate and move to opposite poles
4. **Telophase:** Nuclear envelopes reform, chromosomes decondense

This is followed by cytokinesis (division of cytoplasm)."""
        
        thinking = "\n".join(thinking_parts)
        
        return ReasoningExample(
            question=question,
            thinking=thinking,
            answer=answer,
            category="science_biology",
            difficulty=difficulty
        )
    
    # ==================== MAIN GENERATION METHOD ====================
    
    def generate_example(self, category: str = None, difficulty: str = None) -> ReasoningExample:
        """Generate a single reasoning example."""
        categories = {
            "math_arithmetic": self.generate_arithmetic,
            "math_word_problem": self.generate_word_problem,
            "math_algebra": self.generate_algebra,
            "logic_deduction": self.generate_logic_deduction,
            "logic_ordering": self.generate_logic_ordering,
            "code_debug": self.generate_code_debug,
            "reasoning_cause_effect": self.generate_cause_effect,
            "reasoning_comparison": self.generate_comparison,
            "reasoning_planning": self.generate_planning,
            "science": self.generate_science,
        }
        
        if category is None:
            category = self.rng.choice(list(categories.keys()))
        
        if difficulty is None:
            difficulty = self.rng.choice(["easy", "medium", "hard"])
        
        generator = categories.get(category, self.generate_arithmetic)
        return generator(difficulty)
    
    def generate_batch(self, count: int, categories: List[str] = None, 
                       difficulties: List[str] = None) -> List[ReasoningExample]:
        """Generate a batch of reasoning examples."""
        if categories is None:
            categories = [
                "math_arithmetic", "math_word_problem", "math_algebra",
                "logic_deduction", "logic_ordering", "code_debug",
                "reasoning_cause_effect", "reasoning_comparison",
                "reasoning_planning", "science"
            ]
        
        if difficulties is None:
            difficulties = ["easy", "medium", "hard"]
        
        examples = []
        for _ in range(count):
            cat = self.rng.choice(categories)
            diff = self.rng.choice(difficulties)
            example = self.generate_example(cat, diff)
            examples.append(example)
        
        return examples
    
    def to_training_format(self, example: ReasoningExample) -> Dict[str, Any]:
        """Convert example to training format."""
        text = self._format_conversation(example.question, example.thinking, example.answer)
        
        return {
            "text": text,
            "type": "chain_of_thought",
            "category": example.category,
            "difficulty": example.difficulty,
            "metadata": example.metadata or {}
        }
    
    def generate_dataset(
        self,
        total_examples: int,
        output_path: str,
        batch_size: int = 1000,
        categories: List[str] = None,
        difficulties: List[str] = None,
        show_progress: bool = True
    ) -> str:
        """
        Generate a complete dataset and save to JSONL file.
        
        Args:
            total_examples: Total number of examples to generate
            output_path: Path to save the JSONL file
            batch_size: Number of examples per batch (for progress reporting)
            categories: List of categories to include
            difficulties: List of difficulties to include
            show_progress: Whether to show progress
            
        Returns:
            Path to the generated file
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        generated = 0
        seen_hashes = set()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            while generated < total_examples:
                batch_count = min(batch_size, total_examples - generated)
                examples = self.generate_batch(batch_count, categories, difficulties)
                
                for example in examples:
                    # Deduplicate based on question hash
                    q_hash = hashlib.md5(example.question.encode()).hexdigest()
                    if q_hash in seen_hashes:
                        continue
                    seen_hashes.add(q_hash)
                    
                    training_data = self.to_training_format(example)
                    f.write(json.dumps(training_data, ensure_ascii=False) + '\n')
                    generated += 1
                    
                    if generated >= total_examples:
                        break
                
                if show_progress:
                    print(f"Generated {generated}/{total_examples} examples ({100*generated/total_examples:.1f}%)")
        
        print(f"✅ Dataset saved to {output_path}")
        print(f"   Total examples: {generated}")
        
        return output_path


def generate_worker(args):
    """Worker function for parallel generation."""
    seed, count, categories, difficulties = args
    generator = ChainOfThoughtGenerator(seed=seed)
    examples = generator.generate_batch(count, categories, difficulties)
    return [generator.to_training_format(ex) for ex in examples]
