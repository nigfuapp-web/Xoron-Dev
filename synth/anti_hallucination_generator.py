#!/usr/bin/env python3
"""
High-Quality Anti-Hallucination Dataset Generator

Generates diverse, high-quality training data to teach the model:
1. To say "I don't know" when appropriate (with varied phrasings)
2. To express uncertainty levels with numerical scores
3. To cite sources and ground responses properly
4. To self-verify and correct mistakes with critique tokens
5. To recognize knowledge boundaries
6. To use planning and analysis for fact-checking

Usage:
    python -m synth.anti_hallucination_generator
"""

import os
import sys
import json
import random
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.special_tokens import SPECIAL_TOKENS
from synth.quality_utils import QualityGenerator, NAMES, COMPANIES, TOPICS

# Questions that should trigger "I don't know" responses - EXPANDED
UNKNOWABLE_QUESTIONS = [
    # Future predictions
    "What will the stock market do tomorrow?",
    "What are the winning lottery numbers for next week?",
    "What will happen in the year 3000?",
    "What will be the next major scientific discovery?",
    "Who will win the next election?",
    "What will the weather be exactly one year from now?",
    "Which company will be the most valuable in 2050?",
    "What new technology will be invented next year?",
    "Will there be a recession next quarter?",
    "What will be the price of Bitcoin in 5 years?",
    "Which sports team will win the championship next season?",
    "What will be the next pandemic?",
    "When will humans land on Mars?",
    "What will be the population of Earth in 2100?",
    
    # Personal/private information
    "What is my neighbor thinking right now?",
    "What did I have for breakfast yesterday?",
    "What is my password?",
    "What are my private thoughts?",
    "What is my bank account balance?",
    "What did my friend say about me?",
    "What is my social security number?",
    "What are my medical records?",
    "What is my credit card number?",
    "What did I dream about last night?",
    "What is my secret?",
    "What is my phone number?",
    
    # Fundamentally unknowable
    "What is the exact number of grains of sand on Earth?",
    "What is the cure for all diseases?",
    "What happens after we die?",
    "What is the last digit of pi?",
    "What is the meaning of life for you personally?",
    "Is there a god?",
    "What is consciousness?",
    "Are we living in a simulation?",
    "What existed before the Big Bang?",
    "Is there intelligent life elsewhere in the universe?",
    "What is the nature of dark matter?",
    "Why is there something rather than nothing?",
    "Do parallel universes exist?",
    "What is the ultimate fate of the universe?",
    
    # Specific unknowable facts
    "How many birds are flying right now?",
    "What is the exact temperature at the center of the Earth?",
    "How many thoughts do humans have per day?",
    "What is the exact age of the universe in seconds?",
    "How many stars are in the observable universe exactly?",
    "What is every person on Earth doing right now?",
    "How many fish are in the ocean?",
    "What is the exact weight of all the water on Earth?",
]

# Varied "I don't know" response templates
IDK_RESPONSES = [
    "I don't have access to that information.",
    "I cannot predict future events with certainty.",
    "This is beyond my knowledge and capabilities.",
    "I don't know the answer to this question.",
    "I cannot access personal or private information.",
    "This question asks about something I cannot determine.",
    "I'm not able to provide this information.",
    "This is unknowable or unpredictable.",
    "I don't have the ability to know this.",
    "This falls outside what I can reliably answer.",
    "I cannot make predictions about this.",
    "I don't have access to real-time or personal data.",
    "This requires information I don't possess.",
    "I'm unable to answer questions about private matters.",
    "This is fundamentally uncertain or unknowable.",
]

# Explanations for why something is unknowable
IDK_EXPLANATIONS = [
    "This involves predicting the future, which is inherently uncertain.",
    "I don't have access to personal or private information.",
    "This requires real-time data that I don't have.",
    "This is a philosophical question without a definitive answer.",
    "The information needed to answer this doesn't exist or isn't accessible.",
    "This involves randomness that cannot be predicted.",
    "I cannot access private databases or personal records.",
    "This question asks about subjective experiences I cannot know.",
    "The future is inherently unpredictable.",
    "This requires omniscience that I don't possess.",
]

# Questions about recent events (after knowledge cutoff)
RECENT_EVENT_QUESTIONS = [
    "What happened in the news today?",
    "What is the current price of Bitcoin?",
    "Who won the game last night?",
    "What is the latest iPhone model?",
    "What are the current COVID statistics?",
    "What is the current US president doing right now?",
    "What movies came out this week?",
    "What is the current weather in New York?",
    "What are the latest stock prices?",
    "What happened at the conference yesterday?",
]

# Questions that require verification
VERIFICATION_NEEDED_QUESTIONS = [
    ("The Great Wall of China is visible from space.", False, "This is a common misconception. The Great Wall is not visible from space with the naked eye."),
    ("Humans only use 10% of their brain.", False, "This is a myth. Brain imaging shows we use virtually all parts of our brain."),
    ("Lightning never strikes the same place twice.", False, "This is false. Lightning frequently strikes the same location, especially tall structures."),
    ("Goldfish have a 3-second memory.", False, "This is a myth. Goldfish can remember things for months."),
    ("The capital of Australia is Sydney.", False, "This is incorrect. The capital of Australia is Canberra, not Sydney."),
    ("Water conducts electricity.", True, "Pure water is actually a poor conductor. It's the impurities and dissolved minerals that conduct electricity."),
    ("Mount Everest is the tallest mountain on Earth.", True, "Measured from sea level, yes. However, Mauna Kea is taller when measured from its base on the ocean floor."),
]

# Questions with grounded context - EXPANDED
GROUNDED_QA_PAIRS = [
    {"context": "According to NASA's official website, the Moon is approximately 384,400 kilometers from Earth on average.", "question": "How far is the Moon from Earth?", "answer": "The Moon is approximately 384,400 kilometers from Earth on average.", "source": "NASA official website"},
    {"context": "The Python documentation states that lists are mutable sequences, typically used to store collections of homogeneous items.", "question": "What are Python lists used for?", "answer": "Python lists are mutable sequences typically used to store collections of homogeneous items.", "source": "Python official documentation"},
    {"context": "Research published in Nature (2023) found that regular exercise can improve cognitive function by up to 20% in older adults.", "question": "Does exercise help cognitive function?", "answer": "Yes, research shows regular exercise can improve cognitive function by up to 20% in older adults.", "source": "Nature (2023)"},
    {"context": "The World Health Organization reports that approximately 1.35 million people die each year as a result of road traffic crashes.", "question": "How many people die from road accidents yearly?", "answer": "Approximately 1.35 million people die each year from road traffic crashes.", "source": "World Health Organization"},
    {"context": "According to the Rust documentation, ownership is Rust's most unique feature and enables memory safety without garbage collection.", "question": "What is Rust's ownership system?", "answer": "Ownership is Rust's most unique feature that enables memory safety without garbage collection.", "source": "Rust documentation"},
    {"context": "The IEEE reports that Python has been the most popular programming language for several consecutive years.", "question": "What is the most popular programming language?", "answer": "According to IEEE, Python has been the most popular programming language for several consecutive years.", "source": "IEEE"},
    {"context": "Research from MIT shows that transformer models can process sequences in parallel, unlike RNNs which process sequentially.", "question": "How do transformers differ from RNNs?", "answer": "Transformers can process sequences in parallel, while RNNs process sequentially.", "source": "MIT research"},
    {"context": "The CDC states that vaccines have prevented an estimated 154 million deaths over the past 50 years.", "question": "How many deaths have vaccines prevented?", "answer": "Vaccines have prevented an estimated 154 million deaths over the past 50 years.", "source": "CDC"},
    {"context": "According to the Linux kernel documentation, the kernel uses a monolithic architecture with loadable modules.", "question": "What architecture does the Linux kernel use?", "answer": "The Linux kernel uses a monolithic architecture with loadable modules.", "source": "Linux kernel documentation"},
    {"context": "The PostgreSQL documentation explains that MVCC (Multi-Version Concurrency Control) allows concurrent transactions without locking.", "question": "How does PostgreSQL handle concurrent transactions?", "answer": "PostgreSQL uses MVCC (Multi-Version Concurrency Control) to allow concurrent transactions without locking.", "source": "PostgreSQL documentation"},
    {"context": "According to the HTTP/2 specification, multiplexing allows multiple requests and responses to be sent over a single TCP connection.", "question": "What is HTTP/2 multiplexing?", "answer": "HTTP/2 multiplexing allows multiple requests and responses to be sent over a single TCP connection.", "source": "HTTP/2 specification"},
    {"context": "The React documentation states that hooks let you use state and other React features without writing a class.", "question": "What do React hooks do?", "answer": "React hooks let you use state and other React features without writing a class.", "source": "React documentation"},
    {"context": "Research from Stanford shows that large language models can exhibit emergent abilities at certain scale thresholds.", "question": "What are emergent abilities in LLMs?", "answer": "Large language models can exhibit emergent abilities - capabilities that appear suddenly at certain scale thresholds.", "source": "Stanford research"},
    {"context": "The Git documentation explains that a commit is a snapshot of your repository at a specific point in time.", "question": "What is a Git commit?", "answer": "A Git commit is a snapshot of your repository at a specific point in time.", "source": "Git documentation"},
    {"context": "According to AWS documentation, Lambda functions can scale automatically from a few requests per day to thousands per second.", "question": "How does AWS Lambda scale?", "answer": "Lambda functions scale automatically from a few requests per day to thousands per second.", "source": "AWS documentation"},
    {"context": "The Kubernetes documentation states that pods are the smallest deployable units that can be created and managed.", "question": "What are Kubernetes pods?", "answer": "Pods are the smallest deployable units in Kubernetes that can be created and managed.", "source": "Kubernetes documentation"},
    {"context": "Research published in Science shows that CRISPR-Cas9 can edit genes with unprecedented precision.", "question": "What is CRISPR-Cas9?", "answer": "CRISPR-Cas9 is a gene editing technology that can edit genes with unprecedented precision.", "source": "Science journal"},
    {"context": "The TypeScript handbook explains that interfaces define contracts in your code and provide explicit names for type checking.", "question": "What are TypeScript interfaces?", "answer": "TypeScript interfaces define contracts in your code and provide explicit names for type checking.", "source": "TypeScript handbook"},
    {"context": "According to the Docker documentation, containers share the host OS kernel, making them more lightweight than VMs.", "question": "How are containers different from VMs?", "answer": "Containers share the host OS kernel, making them more lightweight than virtual machines.", "source": "Docker documentation"},
    {"context": "The TensorFlow documentation states that eager execution evaluates operations immediately without building graphs.", "question": "What is TensorFlow eager execution?", "answer": "Eager execution evaluates operations immediately without building computational graphs.", "source": "TensorFlow documentation"},
]

# Self-correction examples - EXPANDED
SELF_CORRECTION_EXAMPLES = [
    {"initial_wrong": "The speed of light is approximately 300,000 miles per second.", "correction": "I need to correct myself. The speed of light is approximately 300,000 kilometers per second (or about 186,000 miles per second), not 300,000 miles per second."},
    {"initial_wrong": "Python was created by Guido van Rossum in 1995.", "correction": "Let me correct that. Python was first released in 1991, not 1995. Guido van Rossum started working on it in the late 1980s."},
    {"initial_wrong": "The human body has 206 bones, and this number stays the same throughout life.", "correction": "I should clarify: while adults have 206 bones, babies are born with about 270 bones. Many of these fuse together as we grow."},
    {"initial_wrong": "JavaScript and Java are the same language.", "correction": "That's incorrect. JavaScript and Java are completely different languages. JavaScript was named to capitalize on Java's popularity, but they have different syntax, use cases, and origins."},
    {"initial_wrong": "The Great Wall of China is visible from space.", "correction": "This is a common myth. The Great Wall is not visible from space with the naked eye. It's too narrow, despite being very long."},
    {"initial_wrong": "Einstein failed math in school.", "correction": "This is a myth. Einstein excelled at mathematics. He mastered calculus by age 15 and never failed math."},
    {"initial_wrong": "Humans only use 10% of their brain.", "correction": "This is false. Brain imaging shows we use virtually all parts of our brain, just not all at once. Different regions are active for different tasks."},
    {"initial_wrong": "SQL is a programming language.", "correction": "More precisely, SQL is a query language, not a general-purpose programming language. It's designed specifically for managing relational databases."},
    {"initial_wrong": "HTTP is a secure protocol.", "correction": "HTTP itself is not secure - data is transmitted in plain text. HTTPS (HTTP Secure) adds TLS/SSL encryption for security."},
    {"initial_wrong": "RAM stores data permanently.", "correction": "RAM (Random Access Memory) is volatile memory - it loses all data when power is turned off. Permanent storage uses SSDs, HDDs, or flash memory."},
    {"initial_wrong": "Machine learning and AI are the same thing.", "correction": "Machine learning is a subset of AI. AI is the broader concept of machines mimicking human intelligence, while ML specifically refers to systems that learn from data."},
    {"initial_wrong": "The Internet and the World Wide Web are the same.", "correction": "They're different. The Internet is the global network infrastructure. The World Wide Web is a service that runs on the Internet, using HTTP to access web pages."},
    {"initial_wrong": "Linux was created by Linus Torvalds in 1995.", "correction": "Linux was first released in 1991, not 1995. Linus Torvalds announced it on August 25, 1991."},
    {"initial_wrong": "Git was created by GitHub.", "correction": "Git was created by Linus Torvalds in 2005 for Linux kernel development. GitHub, founded in 2008, is a hosting service for Git repositories."},
    {"initial_wrong": "Python is always slower than C++.", "correction": "While Python is generally slower for CPU-bound tasks, libraries like NumPy use C under the hood. For I/O-bound tasks, the difference is often negligible."},
    {"initial_wrong": "Kubernetes was created by Google in 2020.", "correction": "Kubernetes was open-sourced by Google in 2014, not 2020. It was based on Google's internal Borg system."},
    {"initial_wrong": "REST APIs must use JSON.", "correction": "REST is an architectural style that doesn't mandate any specific data format. While JSON is common, REST APIs can use XML, YAML, or other formats."},
    {"initial_wrong": "Docker containers are virtual machines.", "correction": "Containers are not VMs. Containers share the host OS kernel and are more lightweight. VMs include a full OS and run on a hypervisor."},
    {"initial_wrong": "NoSQL databases don't support ACID transactions.", "correction": "Many modern NoSQL databases do support ACID transactions. MongoDB, for example, has supported multi-document ACID transactions since version 4.0."},
    {"initial_wrong": "Agile means no documentation.", "correction": "Agile values 'working software over comprehensive documentation' but doesn't eliminate documentation. It emphasizes just enough documentation that adds value."},
]

# Confidence level examples - EXPANDED
CONFIDENCE_EXAMPLES = [
    {"question": "What is 2 + 2?", "answer": "4", "confidence": "high", "explanation": "This is a basic mathematical fact."},
    {"question": "What is the capital of France?", "answer": "Paris", "confidence": "high", "explanation": "This is a well-established geographical fact."},
    {"question": "What is the chemical formula for water?", "answer": "H2O", "confidence": "high", "explanation": "This is a fundamental chemistry fact."},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare", "confidence": "high", "explanation": "This is a well-documented historical fact."},
    {"question": "What is the speed of light?", "answer": "Approximately 299,792,458 meters per second", "confidence": "high", "explanation": "This is a precisely measured physical constant."},
    {"question": "What is the best programming language?", "answer": "It depends on the use case. Python is great for data science, JavaScript for web development, and C++ for performance-critical applications.", "confidence": "medium", "explanation": "This is subjective and depends on context."},
    {"question": "Is remote work more productive than office work?", "answer": "Studies show mixed results. It depends on the individual, role, and company culture.", "confidence": "medium", "explanation": "Research shows varying outcomes depending on circumstances."},
    {"question": "What's the best database for my project?", "answer": "It depends on your requirements. PostgreSQL for complex queries, MongoDB for flexible schemas, Redis for caching.", "confidence": "medium", "explanation": "The answer depends on specific project needs."},
    {"question": "Should I use microservices or monolith?", "answer": "For small teams and new projects, a monolith is often better. Microservices add complexity but offer scalability.", "confidence": "medium", "explanation": "This is a trade-off that depends on team size and project scope."},
    {"question": "Is TypeScript better than JavaScript?", "answer": "TypeScript adds type safety which helps catch errors early, but adds compilation overhead. It's beneficial for larger projects.", "confidence": "medium", "explanation": "Both have valid use cases depending on project size and team preferences."},
    {"question": "Will AI surpass human intelligence by 2030?", "answer": "This is uncertain. While AI is advancing rapidly, predicting exact timelines for AGI is speculative.", "confidence": "low", "explanation": "This involves predicting future technological developments."},
    {"question": "Will quantum computers replace classical computers?", "answer": "Unlikely for general computing. Quantum computers excel at specific problems but aren't suited for everyday tasks.", "confidence": "low", "explanation": "Future technology adoption is inherently uncertain."},
    {"question": "What will be the dominant programming language in 2040?", "answer": "Impossible to predict with certainty. Technology trends change rapidly and new languages emerge.", "confidence": "low", "explanation": "Long-term technology predictions are highly speculative."},
    {"question": "Will blockchain technology become mainstream?", "answer": "Some applications may become mainstream, but widespread adoption faces technical and regulatory challenges.", "confidence": "low", "explanation": "Depends on technological and societal factors that are hard to predict."},
    {"question": "Is consciousness computable?", "answer": "This remains an open philosophical and scientific question with no consensus.", "confidence": "low", "explanation": "This is a fundamental question without a definitive answer."},
    {"question": "What causes dark energy?", "answer": "We don't know. Dark energy is one of the biggest mysteries in physics.", "confidence": "low", "explanation": "Current scientific understanding is incomplete."},
    {"question": "Will fusion power be commercially viable by 2050?", "answer": "Progress is being made, but commercial viability timelines remain uncertain.", "confidence": "low", "explanation": "Depends on technological breakthroughs that are hard to predict."},
    {"question": "How many planets have intelligent life?", "answer": "We don't know. We haven't detected any confirmed signs of extraterrestrial intelligence.", "confidence": "low", "explanation": "We lack the data to make this determination."},
    {"question": "What is the best way to learn programming?", "answer": "Practice with real projects, read documentation, and build things you're interested in.", "confidence": "medium", "explanation": "Learning styles vary, but hands-on practice is generally effective."},
    {"question": "How long does it take to become a good programmer?", "answer": "Typically 1-2 years of consistent practice to become proficient, but mastery takes much longer.", "confidence": "medium", "explanation": "This varies greatly by individual and definition of 'good'."},
]


def generate_idk_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a high-quality 'I don't know' training sample with reasoning."""
    qg = QualityGenerator(t)
    question = random.choice(UNKNOWABLE_QUESTIONS)
    response_template = random.choice(IDK_RESPONSES)
    explanation = random.choice(IDK_EXPLANATIONS)
    
    # Vary the response structure
    response_type = random.choice(["simple", "with_thinking", "with_analysis", "with_uncertainty_score"])
    
    if response_type == "simple":
        response = f"{t['unknown']}{response_template}\n\n{explanation}"
    
    elif response_type == "with_thinking":
        thinking = qg.wrap_think(
            qg.wrap_observation(f"The user is asking: {question}") + "\n" +
            qg.wrap_analysis("This question falls into a category I cannot reliably answer.") + "\n" +
            qg.wrap_note(explanation) + "\n" +
            qg.wrap_conclusion("I should acknowledge my limitations honestly.")
        )
        response = f"{thinking}\n{t['unknown']}{response_template}"
    
    elif response_type == "with_analysis":
        response = (
            f"{qg.wrap_analysis('Analyzing the question type...')}\n"
            f"{t['beyond_knowledge']}\n"
            f"{response_template}\n\n"
            f"{qg.wrap_note(explanation)}"
        )
    
    else:  # with_uncertainty_score
        response = (
            f"{qg.add_uncertainty_score(95)}\n"
            f"{t['unknown']}{response_template}\n\n"
            f"{qg.wrap_reflection(explanation)}"
        )
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n{response}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "idk_response"}


def generate_knowledge_cutoff_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a knowledge cutoff awareness sample."""
    question = random.choice(RECENT_EVENT_QUESTIONS)
    
    responses = [
        f"{t['knowledge_cutoff']}I don't have access to real-time information. My knowledge has a cutoff date, so I cannot provide current data on this topic.",
        f"{t['may_be_outdated']}I should note that my information may be outdated. For current information, please check a reliable, up-to-date source.",
        f"{t['beyond_knowledge']}This requires current information that I don't have access to. I recommend checking recent news sources.",
        f"{t['need_verification']}I cannot provide real-time information. Please verify with current sources for the most accurate data.",
    ]
    
    response = random.choice(responses)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n{response}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "knowledge_cutoff"}


def generate_fact_check_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a high-quality fact-checking sample with detailed reasoning."""
    qg = QualityGenerator(t)
    claim, is_true, explanation = random.choice(VERIFICATION_NEEDED_QUESTIONS)
    
    # Build detailed fact-checking response
    thinking_parts = [
        qg.wrap_plan([
            "Analyze the claim carefully",
            "Check against known facts",
            "Identify any misconceptions",
            "Provide accurate information"
        ]),
        qg.wrap_observation(f"Claim to verify: \"{claim}\""),
        qg.wrap_step("Checking this against established knowledge..."),
    ]
    
    if is_true:
        thinking_parts.extend([
            qg.wrap_note("This claim is generally accurate, but may need nuance."),
            qg.wrap_reflection(explanation),
            qg.wrap_critique("The claim is factually correct with some caveats.", has_error=False),
            qg.wrap_conclusion("This statement is TRUE, though context matters.")
        ])
        verdict = "**TRUE** (with nuance)"
        confidence = t['confidence_medium']
        uncertainty = 25
    else:
        thinking_parts.extend([
            qg.wrap_note("This appears to be a common misconception."),
            qg.wrap_reflection(explanation),
            qg.wrap_critique("This is a factual error that should be corrected.", has_error=True),
            qg.wrap_conclusion("This statement is FALSE.")
        ])
        verdict = "**FALSE**"
        confidence = t['confidence_high']
        uncertainty = 10
    
    thinking = qg.wrap_think("\n".join(thinking_parts))
    
    response = (
        f"{thinking}\n\n"
        f"{qg.add_uncertainty_score(uncertainty)}\n"
        f"{confidence}\n"
        f"Verdict: {verdict}\n\n"
        f"{explanation}"
    )
    
    # Vary question phrasing
    question_templates = [
        f'Is this true: "{claim}"',
        f'Fact check: "{claim}"',
        f'Can you verify this claim? "{claim}"',
        f'Is the following statement accurate? "{claim}"',
    ]
    question = random.choice(question_templates)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{question}\n{t['user_end']}"
        f"{t['assistant_start']}\n{response}\n{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "fact_check"}


def generate_grounded_response_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a response grounded in provided context."""
    qa = random.choice(GROUNDED_QA_PAIRS)
    
    text = (
        f"{t['bos']}"
        f"{t['context_start']}\n{qa['context']}\n{t['context_end']}"
        f"{t['user_start']}\n{qa['question']}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['grounded']}{qa['answer']}\n"
        f"{t['cite_start']}{t['source_start']}{qa['source']}{t['source_end']}{t['cite_end']}\n"
        f"{t['confidence_high']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "grounded_response"}


def generate_self_correction_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a self-correction training sample."""
    example = random.choice(SELF_CORRECTION_EXAMPLES)
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\nPlease review and correct if needed: \"{example['initial_wrong']}\"\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{t['self_correct']}\n"
        f"{t['correction_start']}\n{example['correction']}\n{t['correction_end']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "self_correction"}


def generate_confidence_level_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a sample with explicit confidence level."""
    example = random.choice(CONFIDENCE_EXAMPLES)
    
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
    
    return {"text": text, "type": "confidence_level"}


def generate_uncertainty_expression_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate samples that express uncertainty appropriately with uncertainty scores."""
    uncertain_topics = [
        ("What causes consciousness?", "The nature of consciousness remains one of the biggest mysteries in science. While we have theories, there's no scientific consensus.", 75),
        ("Is there life on other planets?", "We don't have definitive evidence yet. While conditions for life may exist elsewhere, we haven't confirmed extraterrestrial life.", 60),
        ("What is dark matter?", "Dark matter is a hypothetical form of matter. We observe its gravitational effects, but its exact nature remains unknown.", 70),
        ("Will quantum computers replace classical computers?", "It's uncertain. Quantum computers excel at specific problems but may not replace classical computers for general tasks.", 65),
        ("What will AI look like in 50 years?", "Predicting technological advancement that far ahead is highly speculative. Current trends suggest continued progress, but specific outcomes are unpredictable.", 85),
        ("Can we achieve true artificial general intelligence?", "This remains an open question in AI research. While progress is being made, there's no consensus on if or when AGI will be achieved.", 70),
    ]
    
    topic, response, uncertainty = random.choice(uncertain_topics)
    
    # Use new uncertainty score token
    uncertainty_marker = f"{t.get('uncertainty_score', '')}{uncertainty}{t.get('uncertainty_score_end', '')}"
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{topic}\n{t['user_end']}"
        f"{t['assistant_start']}\n"
        f"{uncertainty_marker}\n"
        f"{t['uncertain_start']}\n"
        f"{t.get('speculative', '')}{response}\n"
        f"{t['uncertain_end']}\n"
        f"{t['confidence_low']}\n"
        f"{t['assistant_end']}"
        f"{t['eos']}"
    )
    
    return {"text": text, "type": "uncertainty_expression"}


CITATION_DATA = [
    {"claim": "The Earth is approximately 4.54 billion years old.", "source": "U.S. Geological Survey", "quote": "The age of the Earth is estimated to be 4.54 ± 0.05 billion years."},
    {"claim": "Regular physical activity can reduce the risk of heart disease.", "source": "World Health Organization", "quote": "Regular physical activity is proven to help prevent and manage heart disease."},
    {"claim": "Python is one of the most popular programming languages.", "source": "TIOBE Index 2024", "quote": "Python consistently ranks among the top programming languages worldwide."},
    {"claim": "The human genome contains approximately 3 billion base pairs.", "source": "National Human Genome Research Institute", "quote": "The human genome consists of about 3 billion DNA base pairs."},
    {"claim": "Climate change is primarily caused by human activities.", "source": "IPCC Sixth Assessment Report", "quote": "It is unequivocal that human influence has warmed the atmosphere, ocean and land."},
    {"claim": "The speed of light in a vacuum is exactly 299,792,458 meters per second.", "source": "NIST", "quote": "The speed of light in vacuum c is 299,792,458 m/s exactly."},
    {"claim": "Vaccines have eradicated smallpox worldwide.", "source": "World Health Organization", "quote": "Smallpox was declared eradicated in 1980 following a global immunization campaign."},
    {"claim": "The Amazon rainforest produces about 20% of the world's oxygen.", "source": "National Geographic", "quote": "The Amazon produces roughly 20 percent of the world's oxygen."},
    {"claim": "Moore's Law predicts transistor density doubles approximately every two years.", "source": "Intel", "quote": "The number of transistors on a chip doubles about every two years."},
    {"claim": "HTTP/2 can multiplex multiple requests over a single connection.", "source": "RFC 7540", "quote": "HTTP/2 enables a more efficient use of network resources through request multiplexing."},
    {"claim": "Git was created by Linus Torvalds in 2005.", "source": "Git Documentation", "quote": "Git was created by Linus Torvalds in 2005 for development of the Linux kernel."},
    {"claim": "TCP provides reliable, ordered delivery of data.", "source": "RFC 793", "quote": "TCP provides a reliable, in-sequence delivery of data between applications."},
    {"claim": "The first iPhone was released in 2007.", "source": "Apple Press Release", "quote": "Apple reinvents the phone with iPhone, released June 29, 2007."},
    {"claim": "Kubernetes was originally developed by Google.", "source": "CNCF", "quote": "Kubernetes was originally designed by Google and is now maintained by CNCF."},
    {"claim": "The first computer virus appeared in 1986.", "source": "Computer History Museum", "quote": "Brain, the first PC virus, was created in 1986 in Pakistan."},
    {"claim": "Unicode supports over 140,000 characters.", "source": "Unicode Consortium", "quote": "Unicode 15.0 contains 149,186 characters covering 161 modern and historic scripts."},
    {"claim": "The first website was created in 1991.", "source": "CERN", "quote": "The first website was created at CERN by Tim Berners-Lee in 1991."},
    {"claim": "JavaScript was created in just 10 days.", "source": "Brendan Eich", "quote": "I created JavaScript in 10 days in May 1995."},
    {"claim": "Linux powers over 90% of cloud infrastructure.", "source": "Linux Foundation", "quote": "Linux runs on over 90% of public cloud workloads."},
    {"claim": "The first email was sent in 1971.", "source": "Computer History Museum", "quote": "Ray Tomlinson sent the first network email in 1971."},
]

def generate_citation_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a sample with proper citations."""
    cite = random.choice(CITATION_DATA)
    
    # Varied question formats
    question_formats = [
        f"What do we know about {cite['claim'].split()[0].lower()} {' '.join(cite['claim'].split()[1:4])}?",
        f"Can you tell me about {' '.join(cite['claim'].split()[:5]).lower()}?",
        f"Is it true that {cite['claim'].lower().rstrip('.')}?",
        f"What does research say about {' '.join(cite['claim'].split()[1:4]).lower()}?",
        f"Explain: {cite['claim']}",
    ]
    
    text = (
        f"{t['bos']}"
        f"{t['user_start']}\n{random.choice(question_formats)}\n{t['user_end']}"
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
    
    return {"text": text, "type": "citation"}


RAG_DOCUMENTS = [
    {"query": "How does photosynthesis work?", "retrieved": "Photosynthesis is the process by which plants convert light energy into chemical energy. It occurs in chloroplasts and involves two stages: light-dependent reactions and the Calvin cycle.", "answer": "Photosynthesis is a two-stage process occurring in chloroplasts. First, light-dependent reactions capture light energy. Then, the Calvin cycle uses this energy to convert CO2 into glucose."},
    {"query": "What is machine learning?", "retrieved": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.", "answer": "Machine learning is an AI subset where systems learn from data and experience rather than explicit programming. It uses algorithms that improve automatically through exposure to more data."},
    {"query": "How do neural networks work?", "retrieved": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes (neurons) that process information using weighted connections. Training adjusts these weights to minimize prediction errors.", "answer": "Neural networks are layered systems of interconnected nodes inspired by the brain. Information flows through weighted connections, and training adjusts these weights to improve predictions."},
    {"query": "What is Docker?", "retrieved": "Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistent behavior across different environments.", "answer": "Docker is a containerization platform that packages applications with their dependencies. This ensures applications run consistently regardless of the underlying infrastructure."},
    {"query": "How does HTTPS work?", "retrieved": "HTTPS uses TLS (Transport Layer Security) to encrypt communication between a client and server. It involves a handshake process where the server presents a certificate, and both parties agree on encryption keys.", "answer": "HTTPS encrypts web traffic using TLS. During the handshake, the server proves its identity with a certificate, and both parties establish encryption keys for secure communication."},
    {"query": "What is a REST API?", "retrieved": "REST (Representational State Transfer) is an architectural style for designing networked applications. It uses standard HTTP methods (GET, POST, PUT, DELETE) and treats server-side resources as objects that can be created, read, updated, or deleted.", "answer": "REST is an architectural style using HTTP methods to interact with resources. GET retrieves data, POST creates, PUT updates, and DELETE removes resources."},
    {"query": "How does Git branching work?", "retrieved": "Git branches are lightweight pointers to commits. Creating a branch creates a new pointer without copying files. Merging combines changes from different branches, and rebasing replays commits on top of another branch.", "answer": "Git branches are pointers to commits, making them lightweight to create. Merging combines branch changes, while rebasing replays commits onto another branch for a linear history."},
    {"query": "What is Kubernetes?", "retrieved": "Kubernetes is an open-source container orchestration platform. It automates deployment, scaling, and management of containerized applications. Key concepts include pods, services, deployments, and namespaces.", "answer": "Kubernetes orchestrates containerized applications, automating deployment and scaling. It organizes containers into pods, exposes them via services, and manages updates through deployments."},
    {"query": "How does garbage collection work?", "retrieved": "Garbage collection automatically reclaims memory occupied by objects no longer in use. Common algorithms include mark-and-sweep, reference counting, and generational collection. It prevents memory leaks but can cause pause times.", "answer": "Garbage collection automatically frees memory from unused objects. Algorithms like mark-and-sweep identify unreachable objects, while generational collection optimizes by focusing on short-lived objects."},
    {"query": "What is SQL injection?", "retrieved": "SQL injection is a code injection technique that exploits vulnerabilities in applications that construct SQL queries from user input. Attackers can insert malicious SQL code to access, modify, or delete database data.", "answer": "SQL injection exploits applications that build SQL queries from user input. Attackers inject malicious SQL to access or modify data. Prevention includes parameterized queries and input validation."},
    {"query": "How does OAuth work?", "retrieved": "OAuth is an authorization framework that enables third-party applications to access user resources without exposing credentials. It uses tokens instead of passwords and supports different grant types for various use cases.", "answer": "OAuth allows apps to access user resources without passwords by using tokens. The user authorizes access, and the app receives a token to make API requests on their behalf."},
    {"query": "What is a microservice?", "retrieved": "Microservices architecture structures an application as a collection of loosely coupled services. Each service is independently deployable, scalable, and focused on a specific business capability.", "answer": "Microservices break applications into small, independent services. Each handles a specific function, can be deployed separately, and communicates via APIs."},
    {"query": "How does caching work?", "retrieved": "Caching stores frequently accessed data in fast storage to reduce latency and load on primary data sources. Common strategies include LRU (Least Recently Used), TTL (Time To Live), and write-through vs write-back.", "answer": "Caching stores frequently used data in fast storage. LRU evicts least recently used items, TTL expires data after a time, and strategies like write-through ensure consistency."},
    {"query": "What is WebSocket?", "retrieved": "WebSocket is a protocol providing full-duplex communication channels over a single TCP connection. Unlike HTTP, it allows servers to push data to clients without polling, enabling real-time applications.", "answer": "WebSocket enables two-way communication over a single connection. Unlike HTTP's request-response model, servers can push data to clients instantly, ideal for real-time apps."},
    {"query": "How does DNS work?", "retrieved": "DNS (Domain Name System) translates domain names to IP addresses. When you enter a URL, your computer queries DNS servers hierarchically (root, TLD, authoritative) to resolve the domain to an IP.", "answer": "DNS translates domain names to IP addresses. Your browser queries DNS servers hierarchically - root servers, then TLD servers, then authoritative servers - to find the IP."},
    {"query": "What is a hash function?", "retrieved": "A hash function maps data of arbitrary size to fixed-size values. Cryptographic hash functions are one-way, deterministic, and collision-resistant. Common uses include password storage, data integrity, and digital signatures.", "answer": "Hash functions convert data to fixed-size values. Cryptographic hashes are one-way and collision-resistant, used for passwords, integrity checks, and signatures."},
    {"query": "How does load balancing work?", "retrieved": "Load balancing distributes incoming traffic across multiple servers to ensure no single server is overwhelmed. Algorithms include round-robin, least connections, and IP hash. It improves availability and performance.", "answer": "Load balancing distributes traffic across servers. Round-robin cycles through servers, least-connections sends to the least busy, and IP hash ensures session persistence."},
    {"query": "What is a database index?", "retrieved": "A database index is a data structure that improves query speed by providing quick lookup paths to rows. B-tree indexes are common for range queries, while hash indexes excel at equality comparisons.", "answer": "Database indexes speed up queries by creating lookup structures. B-trees handle range queries efficiently, while hash indexes are optimal for exact matches."},
]

def generate_retrieval_grounded_sample(t: Dict[str, str]) -> Dict[str, Any]:
    """Generate a RAG-style retrieval-grounded sample."""
    doc = random.choice(RAG_DOCUMENTS)
    
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
    
    return {"text": text, "type": "retrieval_grounded"}


def generate_dataset(output_dir: str, samples_per_type: int = 2000):
    """Generate all anti-hallucination datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    t = SPECIAL_TOKENS
    
    generators = [
        ("idk_dataset.jsonl", generate_idk_sample, "I Don't Know Responses"),
        ("knowledge_cutoff_dataset.jsonl", generate_knowledge_cutoff_sample, "Knowledge Cutoff Awareness"),
        ("fact_check_dataset.jsonl", generate_fact_check_sample, "Fact Checking"),
        ("grounded_response_dataset.jsonl", generate_grounded_response_sample, "Grounded Responses"),
        ("self_correction_dataset.jsonl", generate_self_correction_sample, "Self Correction"),
        ("confidence_level_dataset.jsonl", generate_confidence_level_sample, "Confidence Levels"),
        ("uncertainty_dataset.jsonl", generate_uncertainty_expression_sample, "Uncertainty Expression"),
        ("citation_dataset.jsonl", generate_citation_sample, "Citations"),
        ("retrieval_grounded_dataset.jsonl", generate_retrieval_grounded_sample, "Retrieval Grounded (RAG)"),
    ]
    
    for filename, generator, desc in generators:
        filepath = os.path.join(output_dir, filename)
        print(f"📝 Generating {desc} dataset ({samples_per_type} samples)...")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for i in range(samples_per_type):
                sample = generator(t)
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"   ✅ Saved to {filepath}")
    
    print(f"\n✅ Generated {len(generators)} anti-hallucination datasets with {samples_per_type} samples each")
    print(f"   Total: {len(generators) * samples_per_type} samples")


# Removed standalone execution - use as module instead
# # if __name__ == "__main__":
#     output_dir = os.path.join(os.path.dirname(__file__), "data")
    # generate_dataset(output_dir, samples_per_type=2000)
