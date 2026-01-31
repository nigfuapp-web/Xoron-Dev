#!/usr/bin/env python3
"""
Xoron-Dev Interactive Test Runner

A comprehensive test runner for the Xoron-Dev multimodal model project.
Supports both interactive mode and automated testing with --all flag.

Usage:
    python test.py          # Interactive mode
    python test.py --all    # Run all tests automatically
    python test.py -v       # Verbose output
    python test.py --module config  # Run specific module tests
"""

import sys
import os
import unittest
import argparse
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class TestModule:
    """Represents a test module."""
    name: str
    path: str
    description: str
    submodules: List['TestModule'] = None
    
    def __post_init__(self):
        if self.submodules is None:
            self.submodules = []


# Define test modules structure
TEST_MODULES = {
    'config': TestModule(
        name='config',
        path='tests.config',
        description='Configuration module tests',
        submodules=[
            TestModule('model_config', 'tests.config.test_model_config', 'XoronConfig tests'),
            TestModule('training_config', 'tests.config.test_training_config', 'TrainingConfig tests'),
            TestModule('special_tokens', 'tests.config.test_special_tokens', 'Special tokens tests'),
            TestModule('chat_template', 'tests.config.test_chat_template', 'Chat template tests'),
            TestModule('dataset_config', 'tests.config.test_dataset_config', 'Dataset config tests'),
        ]
    ),
    'models': TestModule(
        name='models',
        path='tests.models',
        description='Model architecture tests',
        submodules=[
            TestModule('components.lora', 'tests.models.components.test_lora', 'LoRA implementation tests'),
            TestModule('components.moe', 'tests.models.components.test_moe', 'MoE implementation tests'),
            TestModule('components.attention', 'tests.models.components.test_attention', 'Attention mechanism tests'),
            TestModule('components.projectors', 'tests.models.components.test_projectors', 'Projector tests'),
            TestModule('encoders.vision', 'tests.models.encoders.test_vision', 'Vision encoder tests'),
            TestModule('encoders.audio', 'tests.models.encoders.test_audio', 'Audio encoder tests'),
            TestModule('generators.image', 'tests.models.generators.test_image', 'Image generator tests'),
            TestModule('generators.video', 'tests.models.generators.test_video', 'Video generator tests'),
            TestModule('llm.moe_llama', 'tests.models.llm.test_moe_llama', 'MoE LLaMA model tests'),
        ]
    ),
    'data': TestModule(
        name='data',
        path='tests.data',
        description='Data processing tests',
        submodules=[
            TestModule('dataset', 'tests.data.test_dataset', 'Dataset class tests'),
            TestModule('formatters', 'tests.data.test_formatters', 'Data formatter tests'),
            TestModule('processors', 'tests.data.test_processors', 'Data processor tests'),
        ]
    ),
    'training': TestModule(
        name='training',
        path='tests.training',
        description='Training module tests',
        submodules=[
            TestModule('trainer', 'tests.training.test_trainer', 'XoronTrainer tests'),
            TestModule('utils', 'tests.training.test_utils', 'Training utilities tests'),
        ]
    ),
    'synth': TestModule(
        name='synth',
        path='tests.synth',
        description='Synthetic data generation tests',
        submodules=[
            TestModule('generator', 'tests.synth.test_generator', 'Generator tests'),
            TestModule('templates', 'tests.synth.test_templates', 'Template tests'),
            TestModule('quality_utils', 'tests.synth.test_quality_utils', 'Quality utils tests'),
        ]
    ),
    'utils': TestModule(
        name='utils',
        path='tests.utils',
        description='Utility module tests',
        submodules=[
            TestModule('device', 'tests.utils.test_device', 'Device utilities tests'),
            TestModule('logging', 'tests.utils.test_logging', 'Logging utilities tests'),
        ]
    ),
    'export': TestModule(
        name='export',
        path='tests.export',
        description='Export functionality tests',
        submodules=[
            TestModule('onnx_export', 'tests.export.test_onnx_export', 'ONNX export tests'),
        ]
    ),
}


@dataclass
class TestResult:
    """Container for test results."""
    module: str
    tests_run: int
    failures: int
    errors: int
    skipped: int
    duration: float
    success: bool
    
    @property
    def passed(self) -> int:
        return self.tests_run - self.failures - self.errors - self.skipped


def print_header():
    """Print the test runner header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}   XORON-DEV TEST SUITE{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}   Comprehensive Unit Testing for Multimodal Model{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")


def print_menu():
    """Print the interactive menu."""
    print(f"\n{Colors.BOLD}Available Test Modules:{Colors.ENDC}\n")
    
    for i, (key, module) in enumerate(TEST_MODULES.items(), 1):
        print(f"  {Colors.CYAN}{i}.{Colors.ENDC} {Colors.BOLD}{module.name}{Colors.ENDC}")
        print(f"     {Colors.YELLOW}{module.description}{Colors.ENDC}")
        if module.submodules:
            for sub in module.submodules:
                print(f"       • {sub.name}: {sub.description}")
        print()
    
    print(f"  {Colors.CYAN}A.{Colors.ENDC} {Colors.BOLD}Run ALL tests{Colors.ENDC}")
    print(f"  {Colors.CYAN}Q.{Colors.ENDC} {Colors.BOLD}Quit{Colors.ENDC}")
    print()


def run_test_module(module_path: str, verbosity: int = 2) -> TestResult:
    """Run tests from a specific module."""
    start_time = time.time()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        # Try to load the module
        tests = loader.loadTestsFromName(module_path)
        suite.addTests(tests)
    except Exception as e:
        print(f"{Colors.RED}Error loading {module_path}: {e}{Colors.ENDC}")
        return TestResult(
            module=module_path,
            tests_run=0,
            failures=0,
            errors=1,
            skipped=0,
            duration=time.time() - start_time,
            success=False,
        )
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)
    
    duration = time.time() - start_time
    
    return TestResult(
        module=module_path,
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        duration=duration,
        success=result.wasSuccessful(),
    )


def run_module_tests(module: TestModule, verbosity: int = 2) -> List[TestResult]:
    """Run all tests for a module and its submodules."""
    results = []
    
    if module.submodules:
        for sub in module.submodules:
            print(f"\n{Colors.BOLD}{Colors.BLUE}Running: {sub.path}{Colors.ENDC}")
            print(f"{Colors.YELLOW}{sub.description}{Colors.ENDC}")
            print("-" * 50)
            
            result = run_test_module(sub.path, verbosity)
            results.append(result)
    else:
        print(f"\n{Colors.BOLD}{Colors.BLUE}Running: {module.path}{Colors.ENDC}")
        result = run_test_module(module.path, verbosity)
        results.append(result)
    
    return results


def run_all_tests(verbosity: int = 2) -> List[TestResult]:
    """Run all tests in all modules."""
    all_results = []
    
    for module in TEST_MODULES.values():
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}  Testing Module: {module.name.upper()}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        
        results = run_module_tests(module, verbosity)
        all_results.extend(results)
    
    return all_results


def print_summary(results: List[TestResult]):
    """Print a summary of test results."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}   TEST SUMMARY{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")
    
    total_tests = sum(r.tests_run for r in results)
    total_passed = sum(r.passed for r in results)
    total_failures = sum(r.failures for r in results)
    total_errors = sum(r.errors for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration for r in results)
    
    # Print per-module results
    print(f"{'Module':<45} {'Tests':<8} {'Pass':<8} {'Fail':<8} {'Skip':<8} {'Time':<10}")
    print("-" * 87)
    
    for result in results:
        status_color = Colors.GREEN if result.success else Colors.RED
        module_name = result.module.replace('tests.', '')
        print(f"{module_name:<45} {result.tests_run:<8} {status_color}{result.passed:<8}{Colors.ENDC} "
              f"{Colors.RED if result.failures else ''}{result.failures:<8}{Colors.ENDC if result.failures else ''} "
              f"{result.skipped:<8} {result.duration:.2f}s")
    
    print("-" * 87)
    
    # Print totals
    overall_success = total_failures == 0 and total_errors == 0
    status_color = Colors.GREEN if overall_success else Colors.RED
    status_text = "PASSED" if overall_success else "FAILED"
    
    print(f"{'TOTAL':<45} {total_tests:<8} {Colors.GREEN}{total_passed:<8}{Colors.ENDC} "
          f"{Colors.RED if total_failures else ''}{total_failures:<8}{Colors.ENDC if total_failures else ''} "
          f"{total_skipped:<8} {total_duration:.2f}s")
    
    print(f"\n{Colors.BOLD}Overall Status: {status_color}{status_text}{Colors.ENDC}")
    
    if total_failures > 0 or total_errors > 0:
        print(f"\n{Colors.RED}⚠️  {total_failures} failures, {total_errors} errors{Colors.ENDC}")
    else:
        print(f"\n{Colors.GREEN}✅ All tests passed!{Colors.ENDC}")
    
    return overall_success


def interactive_mode(verbosity: int = 2):
    """Run the interactive test menu."""
    while True:
        print_header()
        print_menu()
        
        choice = input(f"{Colors.BOLD}Enter your choice: {Colors.ENDC}").strip().upper()
        
        if choice == 'Q':
            print(f"\n{Colors.CYAN}Goodbye!{Colors.ENDC}\n")
            break
        elif choice == 'A':
            results = run_all_tests(verbosity)
            print_summary(results)
            input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
        elif choice.isdigit():
            idx = int(choice) - 1
            modules = list(TEST_MODULES.values())
            if 0 <= idx < len(modules):
                module = modules[idx]
                results = run_module_tests(module, verbosity)
                print_summary(results)
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid choice. Please try again.{Colors.ENDC}")
        else:
            # Check if it's a module name
            if choice.lower() in TEST_MODULES:
                module = TEST_MODULES[choice.lower()]
                results = run_module_tests(module, verbosity)
                print_summary(results)
                input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid choice. Please try again.{Colors.ENDC}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Xoron-Dev Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py              # Interactive mode
  python test.py --all        # Run all tests
  python test.py --module config  # Run config module tests
  python test.py -v --all     # Verbose output for all tests
        """
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all tests automatically (non-interactive)'
    )
    
    parser.add_argument(
        '--module', '-m',
        type=str,
        choices=list(TEST_MODULES.keys()),
        help='Run tests for a specific module'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output (minimal)'
    )
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    print_header()
    
    if args.all:
        # Run all tests automatically
        print(f"{Colors.BOLD}Running all tests...{Colors.ENDC}\n")
        results = run_all_tests(verbosity)
        success = print_summary(results)
        sys.exit(0 if success else 1)
        
    elif args.module:
        # Run specific module tests
        module = TEST_MODULES[args.module]
        print(f"{Colors.BOLD}Running {module.name} tests...{Colors.ENDC}\n")
        results = run_module_tests(module, verbosity)
        success = print_summary(results)
        sys.exit(0 if success else 1)
        
    else:
        # Interactive mode
        interactive_mode(verbosity)


if __name__ == '__main__':
    main()
