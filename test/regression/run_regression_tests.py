#!/usr/bin/env python3
import os
import sys
import unittest
import argparse
import importlib.util


def discover_tests(test_dir):
  """
  Discover all test modules in the given directory.
  Returns a list of module names.
  """
  test_modules = []

  # Get all Python files in the test directory
  for filename in os.listdir(test_dir):
    if (
      filename.startswith("test_")
      and filename.endswith(".py")
    ):
      module_name = filename[:-3]  # Remove .py extension
      test_modules.append(module_name)

  return test_modules


def load_test_module(module_name, test_dir):
  """
  Load a test module by name.
  Returns the module object.
  """
  module_path = os.path.join(test_dir, module_name + ".py")
  spec = importlib.util.spec_from_file_location(module_name, module_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def create_test_suite(test_modules, test_dir, executable_path=None):
  """
  Create a test suite from the given test modules.

  Args:
      test_modules: List of test module names
      test_dir: Directory containing test modules
      executable_path: Optional path to an existing executable to use instead of building
  """
  suite = unittest.TestSuite()

  for module_name in test_modules:
    module = load_test_module(module_name, test_dir)

    # Check if the module has a create_test_suite function
    if hasattr(module, "create_test_suite"):
      module_suite = module.create_test_suite(executable_path=executable_path)
      suite.addTest(module_suite)
    else:
      # If not, try to load all tests from the module
      module_suite = unittest.defaultTestLoader.loadTestsFromModule(module)
      suite.addTest(module_suite)

  return suite


def main():
  parser = argparse.ArgumentParser(description="Run Athelas regression tests")
  parser.add_argument(
    "--test", "-t", help="Run a specific test (e.g., test_sod)"
  )
  parser.add_argument(
    "--upgold", "-u", action="store_true", help="Update gold files"
  )
  parser.add_argument(
    "--list", "-l", action="store_true", help="List available tests"
  )
  parser.add_argument(
    "--executable",
    "-e",
    help="Path to an existing executable to use instead of building",
  )
  args = parser.parse_args()

  # Get the directory of this script
  test_dir = os.path.dirname(os.path.abspath(__file__))

  # Discover all test modules
  test_modules = discover_tests(test_dir)

  if args.list:
    print("Available tests:")
    for module in test_modules:
      print(f"  - {module}")
    return os.EX_OK

  if args.test:
    # Run a specific test
    if args.test not in test_modules:
      print(f"Error: Test '{args.test}' not found.")
      print("Available tests:")
      for module in test_modules:
        print(f"  - {module}")
      return os.EX_SOFTWARE

    # Load and run the specified test
    module = load_test_module(args.test, test_dir)

    # If the module has a create_test_suite function, use it
    if hasattr(module, "create_test_suite"):
      suite = module.create_test_suite(executable_path=args.executable)
    else:
      # Otherwise, load all tests from the module
      suite = unittest.defaultTestLoader.loadTestsFromModule(module)

    # Set upgold flag if requested
    if args.upgold:
      for test in suite:
        if hasattr(test, "upgold"):
          test.upgold = True

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return os.EX_OK if result.wasSuccessful() else os.EX_SOFTWARE

  # Run all tests
  suite = create_test_suite(
    test_modules, test_dir, executable_path=args.executable
  )

  # Set upgold flag if requested
  if args.upgold:
    for test in suite:
      if hasattr(test, "upgold"):
        test.upgold = True

  runner = unittest.TextTestRunner(verbosity=2)
  result = runner.run(suite)
  return os.EX_OK if result.wasSuccessful() else os.EX_SOFTWARE


if __name__ == "__main__":
  sys.exit(main())
