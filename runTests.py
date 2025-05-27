#!/usr/bin/env python3

"""
Improved test runner - replacement for runtest target in Makefile
Usage: python runTests.py [-j<cores>] <test_dir_or_file> [<test_dir_or_file> ...]
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
from typing import List, Optional
import concurrent.futures


class TestRunner:
    def __init__(self, num_jobs: Optional[int] = None, batch_size: int = 50, debug: bool = False):
        self.num_jobs = num_jobs
        self.batch_size = batch_size
        self.debug = debug
        self.test_suffix = '_test.cpp'
        self.win_suffix = '.exe' if self.is_windows() else ''
        self.stan_mpi = self._detect_stan_mpi()
    
    @staticmethod
    def is_windows() -> bool:
        return platform.system().lower().startswith('windows')
    
    def _detect_stan_mpi(self) -> bool:
        """Check if STAN_MPI is enabled in make/local"""
        try:
            with open('make/local') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#'):
                        continue
                    if 'STAN_MPI=true' in line:
                        return True
        except FileNotFoundError:
            pass
        return False
    
    def _run_command(self, command: str) -> None:
        """Execute a shell command and handle errors"""
        print('-' * 60)
        print(f'Running: {command}')
        
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f'ERROR: Command failed with return code {result.returncode}')
            sys.exit(result.returncode)
    
    def _get_make_target(self, test_path: str) -> str:
        """Convert test file path to make target name"""
        target = str(test_path)
        
        # Remove leading ./
        if target.startswith('./'):
            target = target[2:]
        
        # Handle src/test/unit prefix
        if target.startswith('src/test/unit'):
            target = target.replace('src/', '', 1)
        
        # Remove .cpp extension
        if target.endswith(self.test_suffix):
            target = target[:-len(self.test_suffix)]
        
        # Add Windows executable suffix
        if self.is_windows():
            target += self.win_suffix
            target = target.replace('\\', '/')
        
        # Escape special characters
        target = target.replace(' ', '\\ ').replace('(', '\\(').replace(')', '\\)')
        
        if self.debug:
            print(f'Target: {test_path} -> {target}')
        
        return target
    
    def _make_command(self, targets: List[str]) -> str:
        """Build make command for given targets"""
        targets_str = ' '.join(targets)
        if self.num_jobs:
            return f'make -j{self.num_jobs} {targets_str}'
        return f'make {targets_str}'
    
    def _build_tests_in_batches(self, test_files: List[Path]) -> None:
        """Build test executables in batches"""
        if not test_files:
            return
        
        targets = [self._get_make_target(str(test_file)) for test_file in test_files]
        
        # Process in batches to avoid command line length limits
        for i in range(0, len(targets), self.batch_size):
            batch = targets[i:i + self.batch_size]
            command = self._make_command(batch)
            self._run_command(command)
    
    def _find_test_files(self, path: Path) -> List[Path]:
        """Find all test files in given path"""
        test_files = []
        
        if path.is_file():
            if path.name.endswith(self.test_suffix):
                test_files.append(path)
            else:
                print(f'ERROR: {path} is not a test file (must end with {self.test_suffix})')
                sys.exit(1)
        elif path.is_dir():
            # Use rglob for recursive search (replaces os.walk)
            test_files.extend(path.rglob(f'*{self.test_suffix}'))
        else:
            print(f'ERROR: {path} does not exist')
            sys.exit(1)
        
        return test_files
    
    def _check_mpi_available(self) -> bool:
        """Check if mpirun is available"""
        try:
            subprocess.run(['mpirun', '--version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _run_single_test(self, test_file: Path) -> None:
        """Run a single test executable"""
        target = self._get_make_target(str(test_file))
        executable = target.replace('/', os.sep)
        xml_name = target.replace(self.win_suffix, '')
        
        command = f'{executable} --gtest_output="xml:{xml_name}.xml"'
        
        # Handle MPI tests
        if self.stan_mpi and 'mpi_' in str(test_file):
            if not self._check_mpi_available():
                print('ERROR: MPI tests require mpirun to be installed')
                print('See: https://github.com/stan-dev/stan/wiki/Parallelism-using-MPI-in-Stan')
                sys.exit(1)
            
            num_procs = max(2, self.num_jobs or 2)
            command = f'mpirun -np {num_procs} {command}'
        
        self._run_command(command)
    
    def run(self, test_paths: List[str]) -> None:
        """Main entry point to run tests"""
        # Convert string paths to Path objects
        path_objects = [Path(p) for p in test_paths]
        
        # Find all test files
        all_test_files = []
        for path in path_objects:
            all_test_files.extend(self._find_test_files(path))
        
        if not all_test_files:
            print('No test files found')
            return
        
        print(f'Found {len(all_test_files)} test files')
        
        # Phase 1: Build prerequisites
        print('\n=== Phase 1: Building prerequisites ===')
        build_cmd = f'make -j{self.num_jobs} build' if self.num_jobs else 'make build'
        self._run_command(build_cmd)
        
        if self.is_windows():
            math_cmd = f'make -j{self.num_jobs} -f stan/lib/stan_math/make/standalone math-libs' if self.num_jobs else 'make -f stan/lib/stan_math/make/standalone math-libs'
            self._run_command(math_cmd)
        
        # Phase 2: Build test executables
        print('\n=== Phase 2: Building test executables ===')
        self._build_tests_in_batches(all_test_files)
        
        # Phase 3: Run tests
        print('\n=== Phase 3: Running tests ===')
        for test_file in all_test_files:
            if self.debug:
                print(f'Running test: {test_file}')
            self._run_single_test(test_file)
        
        print(f'\n=== Completed: {len(all_test_files)} tests ===')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run C++ tests with automatic compilation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runTests.py src/test/unit/math
  python runTests.py -j4 src/test/unit/math/functions
  python runTests.py src/test/unit/math/functions/add_test.cpp
  python runTests.py --debug -j2 src/test/unit/math src/test/unit/io
        """
    )
    
    parser.add_argument(
        'paths', 
        nargs='+', 
        help='Test directories or files to run'
    )
    
    parser.add_argument(
        '-j', '--jobs', 
        type=int, 
        metavar='N',
        help='Number of parallel jobs for make (default: auto-detect)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=50,
        help='Number of targets to build in each batch (default: 50)'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug output'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    runner = TestRunner(
        num_jobs=args.jobs,
        batch_size=args.batch_size,
        debug=args.debug
    )
    
    try:
        runner.run(args.paths)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
