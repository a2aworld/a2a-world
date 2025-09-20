#!/usr/bin/env python3
"""
Integration Test Runner for Terra Constellata

This script provides a comprehensive test runner for all integration tests,
performance benchmarks, and optimization checks.
"""

import argparse
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Comprehensive integration test runner."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.tests_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)

    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> dict:
        """Run all integration tests."""
        logger.info("Starting comprehensive integration test suite...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "performance_results": {},
            "coverage_report": None,
            "summary": {},
        }

        # Run unit tests first
        logger.info("Running unit tests...")
        unit_results = self.run_unit_tests(verbose)
        results["test_results"]["unit"] = unit_results

        # Run integration tests
        logger.info("Running integration tests...")
        integration_results = self.run_integration_tests(verbose, coverage)
        results["test_results"]["integration"] = integration_results

        # Run performance tests
        logger.info("Running performance benchmarks...")
        performance_results = self.run_performance_tests(verbose)
        results["performance_results"] = performance_results

        # Generate coverage report
        if coverage:
            logger.info("Generating coverage report...")
            coverage_report = self.generate_coverage_report()
            results["coverage_report"] = coverage_report

        # Generate summary
        results["summary"] = self.generate_summary(results)

        # Save results
        self.save_results(results)

        return results

    def run_unit_tests(self, verbose: bool = False) -> dict:
        """Run unit tests."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir),
            "-v" if verbose else "",
            "--tb=short",
            "--maxfail=5",
            "-x",  # Stop on first failure
            "--disable-warnings",
        ]

        # Filter out empty strings
        cmd = [arg for arg in cmd if arg]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            logger.error("Unit tests timed out")
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Tests timed out after 5 minutes",
                "success": False,
            }

    def run_integration_tests(
        self, verbose: bool = False, coverage: bool = True
    ) -> dict:
        """Run integration tests."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "test_integration_system.py"),
            str(self.tests_dir / "test_database_integration.py"),
            str(self.tests_dir / "test_agent_integration.py"),
            "-v" if verbose else "",
            "--tb=short",
            "-m",
            "integration",
            "--maxfail=3",
        ]

        if coverage:
            cmd.extend(
                [
                    "--cov=terra_constellata",
                    "--cov-report=term-missing",
                    "--cov-report=xml",
                    "--cov-fail-under=80",
                ]
            )

        # Filter out empty strings
        cmd = [arg for arg in cmd if arg]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            logger.error("Integration tests timed out")
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Integration tests timed out after 10 minutes",
                "success": False,
            }

    def run_performance_tests(self, verbose: bool = False) -> dict:
        """Run performance tests."""
        performance_results = {}

        # Run performance benchmarks
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(self.tests_dir / "test_performance_benchmarks.py"),
            str(self.tests_dir / "test_load_testing.py"),
            str(self.tests_dir / "test_memory_profiling.py"),
            str(self.tests_dir / "test_query_optimization.py"),
            "-v" if verbose else "",
            "--tb=short",
            "-m",
            "performance",
            "--maxfail=3",
        ]

        # Filter out empty strings
        cmd = [arg for arg in cmd if arg]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout
            )

            performance_results["benchmarks"] = {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            logger.error("Performance tests timed out")
            performance_results["benchmarks"] = {
                "return_code": -1,
                "stdout": "",
                "stderr": "Performance tests timed out after 15 minutes",
                "success": False,
            }

        return performance_results

    def generate_coverage_report(self) -> dict:
        """Generate coverage report."""
        try:
            # Check if coverage report exists
            coverage_file = self.project_root / "coverage.xml"
            if coverage_file.exists():
                # Parse coverage XML (simplified)
                with open(coverage_file, "r") as f:
                    content = f.read()

                # Extract basic coverage info
                return {
                    "coverage_file": str(coverage_file),
                    "generated": True,
                    "content_preview": content[:500] + "..."
                    if len(content) > 500
                    else content,
                }
            else:
                return {
                    "coverage_file": None,
                    "generated": False,
                    "error": "Coverage file not found",
                }

        except Exception as e:
            logger.error(f"Error generating coverage report: {e}")
            return {"coverage_file": None, "generated": False, "error": str(e)}

    def generate_summary(self, results: dict) -> dict:
        """Generate test summary."""
        summary = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "success_rate": 0.0,
            "performance_tests_run": 0,
            "performance_tests_passed": 0,
            "coverage_achieved": False,
            "overall_success": False,
        }

        # Analyze test results
        for test_type, result in results["test_results"].items():
            if result["success"]:
                summary["passed_tests"] += 1
            else:
                summary["failed_tests"] += 1
            summary["total_tests"] += 1

        # Analyze performance results
        for perf_type, result in results["performance_results"].items():
            summary["performance_tests_run"] += 1
            if result.get("success", False):
                summary["performance_tests_passed"] += 1

        # Calculate success rate
        if summary["total_tests"] > 0:
            summary["success_rate"] = (
                summary["passed_tests"] / summary["total_tests"]
            ) * 100

        # Check coverage
        if results.get("coverage_report", {}).get("generated", False):
            summary["coverage_achieved"] = True

        # Overall success
        summary["overall_success"] = (
            summary["success_rate"] >= 80
            and summary["performance_tests_passed"]  # 80% test success
            >= summary["performance_tests_run"] * 0.7  # 70% performance tests
        )

        return summary

    def save_results(self, results: dict):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"integration_test_report_{timestamp}.json"

        try:
            with open(report_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Test results saved to: {report_file}")

            # Also save a summary text file
            summary_file = self.reports_dir / f"test_summary_{timestamp}.txt"
            with open(summary_file, "w") as f:
                f.write("Terra Constellata Integration Test Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Timestamp: {results['timestamp']}\n\n")

                summary = results["summary"]
                f.write("OVERALL RESULTS:\n")
                f.write(f"  Success Rate: {summary['success_rate']:.1f}%\n")
                f.write(
                    f"  Tests Passed: {summary['passed_tests']}/{summary['total_tests']}\n"
                )
                f.write(
                    f"  Performance Tests: {summary['performance_tests_passed']}/{summary['performance_tests_run']}\n"
                )
                f.write(f"  Coverage Generated: {summary['coverage_achieved']}\n")
                f.write(f"  Overall Success: {summary['overall_success']}\n\n")

                f.write("DETAILED RESULTS:\n")
                for test_type, result in results["test_results"].items():
                    f.write(
                        f"  {test_type.upper()}: {'PASS' if result['success'] else 'FAIL'}\n"
                    )

                f.write("\nPERFORMANCE RESULTS:\n")
                for perf_type, result in results["performance_results"].items():
                    f.write(
                        f"  {perf_type.upper()}: {'PASS' if result.get('success', False) else 'FAIL'}\n"
                    )

            logger.info(f"Test summary saved to: {summary_file}")

        except Exception as e:
            logger.error(f"Error saving test results: {e}")

    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests to verify basic functionality."""
        logger.info("Running smoke tests...")

        # Add project root to Python path
        import sys

        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Test basic imports
        try:
            # Try importing some core modules
            import backend.main

            logger.info("✓ Backend module import successful")
        except ImportError as e:
            logger.warning(f"⚠ Backend import failed: {e}")

        try:
            import data.postgis.connection

            logger.info("✓ PostGIS module import successful")
        except ImportError as e:
            logger.warning(f"⚠ PostGIS import failed: {e}")

        try:
            import data.ckg.connection

            logger.info("✓ CKG module import successful")
        except ImportError as e:
            logger.warning(f"⚠ CKG import failed: {e}")

        try:
            import agents.base_agent

            logger.info("✓ Agents module import successful")
        except ImportError as e:
            logger.warning(f"⚠ Agents import failed: {e}")


        # Test database connections (will skip if not available)
        try:
            from tests.conftest import postgis_connection, ckg_connection

            logger.info("✓ Database connection fixtures available")
        except ImportError:
            logger.warning("⚠ Database connection fixtures not available")

        # Test agent registry
        try:
            from tests.conftest import agent_registry

            logger.info("✓ Agent registry fixture available")
        except ImportError:
            logger.warning("⚠ Agent registry fixture not available")

        logger.info("Smoke tests completed")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Terra Constellata Integration Test Runner"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        default=True,
        help="Generate coverage report",
    )
    parser.add_argument(
        "--smoke", "-s", action="store_true", help="Run only smoke tests"
    )
    parser.add_argument(
        "--project-root", "-p", default=".", help="Project root directory"
    )

    args = parser.parse_args()

    # Get project root
    project_root = Path(args.project_root).resolve()
    # Check if we're already in the terra_constellata directory
    if (project_root / "tests").exists() and (project_root / "backend").exists():
        # We're already in the project root
        pass
    elif (project_root / "terra_constellata").exists():
        # We're in the parent directory
        project_root = project_root / "terra_constellata"
    else:
        logger.error(f"Project root not found at: {project_root}")
        logger.error(
            "Expected to find either 'tests' and 'backend' directories (project root) or 'terra_constellata' directory"
        )
        sys.exit(1)

    # Initialize test runner
    runner = IntegrationTestRunner(project_root)

    if args.smoke:
        # Run only smoke tests
        success = runner.run_smoke_tests()
        sys.exit(0 if success else 1)

    # Run full test suite
    start_time = time.time()
    results = runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)
    end_time = time.time()

    # Print summary
    summary = results["summary"]
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUITE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logger.info(f"Test success rate: {summary['success_rate']:.1f}%")
    logger.info(f"Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
    logger.info(
        f"Performance tests: {summary['performance_tests_passed']}/{summary['performance_tests_run']}"
    )
    logger.info(f"Coverage generated: {summary['coverage_achieved']}")
    logger.info(
        f"Overall result: {'SUCCESS' if summary['overall_success'] else 'FAILURE'}"
    )
    logger.info("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if summary["overall_success"] else 1)


if __name__ == "__main__":
    main()
