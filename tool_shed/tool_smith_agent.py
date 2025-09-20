"""
ToolSmith Agent - Gatekeeper for Tool Validation

The ToolSmithAgent is responsible for validating, testing, and approving
new tools before they can be registered in the Tool Shed.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
import ast
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import importlib.util

from langchain.llms.base import BaseLLM
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory

from ..agents.base_agent import BaseSpecialistAgent
from ..a2a_protocol.client import A2AClient
from ..a2a_protocol.schemas import ToolProposal
from .models import Tool, ToolValidation, ToolMetadata, ToolCapabilities
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Security scanner for tool code analysis."""

    def __init__(self):
        self.dangerous_patterns = [
            r"exec\(",
            r"eval\(",
            r"__import__\(",
            r"open\(",
            r"subprocess\.",
            r"os\.system",
            r"os\.popen",
            r"importlib\.import_module",
            r"import\s+os",
            r"import\s+subprocess",
            r"import\s+sys",
            r"from\s+os\s+import",
            r"from\s+subprocess\s+import",
        ]

    def scan_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Scan code for security issues.

        Args:
            code: Code to scan

        Returns:
            Tuple of (passed, issues)
        """
        issues = []

        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, code)
            if matches:
                issues.extend(
                    [f"Potentially dangerous pattern found: {pattern}" for _ in matches]
                )

        # Check for network access
        if re.search(r"(socket|requests|urllib|http)", code):
            issues.append("Code contains network access capabilities")

        # Check for file system access
        if re.search(r"(Path|open|shutil|glob)", code):
            issues.append("Code contains file system access capabilities")

        passed = len(issues) == 0
        return passed, issues


class CodeLinter:
    """Code linter for style and quality checks."""

    def __init__(self):
        self.python_keywords = {
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        }

    def lint_code(self, code: str) -> Tuple[bool, List[str]]:
        """
        Lint code for style and quality issues.

        Args:
            code: Code to lint

        Returns:
            Tuple of (passed, issues)
        """
        issues = []

        try:
            # Parse AST to check syntax
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            return False, issues

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # PEP 8 recommendation
                issues.append(f"Line {i}: Line too long ({len(line)} > 88 characters)")

            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append(f"Line {i}: Trailing whitespace")

            # Check for multiple statements on one line
            if line.count(";") > 1:
                issues.append(f"Line {i}: Multiple statements on one line")

            # Check for proper indentation (basic check)
            stripped = line.lstrip()
            if stripped and not line.startswith(" " * (len(line) - len(stripped))):
                if not (stripped.startswith("#") or len(stripped) == 0):
                    issues.append(f"Line {i}: Inconsistent indentation")

        # Check for proper docstrings
        if "def " in code or "class " in code:
            if '"""' not in code and "'''" not in code:
                issues.append("Missing docstring for function/class")

        # Check for unused imports (basic check)
        import_lines = [
            line
            for line in lines
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        if len(import_lines) > 10:
            issues.append("Too many imports - consider organizing into modules")

        passed = len(issues) == 0
        return passed, issues


class UnitTestRunner:
    """Unit test runner for tool validation."""

    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def run_tests(
        self, code: str, test_code: Optional[str] = None
    ) -> Tuple[bool, List[str], float]:
        """
        Run unit tests on the code.

        Args:
            code: Main code to test
            test_code: Optional test code

        Returns:
            Tuple of (passed, issues, coverage)
        """
        issues = []
        coverage = 0.0

        try:
            # Create temporary files
            main_file = self.temp_dir / "tool_main.py"
            test_file = self.temp_dir / "test_tool.py"

            with open(main_file, "w") as f:
                f.write(code)

            if test_code:
                with open(test_file, "w") as f:
                    f.write(test_code)
            else:
                # Generate basic tests
                test_content = self._generate_basic_tests(code)
                with open(test_file, "w") as f:
                    f.write(test_content)

            # Run tests using subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_file), "-v", "--tb=short"],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                passed = True
                # Try to extract coverage if pytest-cov is available
                try:
                    cov_result = subprocess.run(
                        [
                            "python",
                            "-m",
                            "pytest",
                            str(test_file),
                            "--cov=tool_main",
                            "--cov-report=term-missing",
                        ],
                        cwd=self.temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if cov_result.returncode == 0:
                        # Extract coverage percentage (basic parsing)
                        output = cov_result.stdout + cov_result.stderr
                        cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
                        if cov_match:
                            coverage = float(cov_match.group(1)) / 100.0
                except:
                    pass  # Coverage not available
            else:
                passed = False
                issues.append(f"Tests failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            passed = False
            issues.append("Tests timed out")
        except Exception as e:
            passed = False
            issues.append(f"Test execution error: {e}")

        return passed, issues, coverage

    def _generate_basic_tests(self, code: str) -> str:
        """Generate basic unit tests for the code."""
        # This is a simplified test generation
        # In a real implementation, you'd use more sophisticated analysis

        test_template = '''
import pytest
from tool_main import *

def test_basic_functionality():
    """Test basic functionality of the tool."""
    # This is a placeholder test - real tests would be more specific
    assert True  # Placeholder assertion

def test_error_handling():
    """Test error handling."""
    # Test with invalid inputs
    pass

def test_edge_cases():
    """Test edge cases."""
    # Test boundary conditions
    pass
'''

        return test_template

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


class ToolSmithAgent(BaseSpecialistAgent):
    """
    ToolSmith Agent - Gatekeeper for tool validation and approval.

    Responsible for:
    - Security scanning of tool code
    - Code linting and quality checks
    - Unit testing and validation
    - Final approval/rejection of tool proposals
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        a2a_server_url: str = "http://localhost:8080",
        **kwargs,
    ):
        """
        Initialize the ToolSmith agent.

        Args:
            llm: Language model for the agent
            registry: Tool registry instance
            a2a_server_url: A2A protocol server URL
            **kwargs: Additional parameters
        """
        tools = [
            self._create_validate_tool(),
            self._create_approve_tool(),
            self._create_reject_tool(),
            self._create_review_proposals_tool(),
        ]

        super().__init__(
            name="ToolSmith",
            llm=llm,
            tools=tools,
            a2a_server_url=a2a_server_url,
            **kwargs,
        )

        self.registry = registry
        self.security_scanner = SecurityScanner()
        self.code_linter = CodeLinter()
        self.test_runner = UnitTestRunner()

        logger.info("Initialized ToolSmith Agent")

    def _create_validate_tool(self) -> BaseTool:
        """Create tool for validating tool proposals."""
        from langchain.tools import tool

        @tool
        def validate_tool_proposal(proposal_id: str) -> str:
            """
            Validate a tool proposal through security scanning, linting, and testing.

            Args:
                proposal_id: ID of the proposal to validate

            Returns:
                Validation results
            """
            return asyncio.run(self._validate_proposal(proposal_id))

        return validate_tool_proposal

    def _create_approve_tool(self) -> BaseTool:
        """Create tool for approving proposals."""
        from langchain.tools import tool

        @tool
        def approve_tool_proposal(proposal_id: str, comments: str = "") -> str:
            """
            Approve a tool proposal after validation.

            Args:
                proposal_id: ID of the proposal to approve
                comments: Optional approval comments

            Returns:
                Approval result
            """
            return asyncio.run(self._approve_proposal(proposal_id, comments))

        return approve_tool_proposal

    def _create_reject_tool(self) -> BaseTool:
        """Create tool for rejecting proposals."""
        from langchain.tools import tool

        @tool
        def reject_tool_proposal(proposal_id: str, reason: str) -> str:
            """
            Reject a tool proposal with reason.

            Args:
                proposal_id: ID of the proposal to reject
                reason: Reason for rejection

            Returns:
                Rejection result
            """
            return asyncio.run(self._reject_proposal(proposal_id, reason))

        return reject_tool_proposal

    def _create_review_proposals_tool(self) -> BaseTool:
        """Create tool for reviewing pending proposals."""
        from langchain.tools import tool

        @tool
        def review_pending_proposals() -> str:
            """
            Review all pending tool proposals.

            Returns:
                Summary of pending proposals
            """
            return asyncio.run(self._review_pending_proposals())

        return review_pending_proposals

    async def _validate_proposal(self, proposal_id: str) -> str:
        """
        Validate a tool proposal.

        Args:
            proposal_id: ID of the proposal to validate

        Returns:
            Validation results
        """
        # Get proposal from registry (assuming it's accessible)
        # For now, we'll simulate getting the proposal
        # In real implementation, this would query the registry

        # Simulate proposal data
        proposal_code = """
def example_tool(x, y):
    return x + y
"""

        # Security scan
        security_passed, security_issues = self.security_scanner.scan_code(
            proposal_code
        )

        # Code linting
        lint_passed, lint_issues = self.code_linter.lint_code(proposal_code)

        # Unit testing
        test_passed, test_issues, coverage = self.test_runner.run_tests(proposal_code)

        # Create validation results
        validation = ToolValidation(
            security_scan_passed=security_passed,
            linting_passed=lint_passed,
            unit_tests_passed=test_passed,
            security_issues=security_issues,
            linting_issues=lint_issues,
            test_coverage=coverage,
            validated_at=datetime.utcnow(),
            validated_by=self.name,
        )

        result = f"""
Validation Results for Proposal {proposal_id}:

Security Scan: {'PASSED' if security_passed else 'FAILED'}
- Issues: {', '.join(security_issues) if security_issues else 'None'}

Code Linting: {'PASSED' if lint_passed else 'FAILED'}
- Issues: {', '.join(lint_issues) if lint_issues else 'None'}

Unit Tests: {'PASSED' if test_passed else 'FAILED'}
- Coverage: {coverage:.1%}
- Issues: {', '.join(test_issues) if test_issues else 'None'}

Overall Status: {'APPROVED' if all([security_passed, lint_passed, test_passed]) else 'REQUIRES REVIEW'}
"""

        return result

    async def _approve_proposal(self, proposal_id: str, comments: str) -> str:
        """
        Approve a tool proposal.

        Args:
            proposal_id: ID of the proposal to approve
            comments: Approval comments

        Returns:
            Approval result
        """
        try:
            success = await self.registry.approve_proposal(proposal_id, self.name)
            if success:
                return f"Proposal {proposal_id} approved successfully. {comments}"
            else:
                return f"Failed to approve proposal {proposal_id}"
        except Exception as e:
            return f"Error approving proposal {proposal_id}: {e}"

    async def _reject_proposal(self, proposal_id: str, reason: str) -> str:
        """
        Reject a tool proposal.

        Args:
            proposal_id: ID of the proposal to reject
            reason: Reason for rejection

        Returns:
            Rejection result
        """
        try:
            success = await self.registry.reject_proposal(
                proposal_id, self.name, reason
            )
            if success:
                return f"Proposal {proposal_id} rejected. Reason: {reason}"
            else:
                return f"Failed to reject proposal {proposal_id}"
        except Exception as e:
            return f"Error rejecting proposal {proposal_id}: {e}"

    async def _review_pending_proposals(self) -> str:
        """
        Review all pending proposals.

        Returns:
            Summary of pending proposals
        """
        try:
            pending = self.registry.get_pending_proposals()

            if not pending:
                return "No pending proposals to review."

            summary = f"Found {len(pending)} pending proposals:\n\n"

            for proposal in pending:
                summary += f"""
Proposal ID: {proposal.id}
Tool Name: {proposal.tool_name}
Proposer: {proposal.proposer_agent}
Priority: {proposal.priority}
Submitted: {proposal.submitted_at}
Description: {proposal.description[:100]}...
---
"""

            return summary

        except Exception as e:
            return f"Error reviewing proposals: {e}"

    async def _create_agent_executor(self):
        """Create the LangChain agent executor."""
        from langchain.agents import initialize_agent, AgentType

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
        )

    async def process_task(self, task: str, **kwargs) -> Any:
        """
        Process a task autonomously.

        Args:
            task: Task description
            **kwargs: Additional parameters

        Returns:
            Task result
        """
        try:
            # Use the agent to process the task
            result = await self.agent_executor.arun(task)
            self.update_memory(task, str(result))
            return result
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return f"Error: {e}"

    async def _autonomous_loop(self):
        """Main autonomous operation loop."""
        while self.is_active:
            try:
                # Check for pending proposals
                pending = self.registry.get_pending_proposals()

                if pending:
                    for proposal in pending:
                        # Validate each pending proposal
                        validation_result = await self._validate_proposal(proposal.id)

                        # Auto-approve if all checks pass
                        if "Overall Status: APPROVED" in validation_result:
                            await self._approve_proposal(
                                proposal.id, "Auto-approved by ToolSmith"
                            )
                        else:
                            # Send notification for manual review
                            await self._notify_manual_review(
                                proposal.id, validation_result
                            )

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(30)

    async def _notify_manual_review(self, proposal_id: str, validation_result: str):
        """Notify for manual review of a proposal."""
        # This would send a message via A2A protocol to human operators
        # For now, just log it
        logger.info(
            f"Proposal {proposal_id} requires manual review:\n{validation_result}"
        )

    def cleanup(self):
        """Clean up resources."""
        self.test_runner.cleanup()
