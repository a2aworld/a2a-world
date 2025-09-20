"""
Unit Tests for Tool Shed Components

Comprehensive tests for all Tool Shed functionality.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from .models import (
    Tool,
    ToolMetadata,
    ToolCapabilities,
    ToolProposal,
    SearchQuery,
    ToolValidation,
)
from .registry import ToolRegistry
from .vector_store import ToolVectorStore
from .search import SemanticSearchEngine
from .evolution import ToolEvolutionManager, VersionManager
from .tool_smith_agent import ToolSmithAgent, SecurityScanner, CodeLinter


class TestModels:
    """Test Pydantic models."""

    def test_tool_creation(self):
        """Test creating a tool with valid data."""
        tool = Tool(
            metadata=ToolMetadata(
                name="TestTool",
                description="A test tool",
                author="TestAgent",
                category="test",
                tags=["test", "demo"],
            ),
            capabilities=ToolCapabilities(
                functions=["test_function"],
                input_types=["text"],
                output_types=["result"],
            ),
            code="def test(): return 'test'",
            documentation="Test tool documentation",
        )

        assert tool.metadata.name == "TestTool"
        assert tool.capabilities.functions == ["test_function"]
        assert tool.is_active is True

    def test_tool_proposal_creation(self):
        """Test creating a tool proposal."""
        proposal = ToolProposal(
            proposer_agent="TestAgent",
            tool_name="NewTool",
            description="A new tool proposal",
            capabilities=["analyze", "process"],
            use_case="Data processing",
            priority="high",
        )

        assert proposal.tool_name == "NewTool"
        assert proposal.status == "pending"
        assert proposal.priority == "high"


class TestVersionManager:
    """Test version management functionality."""

    def test_version_comparison(self):
        """Test version comparison."""
        vm = VersionManager()

        assert vm.compare_versions("1.0.0", "1.0.1") == -1
        assert vm.compare_versions("1.1.0", "1.0.0") == 1
        assert vm.compare_versions("1.0.0", "1.0.0") == 0

    def test_breaking_change_detection(self):
        """Test breaking change detection."""
        vm = VersionManager()

        assert vm.is_breaking_change("1.0.0", "2.0.0") is True
        assert vm.is_breaking_change("1.0.0", "1.1.0") is False
        assert vm.is_breaking_change("1.0.0", "1.0.1") is False

    def test_version_generation(self):
        """Test automatic version generation."""
        vm = VersionManager()

        assert vm.generate_next_version("1.0.0", "major") == "2.0.0"
        assert vm.generate_next_version("1.0.0", "minor") == "1.1.0"
        assert vm.generate_next_version("1.0.0", "patch") == "1.0.1"


class TestSecurityScanner:
    """Test security scanning functionality."""

    def test_dangerous_pattern_detection(self):
        """Test detection of dangerous code patterns."""
        scanner = SecurityScanner()

        # Test dangerous patterns
        dangerous_code = """
exec("malicious_code")
eval("dangerous")
os.system("rm -rf /")
subprocess.call(["dangerous", "command"])
"""

        passed, issues = scanner.scan_code(dangerous_code)
        assert passed is False
        assert len(issues) > 0
        assert any("exec" in issue for issue in issues)

    def test_safe_code(self):
        """Test scanning of safe code."""
        scanner = SecurityScanner()

        safe_code = """
def safe_function(x):
    return x * 2

result = safe_function(5)
"""

        passed, issues = scanner.scan_code(safe_code)
        assert passed is True
        assert len(issues) == 0


class TestCodeLinter:
    """Test code linting functionality."""

    def test_syntax_checking(self):
        """Test syntax validation."""
        linter = CodeLinter()

        # Valid syntax
        valid_code = """
def test_function():
    x = 1
    return x + 1
"""

        passed, issues = linter.lint_code(valid_code)
        assert passed is True

        # Invalid syntax
        invalid_code = """
def broken_function(
    return "broken"
"""

        passed, issues = linter.lint_code(invalid_code)
        assert passed is False
        assert any("Syntax error" in issue for issue in issues)

    def test_style_checking(self):
        """Test code style validation."""
        linter = CodeLinter()

        bad_style_code = """
def badStyleFunction(x,y,z):
    x=1+2+3+4+5+6+7+8+9+10  # Very long line
    return x
"""

        passed, issues = linter.lint_code(bad_style_code)
        assert passed is False
        assert any("Line too long" in issue for issue in issues)


class TestToolRegistry:
    """Test tool registry functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def registry(self, temp_dir):
        """Create test registry."""
        vector_store = ToolVectorStore(persist_directory=str(temp_dir / "db"))
        registry = ToolRegistry(vector_store=vector_store)
        yield registry

        # Cleanup
        try:
            vector_store.client.delete_collection("tool_shed")
        except:
            pass

    @pytest.mark.asyncio
    async def test_tool_registration(self, registry):
        """Test tool registration."""
        tool = Tool(
            metadata=ToolMetadata(
                name="TestTool",
                description="Test tool",
                author="TestAgent",
                category="test",
            ),
            capabilities=ToolCapabilities(
                functions=["test"], input_types=["text"], output_types=["result"]
            ),
            code="def test(): return 'test'",
            documentation="Test documentation",
        )

        success = await registry.register_tool(tool)
        assert success is True

        # Verify tool is registered
        retrieved = registry.get_tool(tool.id)
        assert retrieved is not None
        assert retrieved.metadata.name == "TestTool"

    @pytest.mark.asyncio
    async def test_tool_search(self, registry):
        """Test tool search functionality."""
        # Register test tools
        tool1 = Tool(
            metadata=ToolMetadata(
                name="DataTool",
                description="Data processing tool",
                author="DataAgent",
                category="data",
            ),
            capabilities=ToolCapabilities(
                functions=["process_data"],
                input_types=["csv"],
                output_types=["processed_data"],
            ),
            code="def process_data(data): return data",
            documentation="Processes data",
        )

        tool2 = Tool(
            metadata=ToolMetadata(
                name="TextTool",
                description="Text analysis tool",
                author="TextAgent",
                category="nlp",
            ),
            capabilities=ToolCapabilities(
                functions=["analyze_text"],
                input_types=["text"],
                output_types=["analysis"],
            ),
            code="def analyze_text(text): return text",
            documentation="Analyzes text",
        )

        await registry.register_tool(tool1)
        await registry.register_tool(tool2)

        # Test listing
        all_tools = registry.list_tools()
        assert len(all_tools) == 2

        # Test category filtering
        data_tools = registry.list_tools(category="data")
        assert len(data_tools) == 1
        assert data_tools[0].metadata.name == "DataTool"

    @pytest.mark.asyncio
    async def test_proposal_workflow(self, registry):
        """Test tool proposal workflow."""
        proposal = ToolProposal(
            proposer_agent="TestAgent",
            tool_name="ProposedTool",
            description="A proposed tool",
            capabilities=["analyze"],
            use_case="Analysis tasks",
        )

        # Submit proposal
        proposal_id = await registry.submit_proposal(proposal)
        assert proposal_id == proposal.id

        # Approve proposal
        success = await registry.approve_proposal(proposal_id, "ToolSmith")
        assert success is True

        # Check proposal status
        # Note: In real implementation, this would be stored differently
        assert proposal.status == "approved"


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def search_engine(self, temp_dir):
        """Create test search engine."""
        vector_store = ToolVectorStore(persist_directory=str(temp_dir / "db"))
        registry = ToolRegistry(vector_store=vector_store)
        search_engine = SemanticSearchEngine(registry, vector_store)
        yield search_engine

        # Cleanup
        try:
            vector_store.client.delete_collection("tool_shed")
        except:
            pass

    @pytest.mark.asyncio
    async def test_basic_search(self, search_engine):
        """Test basic semantic search."""
        # Register a test tool
        tool = Tool(
            metadata=ToolMetadata(
                name="DataAnalyzer",
                description="Analyzes data and generates insights",
                author="DataAgent",
                category="data_science",
            ),
            capabilities=ToolCapabilities(
                functions=["analyze", "visualize"],
                input_types=["csv", "json"],
                output_types=["charts", "reports"],
            ),
            code="def analyze(data): return data",
            documentation="Data analysis tool",
        )

        await search_engine.registry.register_tool(tool)

        # Perform search
        query = SearchQuery(query="data analysis", limit=5)
        results = await search_engine.search(query)

        assert results.total_count >= 1
        assert len(results.tools) >= 1


class TestToolEvolution:
    """Test tool evolution functionality."""

    @pytest.fixture
    async def evolution_manager(self):
        """Create test evolution manager."""
        registry = ToolRegistry()
        manager = ToolEvolutionManager(registry)
        yield manager

    @pytest.mark.asyncio
    async def test_evolution_request_creation(self, evolution_manager):
        """Test creating evolution requests."""
        request_id = await evolution_manager.create_evolution_request(
            tool_id="test_tool_id",
            evolution_type="enhancement",
            description="Add new feature",
            proposed_changes={"functions": ["new_feature"]},
            requester_agent="TestAgent",
        )

        assert request_id is not None

        # Check request was created
        requests = evolution_manager.registry.get_evolution_requests()
        assert len(requests) == 1
        assert requests[0].evolution_type == "enhancement"

    def test_version_compatibility(self, evolution_manager):
        """Test backward compatibility checking."""
        old_code = """
def old_function(x):
    return x * 2
"""

        new_code = """
def old_function(x):
    return x * 2

def new_function(y):
    return y + 1
"""

        (
            is_compatible,
            issues,
        ) = evolution_manager.compatibility_checker.check_compatibility(
            old_code, new_code
        )

        assert is_compatible is True
        assert len(issues) == 0


class TestToolSmithAgent:
    """Test ToolSmith agent functionality."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = Mock()
        llm.arun = AsyncMock(return_value="Mock response")
        return llm

    @pytest.fixture
    async def tool_smith(self, mock_llm):
        """Create test ToolSmith agent."""
        registry = ToolRegistry()
        agent = ToolSmithAgent(
            llm=mock_llm, registry=registry, a2a_server_url="http://localhost:8080"
        )
        yield agent
        agent.cleanup()

    @pytest.mark.asyncio
    async def test_validation_workflow(self, tool_smith):
        """Test tool validation workflow."""
        # This would normally validate a real proposal
        # For testing, we test the validation components

        # Test security scanner
        scanner = tool_smith.security_scanner
        passed, issues = scanner.scan_code("def safe(): return 1")
        assert passed is True

        # Test code linter
        linter = tool_smith.code_linter
        passed, issues = linter.lint_code("def test():\n    return 1")
        assert passed is True


# Integration test
@pytest.mark.asyncio
async def test_full_tool_lifecycle():
    """Test complete tool lifecycle from proposal to registration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        vector_store = ToolVectorStore(persist_directory=tmpdir)
        registry = ToolRegistry(vector_store=vector_store)
        search_engine = SemanticSearchEngine(registry, vector_store)
        evolution_manager = ToolEvolutionManager(registry)

        # Create and register tool
        tool = Tool(
            metadata=ToolMetadata(
                name="LifecycleTestTool",
                description="Tool for testing lifecycle",
                author="TestAgent",
                category="test",
            ),
            capabilities=ToolCapabilities(
                functions=["test"], input_types=["text"], output_types=["result"]
            ),
            code="def test(): return 'success'",
            documentation="Lifecycle test tool",
        )

        success = await registry.register_tool(tool)
        assert success is True

        # Search for tool
        query = SearchQuery(query="lifecycle test", limit=5)
        results = await search_engine.search(query)
        assert results.total_count >= 1

        # Create evolution request
        evolution_id = await evolution_manager.create_evolution_request(
            tool_id=tool.id,
            evolution_type="enhancement",
            description="Add more features",
            proposed_changes={"functions": ["test", "enhanced_test"]},
            requester_agent="TestAgent",
        )

        # Approve evolution
        success = await evolution_manager.approve_evolution_request(
            evolution_id, "ToolSmith"
        )
        assert success is True

        # Verify evolution
        versions = evolution_manager.get_evolution_history(tool.id)
        assert len(versions) >= 2  # Original + evolved

        # Cleanup
        vector_store.client.delete_collection("tool_shed")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])
