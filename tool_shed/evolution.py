"""
Tool Evolution System

Manages tool versioning, updates, deprecation, and evolution requests.
Supports backward compatibility checking and migration paths.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from packaging import version
import re

from .models import (
    Tool,
    ToolVersion,
    ToolEvolutionRequest,
    ToolRegistryEntry,
    ToolMetadata,
    ToolCapabilities,
)
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages tool versioning and compatibility.
    """

    @staticmethod
    def parse_version(version_str: str) -> version.Version:
        """Parse version string into comparable object."""
        try:
            return version.Version(version_str)
        except version.InvalidVersion:
            # Handle non-standard versions
            return version.Version("0.0.0")

    @staticmethod
    def is_breaking_change(old_version: str, new_version: str) -> bool:
        """
        Determine if version change is breaking.

        Args:
            old_version: Current version
            new_version: New version

        Returns:
            True if breaking change
        """
        old_ver = VersionManager.parse_version(old_version)
        new_ver = VersionManager.parse_version(new_version)

        # Major version change is always breaking
        if new_ver.major > old_ver.major:
            return True

        # Minor version changes can be breaking if they introduce incompatibilities
        # This is a simplified check - real implementation would analyze the changes
        return False

    @staticmethod
    def generate_next_version(current_version: str, change_type: str) -> str:
        """
        Generate next version based on change type.

        Args:
            current_version: Current version string
            change_type: Type of change ("major", "minor", "patch")

        Returns:
            Next version string
        """
        try:
            ver = version.Version(current_version)

            if change_type == "major":
                return f"{ver.major + 1}.0.0"
            elif change_type == "minor":
                return f"{ver.major}.{ver.minor + 1}.0"
            elif change_type == "patch":
                return f"{ver.major}.{ver.minor}.{ver.micro + 1}"
            else:
                return f"{ver.major}.{ver.minor}.{ver.micro + 1}"

        except:
            return "1.0.0"

    @staticmethod
    def compare_versions(version1: str, version2: str) -> int:
        """
        Compare two versions.

        Returns:
            -1 if version1 < version2
             0 if version1 == version2
             1 if version1 > version2
        """
        v1 = VersionManager.parse_version(version1)
        v2 = VersionManager.parse_version(version2)

        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0


class CompatibilityChecker:
    """
    Checks backward compatibility between tool versions.
    """

    def __init__(self):
        self.compatibility_patterns = {
            "function_removed": re.compile(r"def\s+(\w+)\s*\("),
            "parameter_changed": re.compile(r"def\s+\w+\s*\((.*?)\)"),
            "class_removed": re.compile(r"class\s+(\w+)"),
            "attribute_removed": re.compile(r"self\.(\w+)\s*="),
        }

    def check_compatibility(
        self, old_code: str, new_code: str
    ) -> Tuple[bool, List[str]]:
        """
        Check backward compatibility between code versions.

        Args:
            old_code: Original code
            new_code: New code

        Returns:
            Tuple of (is_compatible, issues)
        """
        issues = []

        # Extract functions from old code
        old_functions = set(
            self.compatibility_patterns["function_removed"].findall(old_code)
        )

        # Extract functions from new code
        new_functions = set(
            self.compatibility_patterns["function_removed"].findall(new_code)
        )

        # Check for removed functions
        removed_functions = old_functions - new_functions
        if removed_functions:
            issues.append(f"Removed functions: {', '.join(removed_functions)}")

        # Extract classes from old code
        old_classes = set(
            self.compatibility_patterns["class_removed"].findall(old_code)
        )

        # Extract classes from new code
        new_classes = set(
            self.compatibility_patterns["class_removed"].findall(new_code)
        )

        # Check for removed classes
        removed_classes = old_classes - new_classes
        if removed_classes:
            issues.append(f"Removed classes: {', '.join(removed_classes)}")

        # Check for parameter changes (simplified)
        # This would require more sophisticated AST analysis

        is_compatible = len(issues) == 0
        return is_compatible, issues


class ToolEvolutionManager:
    """
    Manages the evolution of tools through versioning and updates.
    """

    def __init__(self, registry: ToolRegistry):
        """
        Initialize the evolution manager.

        Args:
            registry: Tool registry instance
        """
        self.registry = registry
        self.version_manager = VersionManager()
        self.compatibility_checker = CompatibilityChecker()

    async def create_evolution_request(
        self,
        tool_id: str,
        evolution_type: str,
        description: str,
        proposed_changes: Dict[str, Any],
        requester_agent: str,
        priority: str = "medium",
    ) -> str:
        """
        Create a new evolution request for a tool.

        Args:
            tool_id: ID of tool to evolve
            evolution_type: Type of evolution
            description: Description of changes
            proposed_changes: Proposed changes
            requester_agent: Agent making the request
            priority: Priority level

        Returns:
            Evolution request ID
        """
        request = ToolEvolutionRequest(
            tool_id=tool_id,
            evolution_type=evolution_type,
            description=description,
            proposed_changes=proposed_changes,
            requester_agent=requester_agent,
            priority=priority,
        )

        request_id = await self.registry.submit_evolution_request(request)
        logger.info(f"Created evolution request {request_id} for tool {tool_id}")
        return request_id

    async def approve_evolution_request(
        self, request_id: str, reviewer_agent: str, new_version: Optional[str] = None
    ) -> bool:
        """
        Approve an evolution request and apply changes.

        Args:
            request_id: ID of evolution request
            reviewer_agent: Agent approving the request
            new_version: New version (auto-generated if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the evolution request
            requests = self.registry.get_evolution_requests()
            request = next((r for r in requests if r.id == request_id), None)

            if not request:
                logger.error(f"Evolution request {request_id} not found")
                return False

            # Get current tool
            current_tool = self.registry.get_tool(request.tool_id)
            if not current_tool:
                logger.error(f"Tool {request.tool_id} not found")
                return False

            # Generate new version if not provided
            if not new_version:
                change_type = self._determine_change_type(request.evolution_type)
                new_version = self.version_manager.generate_next_version(
                    current_tool.metadata.version, change_type
                )

            # Check for breaking changes
            is_breaking = self.version_manager.is_breaking_change(
                current_tool.metadata.version, new_version
            )

            # Apply changes to create new tool version
            updated_tool = self._apply_evolution_changes(
                current_tool, request, new_version
            )

            # Update tool in registry
            success = await self.registry.update_tool(
                request.tool_id, updated_tool, reviewer_agent
            )

            if success:
                # Mark request as approved
                request.status = "approved"
                logger.info(
                    f"Evolution request {request_id} approved, tool updated to {new_version}"
                )
            else:
                logger.error(f"Failed to update tool {request.tool_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to approve evolution request {request_id}: {e}")
            return False

    async def reject_evolution_request(
        self, request_id: str, reviewer_agent: str, reason: str
    ) -> bool:
        """
        Reject an evolution request.

        Args:
            request_id: ID of evolution request
            reviewer_agent: Agent rejecting the request
            reason: Reason for rejection

        Returns:
            True if successful, False otherwise
        """
        try:
            requests = self.registry.get_evolution_requests()
            request = next((r for r in requests if r.id == request_id), None)

            if not request:
                logger.error(f"Evolution request {request_id} not found")
                return False

            request.status = "rejected"
            logger.info(f"Evolution request {request_id} rejected: {reason}")
            return True

        except Exception as e:
            logger.error(f"Failed to reject evolution request {request_id}: {e}")
            return False

    def _determine_change_type(self, evolution_type: str) -> str:
        """
        Determine the type of version change based on evolution type.

        Args:
            evolution_type: Type of evolution

        Returns:
            Version change type ("major", "minor", "patch")
        """
        breaking_types = ["deprecation", "breaking_change"]
        feature_types = ["enhancement", "new_feature"]

        if evolution_type in breaking_types:
            return "major"
        elif evolution_type in feature_types:
            return "minor"
        else:
            return "patch"

    def _apply_evolution_changes(
        self, current_tool: Tool, request: ToolEvolutionRequest, new_version: str
    ) -> Tool:
        """
        Apply evolution changes to create updated tool.

        Args:
            current_tool: Current tool version
            request: Evolution request with changes
            new_version: New version string

        Returns:
            Updated tool
        """
        # Create a copy of the current tool
        updated_tool = current_tool.copy()

        # Update metadata
        updated_tool.metadata.version = new_version
        updated_tool.metadata.updated_at = datetime.utcnow()

        # Apply proposed changes
        changes = request.proposed_changes

        if "code" in changes:
            updated_tool.code = changes["code"]

        if "description" in changes:
            updated_tool.metadata.description = changes["description"]

        if "capabilities" in changes:
            # Update capabilities
            cap_changes = changes["capabilities"]
            if "functions" in cap_changes:
                updated_tool.capabilities.functions = cap_changes["functions"]
            if "parameters" in cap_changes:
                updated_tool.capabilities.parameters = cap_changes["parameters"]

        if "documentation" in changes:
            updated_tool.documentation = changes["documentation"]

        # Check backward compatibility
        if current_tool.code and updated_tool.code:
            (
                is_compatible,
                compatibility_issues,
            ) = self.compatibility_checker.check_compatibility(
                current_tool.code, updated_tool.code
            )

            if not is_compatible:
                # Add deprecation warnings
                updated_tool.metadata.tags.append("backward-incompatible")
                # This would be stored in the version info

        return updated_tool

    def get_evolution_history(self, tool_id: str) -> List[ToolVersion]:
        """
        Get evolution history for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            List of version changes
        """
        return self.registry.get_tool_versions(tool_id)

    def suggest_evolution(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Suggest potential evolutions for a tool based on usage patterns.

        Args:
            tool_id: ID of the tool

        Returns:
            List of evolution suggestions
        """
        tool = self.registry.get_tool(tool_id)
        if not tool:
            return []

        suggestions = []

        # Analyze usage patterns
        if tool.usage_count > 100:
            suggestions.append(
                {
                    "type": "optimization",
                    "description": "High usage suggests optimization opportunities",
                    "priority": "medium",
                }
            )

        # Check for outdated dependencies
        # This would require dependency analysis

        # Check for missing features based on reviews
        if tool.reviews:
            # Analyze reviews for common feature requests
            feature_requests = []
            for review in tool.reviews:
                if "feature" in review.get("type", "").lower():
                    feature_requests.append(review.get("content", ""))

            if feature_requests:
                suggestions.append(
                    {
                        "type": "enhancement",
                        "description": f"Common feature requests: {', '.join(feature_requests[:3])}",
                        "priority": "medium",
                    }
                )

        # Check for security improvements
        if tool.capabilities.security_level == "low":
            suggestions.append(
                {
                    "type": "security",
                    "description": "Consider improving security level",
                    "priority": "high",
                }
            )

        return suggestions

    def create_migration_path(
        self, tool_id: str, from_version: str, to_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a migration path between versions.

        Args:
            tool_id: ID of the tool
            from_version: Source version
            to_version: Target version

        Returns:
            Migration path information or None if not possible
        """
        versions = self.get_evolution_history(tool_id)

        # Find the relevant versions
        from_ver_info = None
        to_ver_info = None

        for ver in versions:
            if ver.version == from_version:
                from_ver_info = ver
            if ver.version == to_version:
                to_ver_info = ver

        if not from_ver_info or not to_ver_info:
            return None

        # Check if migration is possible
        is_breaking = self.version_manager.is_breaking_change(from_version, to_version)

        migration_path = {
            "tool_id": tool_id,
            "from_version": from_version,
            "to_version": to_version,
            "is_breaking_change": is_breaking,
            "migration_steps": [],
            "compatibility_issues": [],
        }

        if is_breaking:
            migration_path["migration_steps"].append(
                {
                    "step": "review_breaking_changes",
                    "description": "Review breaking changes in the changelog",
                }
            )
            migration_path["migration_steps"].append(
                {
                    "step": "update_code",
                    "description": "Update code to handle API changes",
                }
            )

        migration_path["migration_steps"].append(
            {
                "step": "test_integration",
                "description": "Test integration with updated tool",
            }
        )

        return migration_path

    def get_deprecation_schedule(self, tool_id: str) -> List[Dict[str, Any]]:
        """
        Get deprecation schedule for a tool.

        Args:
            tool_id: ID of the tool

        Returns:
            List of deprecation events
        """
        versions = self.get_evolution_history(tool_id)

        deprecation_events = []

        for ver in versions:
            if ver.deprecation_warnings:
                deprecation_events.append(
                    {
                        "version": ver.version,
                        "warnings": ver.deprecation_warnings,
                        "created_at": ver.created_at,
                    }
                )

        return deprecation_events
