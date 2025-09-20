"""
Prompt Optimization Module

This module implements prompt optimization using reinforcement learning and
feedback data to improve the quality and effectiveness of prompts used by
agents in the Terra Constellata system.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re
import json
from collections import defaultdict
import numpy as np

from .feedback_collector import FeedbackCollector, UserFeedback
from .workflow_tracer import WorkflowTrace
from .pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Represents an optimizable prompt template."""

    def __init__(
        self, template_id: str, base_template: str, variables: List[str], domain: str
    ):
        """
        Initialize a prompt template.

        Args:
            template_id: Unique identifier for the template
            base_template: The base prompt template with placeholders
            variables: List of variable names in the template
            domain: Domain this template is used for
        """
        self.template_id = template_id
        self.base_template = base_template
        self.variables = variables
        self.domain = domain
        self.performance_history: List[Dict[str, Any]] = []
        self.current_parameters: Dict[str, Any] = {}
        self.version = 1

    def render(self, **kwargs) -> str:
        """Render the template with given variables."""
        try:
            return self.base_template.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing variable in template {self.template_id}: {e}")
            return self.base_template

    def add_performance_data(
        self,
        performance_score: float,
        feedback_score: Optional[float] = None,
        usage_context: Optional[Dict[str, Any]] = None,
    ):
        """Add performance data for this template."""
        self.performance_history.append(
            {
                "timestamp": datetime.utcnow(),
                "performance_score": performance_score,
                "feedback_score": feedback_score,
                "usage_context": usage_context or {},
                "version": self.version,
            }
        )

    def get_average_performance(self, last_n: Optional[int] = None) -> float:
        """Get average performance score."""
        history = self.performance_history
        if last_n:
            history = history[-last_n:]

        if not history:
            return 0.5  # Default neutral score

        scores = [
            h["performance_score"]
            for h in history
            if h["performance_score"] is not None
        ]
        return sum(scores) / len(scores) if scores else 0.5

    def optimize_template(self, optimization_data: Dict[str, Any]):
        """Optimize the template based on performance data."""
        # This would implement template optimization logic
        # For now, it's a placeholder for future implementation
        pass


class PromptPerformanceData:
    """Tracks performance data for prompts."""

    def __init__(
        self, prompt_id: str, prompt_text: str, template_id: Optional[str] = None
    ):
        self.prompt_id = prompt_id
        self.prompt_text = prompt_text
        self.template_id = template_id
        self.usage_count = 0
        self.success_count = 0
        self.total_feedback_score = 0.0
        self.feedback_count = 0
        self.avg_response_time = 0.0
        self.creation_timestamp = datetime.utcnow()
        self.last_used = None
        self.performance_trends: List[Dict[str, Any]] = []

    def record_usage(
        self,
        success: bool,
        response_time: Optional[float] = None,
        feedback_score: Optional[float] = None,
    ):
        """Record a usage instance."""
        self.usage_count += 1
        self.last_used = datetime.utcnow()

        if success:
            self.success_count += 1

        if response_time is not None:
            # Update rolling average
            self.avg_response_time = (
                (self.avg_response_time * (self.usage_count - 1)) + response_time
            ) / self.usage_count

        if feedback_score is not None:
            self.feedback_count += 1
            self.total_feedback_score += feedback_score

        # Record trend data
        self.performance_trends.append(
            {
                "timestamp": datetime.utcnow(),
                "success": success,
                "response_time": response_time,
                "feedback_score": feedback_score,
            }
        )

    def get_success_rate(self) -> float:
        """Get success rate."""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

    def get_avg_feedback_score(self) -> Optional[float]:
        """Get average feedback score."""
        return (
            self.total_feedback_score / self.feedback_count
            if self.feedback_count > 0
            else None
        )

    def get_performance_score(self) -> float:
        """Calculate overall performance score."""
        success_weight = 0.4
        feedback_weight = 0.4
        time_weight = 0.2

        success_score = self.get_success_rate()
        feedback_score = self.get_avg_feedback_score() or 0.5

        # Time efficiency score (lower time is better, max expected time = 60 seconds)
        time_score = (
            max(0, 1.0 - (self.avg_response_time / 60.0))
            if self.avg_response_time > 0
            else 0.5
        )

        return (
            success_score * success_weight
            + feedback_score * feedback_weight
            + time_score * time_weight
        )


class PromptOptimizer:
    """Main system for optimizing prompts using learning data."""

    def __init__(
        self, feedback_collector: FeedbackCollector, pattern_analyzer: PatternAnalyzer
    ):
        """
        Initialize the prompt optimizer.

        Args:
            feedback_collector: System for collecting user feedback
            pattern_analyzer: System for analyzing workflow patterns
        """
        self.feedback_collector = feedback_collector
        self.pattern_analyzer = pattern_analyzer

        self.templates: Dict[str, PromptTemplate] = {}
        self.prompt_performance: Dict[str, PromptPerformanceData] = {}
        self.optimization_history: List[Dict[str, Any]] = []

        # Default prompt templates
        self._initialize_default_templates()

        logger.info("Prompt optimizer initialized")

    def _initialize_default_templates(self):
        """Initialize default prompt templates."""
        default_templates = {
            "creative_inspiration": {
                "template": """
You are a creative inspiration specialist. Your task is to generate novel and insightful ideas for: {topic}

Context: {context}
Domain: {domain}
Style: {style}

Please provide {num_ideas} creative ideas that are:
- Innovative and original
- Feasible to implement
- Relevant to the given context
- Well-reasoned and explained

Focus on {focus_area} aspects particularly.
""",
                "variables": [
                    "topic",
                    "context",
                    "domain",
                    "style",
                    "num_ideas",
                    "focus_area",
                ],
                "domain": "creativity",
            },
            "agent_coordination": {
                "template": """
You are coordinating multiple AI agents for the task: {task_description}

Available agents and their capabilities:
{agent_capabilities}

Current system state:
{system_state}

Your goal is to:
1. Analyze the task requirements
2. Select the most appropriate agents
3. Design an efficient coordination strategy
4. Monitor and adjust the coordination as needed

Provide a detailed coordination plan with specific agent assignments and success criteria.
""",
                "variables": ["task_description", "agent_capabilities", "system_state"],
                "domain": "coordination",
            },
            "mythology_analysis": {
                "template": """
Analyze the mythological significance of: {entity}

Cultural context: {cultural_context}
Comparative elements: {comparative_elements}
Analysis depth: {analysis_depth}

Please provide:
1. Historical and cultural significance
2. Symbolic meanings and interpretations
3. Connections to other mythological traditions
4. Modern relevance and applications
5. Key insights and patterns

Focus on {focus_areas} particularly.
""",
                "variables": [
                    "entity",
                    "cultural_context",
                    "comparative_elements",
                    "analysis_depth",
                    "focus_areas",
                ],
                "domain": "mythology",
            },
        }

        for template_id, config in default_templates.items():
            template = PromptTemplate(
                template_id=template_id,
                base_template=config["template"],
                variables=config["variables"],
                domain=config["domain"],
            )
            self.templates[template_id] = template

    def register_template(self, template: PromptTemplate):
        """Register a new prompt template."""
        self.templates[template.template_id] = template
        logger.info(f"Registered prompt template: {template.template_id}")

    def generate_optimized_prompt(self, template_id: str, **kwargs) -> Optional[str]:
        """Generate an optimized prompt using the specified template."""
        if template_id not in self.templates:
            logger.warning(f"Template not found: {template_id}")
            return None

        template = self.templates[template_id]

        # Apply any optimizations to the template
        optimized_template = self._optimize_template_parameters(template, kwargs)

        # Render the prompt
        prompt = optimized_template.render(**kwargs)

        # Track this prompt usage
        prompt_id = f"{template_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.prompt_performance[prompt_id] = PromptPerformanceData(
            prompt_id=prompt_id, prompt_text=prompt, template_id=template_id
        )

        return prompt

    def _optimize_template_parameters(
        self, template: PromptTemplate, context: Dict[str, Any]
    ) -> PromptTemplate:
        """Optimize template parameters based on context and performance history."""
        # Create a copy of the template for optimization
        optimized = PromptTemplate(
            template_id=f"{template.template_id}_optimized",
            base_template=template.base_template,
            variables=template.variables,
            domain=template.domain,
        )

        # Apply optimizations based on performance data
        if template.performance_history:
            # Analyze what parameters work best
            best_performing_contexts = sorted(
                template.performance_history,
                key=lambda x: x["performance_score"],
                reverse=True,
            )[
                :5
            ]  # Top 5 performances

            # Extract common patterns from successful contexts
            optimization_params = self._extract_optimization_parameters(
                best_performing_contexts, context
            )
            optimized.current_parameters = optimization_params

        return optimized

    def _extract_optimization_parameters(
        self, successful_contexts: List[Dict[str, Any]], current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract optimization parameters from successful contexts."""
        # This is a simplified implementation
        # In a real system, this would use more sophisticated analysis

        params = {}

        # Analyze successful parameter values
        for context_data in successful_contexts:
            usage_context = context_data.get("usage_context", {})
            for key, value in usage_context.items():
                if key not in params:
                    params[key] = []
                params[key].append(value)

        # Choose most common or best performing values
        optimized_params = {}
        for param, values in params.items():
            if values:
                # For numeric values, use mean
                if isinstance(values[0], (int, float)):
                    optimized_params[param] = sum(values) / len(values)
                else:
                    # For categorical, use most common
                    from collections import Counter

                    optimized_params[param] = Counter(values).most_common(1)[0][0]

        return optimized_params

    def record_prompt_performance(
        self,
        prompt_id: str,
        success: bool,
        response_time: Optional[float] = None,
        feedback_score: Optional[float] = None,
        workflow_id: Optional[str] = None,
    ):
        """Record performance data for a prompt."""
        if prompt_id not in self.prompt_performance:
            logger.warning(f"Prompt not found: {prompt_id}")
            return

        prompt_data = self.prompt_performance[prompt_id]
        prompt_data.record_usage(success, response_time, feedback_score)

        # Update template performance if applicable
        if prompt_data.template_id and prompt_data.template_id in self.templates:
            template = self.templates[prompt_data.template_id]
            template.add_performance_data(
                performance_score=prompt_data.get_performance_score(),
                feedback_score=feedback_score,
                usage_context={"workflow_id": workflow_id},
            )

        logger.info(f"Recorded performance for prompt {prompt_id}: success={success}")

    def get_best_template(
        self, domain: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Get the best performing template for a domain."""
        domain_templates = [t for t in self.templates.values() if t.domain == domain]

        if not domain_templates:
            return None

        # Score templates based on performance
        template_scores = []
        for template in domain_templates:
            score = template.get_average_performance(last_n=10)  # Last 10 usages
            template_scores.append((template.template_id, score))

        # Return best performing template
        best_template = max(template_scores, key=lambda x: x[1])
        return best_template[0]

    def analyze_prompt_effectiveness(self) -> Dict[str, Any]:
        """Analyze overall prompt effectiveness."""
        if not self.prompt_performance:
            return {"error": "No prompt performance data available"}

        total_prompts = len(self.prompt_performance)
        successful_prompts = sum(
            1 for p in self.prompt_performance.values() if p.get_success_rate() > 0.7
        )

        avg_success_rate = np.mean(
            [p.get_success_rate() for p in self.prompt_performance.values()]
        )
        avg_feedback_score = np.mean(
            [
                p.get_avg_feedback_score() or 0.5
                for p in self.prompt_performance.values()
            ]
        )

        # Template performance
        template_performance = {}
        for template_id, template in self.templates.items():
            template_performance[template_id] = {
                "avg_performance": template.get_average_performance(),
                "usage_count": len(template.performance_history),
                "domain": template.domain,
            }

        return {
            "total_prompts": total_prompts,
            "successful_prompts": successful_prompts,
            "success_rate": successful_prompts / total_prompts
            if total_prompts > 0
            else 0,
            "avg_success_rate": avg_success_rate,
            "avg_feedback_score": avg_feedback_score,
            "template_performance": template_performance,
        }

    def optimize_prompt_from_feedback(self, workflow_id: str) -> Optional[str]:
        """Optimize a prompt based on feedback for a specific workflow."""
        feedback = self.feedback_collector.get_feedback(workflow_id)
        if not feedback:
            return None

        # Find prompts used in this workflow
        workflow_prompts = [
            p
            for p in self.prompt_performance.values()
            if any(
                trend.get("workflow_id") == workflow_id
                for trend in p.performance_trends
            )
        ]

        if not workflow_prompts:
            return None

        # Analyze what worked well and what didn't
        successful_prompts = [p for p in workflow_prompts if p.get_success_rate() > 0.7]
        unsuccessful_prompts = [
            p for p in workflow_prompts if p.get_success_rate() <= 0.7
        ]

        if successful_prompts:
            # Extract patterns from successful prompts
            best_prompt = max(
                successful_prompts, key=lambda p: p.get_performance_score()
            )

            # Generate optimization suggestions
            optimization = self._generate_prompt_optimization(
                best_prompt.prompt_text, feedback
            )
            return optimization

        return None

    def _generate_prompt_optimization(
        self, prompt_text: str, feedback: UserFeedback
    ) -> str:
        """Generate prompt optimization suggestions based on feedback."""
        suggestions = []

        # Analyze feedback for optimization insights
        if feedback.cat_score and feedback.cat_score.score < 7:
            suggestions.append("Increase specificity in task requirements")
            suggestions.append("Add more context about desired output format")

        if feedback.satisfaction_rating and feedback.satisfaction_rating < 0.7:
            suggestions.append("Include clearer success criteria")
            suggestions.append("Add examples of desired output")

        if feedback.usefulness_rating and feedback.usefulness_rating < 0.7:
            suggestions.append("Focus on practical applications")
            suggestions.append("Include implementation guidance")

        if feedback.feedback_text:
            # Simple keyword analysis for suggestions
            feedback_lower = feedback.feedback_text.lower()
            if "unclear" in feedback_lower or "confusing" in feedback_lower:
                suggestions.append("Clarify ambiguous terms and requirements")
            if "too long" in feedback_lower or "verbose" in feedback_lower:
                suggestions.append("Shorten and focus the prompt")
            if "missing" in feedback_lower:
                suggestions.append("Add missing context or requirements")

        return (
            " | ".join(suggestions)
            if suggestions
            else "No specific optimizations identified"
        )

    def export_optimization_data(self, filename: str):
        """Export prompt optimization data for analysis."""
        data = {
            "templates": {
                tid: {
                    "base_template": t.base_template,
                    "variables": t.variables,
                    "domain": t.domain,
                    "performance_history": t.performance_history,
                    "avg_performance": t.get_average_performance(),
                }
                for tid, t in self.templates.items()
            },
            "prompt_performance": {
                pid: {
                    "prompt_text": p.prompt_text[:200] + "..."
                    if len(p.prompt_text) > 200
                    else p.prompt_text,
                    "template_id": p.template_id,
                    "usage_count": p.usage_count,
                    "success_rate": p.get_success_rate(),
                    "avg_feedback_score": p.get_avg_feedback_score(),
                    "performance_score": p.get_performance_score(),
                }
                for pid, p in self.prompt_performance.items()
            },
            "optimization_history": self.optimization_history,
            "export_timestamp": datetime.utcnow().isoformat(),
        }

        import os

        os.makedirs("./optimization_data", exist_ok=True)
        filepath = f"./optimization_data/{filename}.json"

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Optimization data exported to {filepath}")

    def get_optimization_recommendations(self, domain: str) -> Dict[str, Any]:
        """Get optimization recommendations for a domain."""
        domain_templates = {
            tid: t for tid, t in self.templates.items() if t.domain == domain
        }

        if not domain_templates:
            return {"error": f"No templates found for domain: {domain}"}

        # Analyze template performance
        recommendations = {
            "best_template": None,
            "worst_template": None,
            "optimization_suggestions": [],
            "performance_summary": {},
        }

        template_performance = {}
        for tid, template in domain_templates.items():
            perf = template.get_average_performance()
            template_performance[tid] = perf

        if template_performance:
            best_tid = max(template_performance, key=template_performance.get)
            worst_tid = min(template_performance, key=template_performance.get)

            recommendations["best_template"] = {
                "id": best_tid,
                "performance": template_performance[best_tid],
            }
            recommendations["worst_template"] = {
                "id": worst_tid,
                "performance": template_performance[worst_tid],
            }

        recommendations["performance_summary"] = template_performance

        # Generate suggestions
        if recommendations["best_template"] and recommendations["worst_template"]:
            perf_diff = (
                recommendations["best_template"]["performance"]
                - recommendations["worst_template"]["performance"]
            )

            if perf_diff > 0.2:
                recommendations["optimization_suggestions"].append(
                    f"Consider using {recommendations['best_template']['id']} instead of {recommendations['worst_template']['id']}"
                )

        return recommendations
