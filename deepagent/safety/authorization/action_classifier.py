"""
Action Classification System

Classifies tool actions by their potential IMPACT, not text content.

This is Foundation #1: Action-Level Safety.
We evaluate what the action DOES, not what it SAYS.
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class ActionRiskLevel(Enum):
    """
    Risk levels for tool actions

    Based on potential impact and reversibility
    """
    SAFE = 0          # Read operations, queries (reversible, no side effects)
    LOW = 1           # Non-destructive writes (reversible, minimal impact)
    MEDIUM = 2        # Data modifications (reversible with effort)
    HIGH = 3          # Code execution, API calls (hard to reverse)
    CRITICAL = 4      # System modifications, deployments (irreversible)


class ActionCategory(Enum):
    """Categories of actions"""
    READ = "read"                    # Read data (safe)
    SEARCH = "search"                # Search/query (safe)
    ANALYZE = "analyze"              # Analysis/computation (safe)
    WRITE = "write"                  # Write data
    MODIFY = "modify"                # Modify existing data
    EXECUTE = "execute"              # Execute code/commands
    DEPLOY = "deploy"                # Deploy/publish
    DELETE = "delete"                # Delete data
    NETWORK = "network"              # Network requests
    SYSTEM = "system"                # System operations


@dataclass
class ActionMetadata:
    """Metadata about a tool action"""
    tool_name: str
    category: ActionCategory
    risk_level: ActionRiskLevel
    description: str
    requires_approval: bool
    reversible: bool
    side_effects: List[str]
    allowed_parameters: Optional[Dict] = None


class ActionClassifier:
    """
    Classifies actions by their impact and risk level

    This system evaluates what actions DO, not what they SAY.
    """

    def __init__(self):
        """Initialize action classifier"""

        # Tool registry with risk classifications
        self.tool_classifications: Dict[str, ActionMetadata] = {
            # SAFE actions (read-only, no side effects)
            "search_pubmed": ActionMetadata(
                tool_name="search_pubmed",
                category=ActionCategory.SEARCH,
                risk_level=ActionRiskLevel.SAFE,
                description="Search PubMed for scientific papers",
                requires_approval=False,
                reversible=True,
                side_effects=[]
            ),
            "search_drugbank": ActionMetadata(
                tool_name="search_drugbank",
                category=ActionCategory.SEARCH,
                risk_level=ActionRiskLevel.SAFE,
                description="Search DrugBank database",
                requires_approval=False,
                reversible=True,
                side_effects=[]
            ),
            "analyze_binding_sites": ActionMetadata(
                tool_name="analyze_binding_sites",
                category=ActionCategory.ANALYZE,
                risk_level=ActionRiskLevel.SAFE,
                description="Analyze protein binding sites",
                requires_approval=False,
                reversible=True,
                side_effects=[]
            ),
            "calculate_drug_properties": ActionMetadata(
                tool_name="calculate_drug_properties",
                category=ActionCategory.ANALYZE,
                risk_level=ActionRiskLevel.SAFE,
                description="Calculate molecular properties",
                requires_approval=False,
                reversible=True,
                side_effects=[]
            ),
            "predict_gene_function": ActionMetadata(
                tool_name="predict_gene_function",
                category=ActionCategory.ANALYZE,
                risk_level=ActionRiskLevel.SAFE,
                description="Predict gene function from sequence",
                requires_approval=False,
                reversible=True,
                side_effects=[]
            ),

            # LOW risk actions (non-destructive writes)
            "send_notification": ActionMetadata(
                tool_name="send_notification",
                category=ActionCategory.NETWORK,
                risk_level=ActionRiskLevel.LOW,
                description="Send notification to user",
                requires_approval=False,
                reversible=True,
                side_effects=["external_communication"]
            ),

            # MEDIUM risk actions (data modifications)
            "update_database": ActionMetadata(
                tool_name="update_database",
                category=ActionCategory.MODIFY,
                risk_level=ActionRiskLevel.MEDIUM,
                description="Update database records",
                requires_approval=True,
                reversible=True,
                side_effects=["data_modification"]
            ),

            # HIGH risk actions (code execution)
            "execute_code": ActionMetadata(
                tool_name="execute_code",
                category=ActionCategory.EXECUTE,
                risk_level=ActionRiskLevel.HIGH,
                description="Execute arbitrary code",
                requires_approval=True,
                reversible=False,
                side_effects=["code_execution", "system_access"]
            ),
            "run_experiment": ActionMetadata(
                tool_name="run_experiment",
                category=ActionCategory.EXECUTE,
                risk_level=ActionRiskLevel.HIGH,
                description="Run computational experiment",
                requires_approval=True,
                reversible=False,
                side_effects=["resource_consumption", "external_api_calls"]
            ),

            # CRITICAL actions (system modifications)
            "deploy_model": ActionMetadata(
                tool_name="deploy_model",
                category=ActionCategory.DEPLOY,
                risk_level=ActionRiskLevel.CRITICAL,
                description="Deploy ML model to production",
                requires_approval=True,
                reversible=False,
                side_effects=["production_deployment", "system_modification"]
            ),
            "delete_data": ActionMetadata(
                tool_name="delete_data",
                category=ActionCategory.DELETE,
                risk_level=ActionRiskLevel.CRITICAL,
                description="Permanently delete data",
                requires_approval=True,
                reversible=False,
                side_effects=["data_loss"]
            ),
        }

        # Default risk levels by category
        self.category_risk_mapping = {
            ActionCategory.READ: ActionRiskLevel.SAFE,
            ActionCategory.SEARCH: ActionRiskLevel.SAFE,
            ActionCategory.ANALYZE: ActionRiskLevel.SAFE,
            ActionCategory.WRITE: ActionRiskLevel.LOW,
            ActionCategory.MODIFY: ActionRiskLevel.MEDIUM,
            ActionCategory.NETWORK: ActionRiskLevel.LOW,
            ActionCategory.EXECUTE: ActionRiskLevel.HIGH,
            ActionCategory.DEPLOY: ActionRiskLevel.CRITICAL,
            ActionCategory.DELETE: ActionRiskLevel.CRITICAL,
            ActionCategory.SYSTEM: ActionRiskLevel.CRITICAL,
        }

    def classify_action(self, tool_name: str, parameters: dict = None) -> ActionMetadata:
        """
        Classify a tool action by its impact

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters (can affect risk)

        Returns:
            ActionMetadata with classification details
        """
        # Check if tool is in registry
        if tool_name in self.tool_classifications:
            metadata = self.tool_classifications[tool_name]

            # Parameter-based risk adjustment
            if parameters:
                metadata = self._adjust_risk_by_parameters(metadata, parameters)

            return metadata

        # Unknown tool - infer from name
        return self._infer_classification(tool_name)

    def _adjust_risk_by_parameters(
        self,
        metadata: ActionMetadata,
        parameters: dict
    ) -> ActionMetadata:
        """
        Adjust risk level based on parameters

        Examples:
        - DELETE with large scope = higher risk
        - EXECUTE with dangerous commands = higher risk
        """
        # Create copy to avoid modifying original
        adjusted = ActionMetadata(
            tool_name=metadata.tool_name,
            category=metadata.category,
            risk_level=metadata.risk_level,
            description=metadata.description,
            requires_approval=metadata.requires_approval,
            reversible=metadata.reversible,
            side_effects=metadata.side_effects.copy()
        )

        # Check for dangerous parameter values
        dangerous_keywords = [
            "DROP", "DELETE", "TRUNCATE", "REMOVE",
            "sudo", "admin", "root", "*", "all"
        ]

        param_str = str(parameters).upper()
        for keyword in dangerous_keywords:
            if keyword in param_str:
                # Escalate risk level
                if adjusted.risk_level.value < ActionRiskLevel.HIGH.value:
                    adjusted.risk_level = ActionRiskLevel.HIGH
                    adjusted.requires_approval = True
                    adjusted.side_effects.append("dangerous_parameters")
                break

        return adjusted

    def _infer_classification(self, tool_name: str) -> ActionMetadata:
        """
        Infer classification from tool name for unknown tools

        Args:
            tool_name: Tool name

        Returns:
            Inferred ActionMetadata
        """
        tool_lower = tool_name.lower()

        # Infer category from name
        if any(keyword in tool_lower for keyword in ["search", "find", "query"]):
            category = ActionCategory.SEARCH
        elif any(keyword in tool_lower for keyword in ["read", "get", "fetch"]):
            category = ActionCategory.READ
        elif any(keyword in tool_lower for keyword in ["analyze", "calculate", "predict"]):
            category = ActionCategory.ANALYZE
        elif any(keyword in tool_lower for keyword in ["write", "create", "save"]):
            category = ActionCategory.WRITE
        elif any(keyword in tool_lower for keyword in ["update", "modify", "change"]):
            category = ActionCategory.MODIFY
        elif any(keyword in tool_lower for keyword in ["execute", "run", "eval"]):
            category = ActionCategory.EXECUTE
        elif any(keyword in tool_lower for keyword in ["deploy", "publish", "release"]):
            category = ActionCategory.DEPLOY
        elif any(keyword in tool_lower for keyword in ["delete", "remove", "drop"]):
            category = ActionCategory.DELETE
        else:
            # Unknown - default to medium risk
            category = ActionCategory.MODIFY

        risk_level = self.category_risk_mapping[category]

        return ActionMetadata(
            tool_name=tool_name,
            category=category,
            risk_level=risk_level,
            description=f"Unknown tool: {tool_name}",
            requires_approval=(risk_level.value >= ActionRiskLevel.MEDIUM.value),
            reversible=(risk_level.value <= ActionRiskLevel.LOW.value),
            side_effects=["unknown_tool"]
        )

    def register_tool(self, metadata: ActionMetadata):
        """
        Register a new tool classification

        Args:
            metadata: Tool metadata
        """
        self.tool_classifications[metadata.tool_name] = metadata

    def get_all_tools_by_risk(self, max_risk: ActionRiskLevel) -> List[str]:
        """
        Get all tools below a certain risk level

        Args:
            max_risk: Maximum allowed risk level

        Returns:
            List of tool names
        """
        return [
            name for name, meta in self.tool_classifications.items()
            if meta.risk_level.value <= max_risk.value
        ]
