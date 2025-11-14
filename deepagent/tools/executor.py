"""
Tool Execution Engine

Handles safe execution of tools with error handling,
timeouts, result validation, and automatic retry logic.

Author: Oluwafemi Idiakhoa
"""

from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
import time
import traceback
from enum import Enum


class ExecutionStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of tool execution"""
    status: ExecutionStatus
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_name: str = ""
    retry_count: int = 0
    retry_history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_name": self.tool_name,
            "retry_count": self.retry_count,
            "retry_history": self.retry_history
        }


class ToolExecutor:
    """
    Executes tools safely with monitoring, error handling, and retry logic

    Production features:
    - Automatic retry with exponential backoff using tenacity
    - Circuit breaker pattern to prevent cascading failures
    - Detailed error tracking and retry history
    - Graceful fallback if tenacity not available
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_on_errors: bool = True,
        use_circuit_breaker: bool = True,
        circuit_failure_threshold: int = 5
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_on_errors = retry_on_errors
        self.use_circuit_breaker = use_circuit_breaker
        self.circuit_failure_threshold = circuit_failure_threshold

        self.tool_implementations: Dict[str, Callable] = {}
        self.circuit_breaker_state: Dict[str, Dict[str, Any]] = {}

        # Try to import tenacity for retry logic
        self.tenacity_available = False
        try:
            from tenacity import (
                retry,
                stop_after_attempt,
                wait_exponential,
                retry_if_exception_type,
                RetryError
            )
            self.retry_decorator = retry
            self.stop_after_attempt = stop_after_attempt
            self.wait_exponential = wait_exponential
            self.retry_if_exception_type = retry_if_exception_type
            self.RetryError = RetryError
            self.tenacity_available = True
            print("Tenacity available for automatic retry logic")
        except ImportError:
            print("Warning: tenacity not installed. Retry logic will use simple fallback.")
            print("Install with: pip install tenacity")

        self._register_default_implementations()

    def register_tool(self, name: str, implementation: Callable) -> None:
        """Register a tool implementation"""
        self.tool_implementations[name] = implementation

        # Initialize circuit breaker state
        if self.use_circuit_breaker:
            self.circuit_breaker_state[name] = {
                "failure_count": 0,
                "last_failure_time": None,
                "is_open": False
            }

    def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None,
        retry: Optional[bool] = None
    ) -> ExecutionResult:
        """
        Execute a tool with given parameters, with retry logic and circuit breaker

        Args:
            tool_name: Name of tool to execute
            parameters: Parameters to pass to tool
            timeout: Optional timeout override
            retry: Override retry setting (default uses self.retry_on_errors)

        Returns:
            ExecutionResult with status, output, and retry information
        """
        start_time = time.time()
        timeout = timeout or self.timeout
        should_retry = retry if retry is not None else self.retry_on_errors

        # Check circuit breaker
        if self.use_circuit_breaker and self._is_circuit_open(tool_name):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                result=None,
                error=f"Circuit breaker open for '{tool_name}' due to repeated failures",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                retry_count=0
            )

        # Check if tool is registered
        if tool_name not in self.tool_implementations:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                result=None,
                error=f"Tool '{tool_name}' not found in registry",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                retry_count=0
            )

        # Execute with retry logic
        if should_retry and self.tenacity_available:
            return self._execute_with_tenacity(tool_name, parameters, timeout, start_time)
        elif should_retry:
            return self._execute_with_simple_retry(tool_name, parameters, timeout, start_time)
        else:
            return self._execute_once(tool_name, parameters, timeout, start_time)

    def _is_circuit_open(self, tool_name: str) -> bool:
        """Check if circuit breaker is open for a tool"""
        if tool_name not in self.circuit_breaker_state:
            return False

        state = self.circuit_breaker_state[tool_name]
        if not state["is_open"]:
            return False

        # Check if enough time has passed to attempt reset
        if state["last_failure_time"]:
            time_since_failure = time.time() - state["last_failure_time"]
            if time_since_failure > 60:  # 60 second cooldown
                state["is_open"] = False
                state["failure_count"] = 0
                return False

        return True

    def _record_failure(self, tool_name: str) -> None:
        """Record a tool execution failure for circuit breaker"""
        if not self.use_circuit_breaker or tool_name not in self.circuit_breaker_state:
            return

        state = self.circuit_breaker_state[tool_name]
        state["failure_count"] += 1
        state["last_failure_time"] = time.time()

        if state["failure_count"] >= self.circuit_failure_threshold:
            state["is_open"] = True
            print(f"Circuit breaker opened for '{tool_name}' after {state['failure_count']} failures")

    def _record_success(self, tool_name: str) -> None:
        """Record a tool execution success for circuit breaker"""
        if not self.use_circuit_breaker or tool_name not in self.circuit_breaker_state:
            return

        state = self.circuit_breaker_state[tool_name]
        state["failure_count"] = max(0, state["failure_count"] - 1)  # Decay failure count

    def _execute_with_tenacity(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float,
        start_time: float
    ) -> ExecutionResult:
        """Execute tool with tenacity retry logic"""
        retry_history = []

        # Create retrying wrapper
        @self.retry_decorator(
            stop=self.stop_after_attempt(self.max_retries),
            wait=self.wait_exponential(multiplier=1, min=1, max=10),
            retry=self.retry_if_exception_type(Exception),
            reraise=True
        )
        def execute_with_retry():
            try:
                tool_func = self.tool_implementations[tool_name]
                result = tool_func(**parameters)
                return result
            except Exception as e:
                retry_history.append(f"{type(e).__name__}: {str(e)}")
                raise

        try:
            result = execute_with_retry()
            execution_time = time.time() - start_time

            # Check timeout
            if execution_time > timeout:
                self._record_failure(tool_name)
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    result=result,
                    error=f"Execution exceeded timeout of {timeout}s",
                    execution_time=execution_time,
                    tool_name=tool_name,
                    retry_count=len(retry_history),
                    retry_history=retry_history
                )

            self._record_success(tool_name)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result=result,
                error=None,
                execution_time=execution_time,
                tool_name=tool_name,
                retry_count=len(retry_history),
                retry_history=retry_history
            )

        except self.RetryError as e:
            self._record_failure(tool_name)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                result=None,
                error=f"Failed after {self.max_retries} retries: {retry_history[-1] if retry_history else str(e)}",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                retry_count=len(retry_history),
                retry_history=retry_history
            )
        except Exception as e:
            self._record_failure(tool_name)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                result=None,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                retry_count=len(retry_history),
                retry_history=retry_history
            )

    def _execute_with_simple_retry(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float,
        start_time: float
    ) -> ExecutionResult:
        """Simple retry logic fallback when tenacity not available"""
        retry_history = []
        last_error = None

        for attempt in range(self.max_retries):
            try:
                tool_func = self.tool_implementations[tool_name]
                result = tool_func(**parameters)
                execution_time = time.time() - start_time

                if execution_time > timeout:
                    self._record_failure(tool_name)
                    return ExecutionResult(
                        status=ExecutionStatus.TIMEOUT,
                        result=result,
                        error=f"Execution exceeded timeout of {timeout}s",
                        execution_time=execution_time,
                        tool_name=tool_name,
                        retry_count=attempt,
                        retry_history=retry_history
                    )

                self._record_success(tool_name)
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result=result,
                    error=None,
                    execution_time=execution_time,
                    tool_name=tool_name,
                    retry_count=attempt,
                    retry_history=retry_history
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                retry_history.append(error_msg)
                last_error = e

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)
                    time.sleep(wait_time)

        self._record_failure(tool_name)
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            result=None,
            error=f"Failed after {self.max_retries} retries: {retry_history[-1] if retry_history else str(last_error)}",
            execution_time=time.time() - start_time,
            tool_name=tool_name,
            retry_count=self.max_retries,
            retry_history=retry_history
        )

    def _execute_once(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float,
        start_time: float
    ) -> ExecutionResult:
        """Execute tool once without retry"""
        try:
            tool_func = self.tool_implementations[tool_name]
            result = tool_func(**parameters)
            execution_time = time.time() - start_time

            if execution_time > timeout:
                self._record_failure(tool_name)
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    result=result,
                    error=f"Execution exceeded timeout of {timeout}s",
                    execution_time=execution_time,
                    tool_name=tool_name,
                    retry_count=0
                )

            self._record_success(tool_name)
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                result=result,
                error=None,
                execution_time=execution_time,
                tool_name=tool_name,
                retry_count=0
            )

        except Exception as e:
            self._record_failure(tool_name)
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                result=None,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                retry_count=0
            )

    def _register_default_implementations(self) -> None:
        """Register mock implementations for demo tools"""

        # Bioinformatics tools
        def get_protein_structure(protein_id: str) -> Dict[str, Any]:
            """Mock implementation"""
            return {
                "protein_id": protein_id,
                "structure": "mock_pdb_data",
                "residues": 234,
                "chains": ["A", "B"],
                "resolution": 2.1
            }

        def analyze_binding_sites(structure_data: Dict, ligand: str) -> list:
            """Mock implementation"""
            return [
                {"site_id": 1, "residues": ["ARG123", "ASP456"], "affinity": 8.5},
                {"site_id": 2, "residues": ["LYS789", "GLU012"], "affinity": 7.2}
            ]

        def sequence_alignment(sequences: list, algorithm: str) -> str:
            """Mock implementation"""
            return f"ALIGNMENT_RESULT using {algorithm}:\n" + "\n".join(sequences[:3])

        def predict_gene_function(sequence: str) -> Dict[str, float]:
            """Mock implementation"""
            return {
                "enzyme_activity": 0.85,
                "transcription_factor": 0.12,
                "structural_protein": 0.03
            }

        # Drug discovery tools
        def search_drugbank(query: str, search_type: str) -> list:
            """Mock implementation"""
            return [
                {
                    "drug_name": f"Drug_A_{query}",
                    "drugbank_id": "DB00001",
                    "indication": "Pain relief",
                    "targets": ["COX-1", "COX-2"]
                },
                {
                    "drug_name": f"Drug_B_{query}",
                    "drugbank_id": "DB00002",
                    "indication": "Inflammation",
                    "targets": ["TNF-alpha"]
                }
            ]

        def calculate_drug_properties(smiles: str) -> Dict[str, float]:
            """Mock implementation"""
            return {
                "molecular_weight": 180.16,
                "logP": 1.19,
                "hbd": 1,  # H-bond donors
                "hba": 4,  # H-bond acceptors
                "tpsa": 63.6,
                "lipinski_violations": 0
            }

        def predict_toxicity(compound: str) -> Dict[str, Any]:
            """Mock implementation"""
            return {
                "ld50": 200,  # mg/kg
                "carcinogenicity": 0.05,
                "hepatotoxicity": 0.12,
                "cardiotoxicity": 0.08,
                "risk_level": "low"
            }

        # Data analysis tools
        def statistical_analysis(data: list, test_type: str) -> Dict[str, float]:
            """Mock implementation"""
            import statistics
            return {
                "mean": statistics.mean(data) if data else 0,
                "median": statistics.median(data) if data else 0,
                "stdev": statistics.stdev(data) if len(data) > 1 else 0,
                "p_value": 0.023,
                "test_statistic": 2.45
            }

        def plot_visualization(data: Dict, plot_type: str) -> str:
            """Mock implementation"""
            return f"Generated {plot_type} plot with {len(data)} data points"

        # Information tools
        def search_pubmed(query: str, max_results: int) -> list:
            """Mock implementation"""
            return [
                {
                    "pmid": "12345678",
                    "title": f"Research on {query}",
                    "authors": ["Smith J", "Doe A"],
                    "year": 2023,
                    "abstract": f"This study investigates {query}..."
                },
                {
                    "pmid": "87654321",
                    "title": f"Novel findings in {query}",
                    "authors": ["Johnson B", "Williams C"],
                    "year": 2024,
                    "abstract": f"We present new insights into {query}..."
                }
            ][:max_results]

        def fetch_wikipedia(topic: str) -> str:
            """Mock implementation"""
            return f"Wikipedia summary for '{topic}':\n\nThis is a comprehensive article about {topic}. It covers the fundamental concepts, historical background, and current research in the field..."

        # Utility tools
        def convert_file_format(data: str, from_format: str, to_format: str) -> str:
            """Mock implementation"""
            return f"Converted data from {from_format} to {to_format}:\n{data[:100]}..."

        def send_notification(message: str) -> bool:
            """Mock implementation"""
            print(f"[NOTIFICATION] {message}")
            return True

        # Register all implementations
        implementations = {
            "get_protein_structure": get_protein_structure,
            "analyze_binding_sites": analyze_binding_sites,
            "sequence_alignment": sequence_alignment,
            "predict_gene_function": predict_gene_function,
            "search_drugbank": search_drugbank,
            "calculate_drug_properties": calculate_drug_properties,
            "predict_toxicity": predict_toxicity,
            "statistical_analysis": statistical_analysis,
            "plot_visualization": plot_visualization,
            "search_pubmed": search_pubmed,
            "fetch_wikipedia": fetch_wikipedia,
            "convert_file_format": convert_file_format,
            "send_notification": send_notification,
        }

        for name, func in implementations.items():
            self.register_tool(name, func)
