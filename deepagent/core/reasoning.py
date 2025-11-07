"""
End-to-End Reasoning Loop

Implements the core reasoning loop that runs inside the model,
integrating tool discovery, execution, and memory management.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class ReasoningStep(Enum):
    """Types of reasoning steps"""
    THINK = "think"
    SEARCH_TOOLS = "search_tools"
    EXECUTE_TOOL = "execute_tool"
    OBSERVE = "observe"
    CONCLUDE = "conclude"


@dataclass
class ReasoningTrace:
    """Records a single step in the reasoning process"""
    step_number: int
    step_type: ReasoningStep
    content: str
    tool_name: Optional[str] = None
    tool_result: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "step_type": self.step_type.value,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_result": str(self.tool_result) if self.tool_result else None,
            "timestamp": self.timestamp
        }


@dataclass
class ReasoningResult:
    """Final result of reasoning process"""
    answer: str
    success: bool
    reasoning_trace: List[ReasoningTrace]
    tools_used: List[str]
    total_steps: int
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "success": self.success,
            "reasoning_trace": [t.to_dict() for t in self.reasoning_trace],
            "tools_used": self.tools_used,
            "total_steps": self.total_steps,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


class ReasoningEngine:
    """
    End-to-end reasoning engine

    This implements the core loop that distinguishes DeepAgent from
    traditional ReAct frameworks. The reasoning happens in a continuous
    stream rather than external orchestration.
    """

    def __init__(self, max_steps: int = 50, verbose: bool = True):
        self.max_steps = max_steps
        self.verbose = verbose
        self.reasoning_trace: List[ReasoningTrace] = []
        self.current_step = 0

    def reason(
        self,
        task: str,
        context: str,
        tool_discovery_fn: callable,
        tool_execution_fn: callable,
        llm_generate_fn: callable
    ) -> ReasoningResult:
        """
        Main reasoning loop

        Args:
            task: The task to accomplish
            context: Current memory context
            tool_discovery_fn: Function to discover relevant tools
            tool_execution_fn: Function to execute tools
            llm_generate_fn: Function to generate next reasoning step

        Returns:
            ReasoningResult with answer and trace
        """
        start_time = time.time()
        self.reasoning_trace = []
        self.current_step = 0
        tools_used = []

        # Initial prompt
        prompt = self._build_initial_prompt(task, context)

        for step in range(self.max_steps):
            self.current_step = step

            # Generate next reasoning step using LLM
            response = llm_generate_fn(prompt)

            # Parse the response to determine action
            action = self._parse_action(response)

            if action["type"] == "think":
                # Internal reasoning
                trace = ReasoningTrace(
                    step_number=step,
                    step_type=ReasoningStep.THINK,
                    content=action["content"]
                )
                self.reasoning_trace.append(trace)

                if self.verbose:
                    print(f"\n[STEP {step}] THINK: {action['content'][:100]}...")

                # Continue reasoning
                prompt = self._update_prompt(prompt, trace)

            elif action["type"] == "search_tools":
                # Discover relevant tools
                query = action["query"]
                tools = tool_discovery_fn(query, max_tools=5)

                trace = ReasoningTrace(
                    step_number=step,
                    step_type=ReasoningStep.SEARCH_TOOLS,
                    content=f"Searching for tools: {query}",
                    tool_result=tools
                )
                self.reasoning_trace.append(trace)

                if self.verbose:
                    print(f"\n[STEP {step}] SEARCH_TOOLS: Found {len(tools)} tools")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description[:60]}...")

                # Add tool options to prompt
                prompt = self._update_prompt_with_tools(prompt, tools, trace)

            elif action["type"] == "execute_tool":
                # Execute a tool
                tool_name = action["tool_name"]
                parameters = action["parameters"]

                trace = ReasoningTrace(
                    step_number=step,
                    step_type=ReasoningStep.EXECUTE_TOOL,
                    content=f"Executing {tool_name} with {parameters}",
                    tool_name=tool_name
                )

                if self.verbose:
                    print(f"\n[STEP {step}] EXECUTE_TOOL: {tool_name}")
                    print(f"  Parameters: {parameters}")

                # Execute the tool
                result = tool_execution_fn(tool_name, parameters)
                trace.tool_result = result

                self.reasoning_trace.append(trace)
                tools_used.append(tool_name)

                if self.verbose:
                    print(f"  Result: {str(result)[:100]}...")

                # Add observation to prompt
                prompt = self._update_prompt_with_observation(prompt, trace, result)

            elif action["type"] == "observe":
                # Reflect on results
                trace = ReasoningTrace(
                    step_number=step,
                    step_type=ReasoningStep.OBSERVE,
                    content=action["content"]
                )
                self.reasoning_trace.append(trace)

                if self.verbose:
                    print(f"\n[STEP {step}] OBSERVE: {action['content'][:100]}...")

                prompt = self._update_prompt(prompt, trace)

            elif action["type"] == "conclude":
                # Task complete
                trace = ReasoningTrace(
                    step_number=step,
                    step_type=ReasoningStep.CONCLUDE,
                    content=action["answer"]
                )
                self.reasoning_trace.append(trace)

                if self.verbose:
                    print(f"\n[STEP {step}] CONCLUDE: Task completed")

                return ReasoningResult(
                    answer=action["answer"],
                    success=True,
                    reasoning_trace=self.reasoning_trace,
                    tools_used=list(set(tools_used)),
                    total_steps=step + 1,
                    execution_time=time.time() - start_time
                )

        # Max steps reached
        return ReasoningResult(
            answer="Maximum reasoning steps reached without conclusion",
            success=False,
            reasoning_trace=self.reasoning_trace,
            tools_used=list(set(tools_used)),
            total_steps=self.max_steps,
            execution_time=time.time() - start_time,
            metadata={"termination_reason": "max_steps"}
        )

    def _build_initial_prompt(self, task: str, context: str) -> str:
        """Build the initial reasoning prompt"""
        return f"""You are an autonomous reasoning agent with access to dynamic tool discovery.

TASK: {task}

MEMORY CONTEXT:
{context}

REASONING PROTOCOL:
You can perform the following actions in your reasoning loop:

1. THINK: Internal reasoning about the task
   Format: THINK: <your reasoning>

2. SEARCH_TOOLS: Discover relevant tools for a specific need
   Format: SEARCH_TOOLS: <description of needed functionality>

3. EXECUTE_TOOL: Use a discovered tool
   Format: EXECUTE_TOOL: <tool_name>
   PARAMETERS: {{"param1": "value1", "param2": "value2"}}

4. OBSERVE: Reflect on tool results
   Format: OBSERVE: <your observation>

5. CONCLUDE: Provide final answer
   Format: CONCLUDE: <final answer>

Begin your reasoning:
"""

    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into action"""
        response = response.strip()

        if response.startswith("THINK:"):
            return {
                "type": "think",
                "content": response[6:].strip()
            }
        elif response.startswith("SEARCH_TOOLS:"):
            return {
                "type": "search_tools",
                "query": response[13:].strip()
            }
        elif response.startswith("EXECUTE_TOOL:"):
            lines = response.split("\n")
            tool_name = lines[0][13:].strip()

            # Parse parameters (simplified)
            parameters = {}
            if len(lines) > 1 and "PARAMETERS:" in lines[1]:
                # In production, use proper JSON parsing
                params_str = lines[1].split("PARAMETERS:")[1].strip()
                try:
                    import json
                    parameters = json.loads(params_str)
                except:
                    parameters = {}

            return {
                "type": "execute_tool",
                "tool_name": tool_name,
                "parameters": parameters
            }
        elif response.startswith("OBSERVE:"):
            return {
                "type": "observe",
                "content": response[8:].strip()
            }
        elif response.startswith("CONCLUDE:"):
            return {
                "type": "conclude",
                "answer": response[9:].strip()
            }
        else:
            # Default to thinking
            return {
                "type": "think",
                "content": response
            }

    def _update_prompt(self, prompt: str, trace: ReasoningTrace) -> str:
        """Add reasoning step to prompt"""
        prompt += f"\n\n{trace.step_type.value.upper()}: {trace.content}"
        return prompt

    def _update_prompt_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        trace: ReasoningTrace
    ) -> str:
        """Add discovered tools to prompt"""
        prompt += f"\n\nSEARCH_TOOLS: {trace.content}"
        prompt += "\n\nAVAILABLE TOOLS:"

        for tool in tools:
            prompt += f"\n- {tool.name}: {tool.description}"
            prompt += f"\n  Signature: {tool.get_signature()}"

        prompt += "\n\nWhat would you like to do next?"
        return prompt

    def _update_prompt_with_observation(
        self,
        prompt: str,
        trace: ReasoningTrace,
        result: Any
    ) -> str:
        """Add tool execution result to prompt"""
        prompt += f"\n\nEXECUTE_TOOL: {trace.tool_name}"
        prompt += f"\nRESULT: {result}"
        prompt += "\n\nWhat do you observe from this result?"
        return prompt

    def get_trace_summary(self) -> str:
        """Get human-readable summary of reasoning trace"""
        summary = []
        summary.append("=" * 60)
        summary.append("REASONING TRACE SUMMARY")
        summary.append("=" * 60)

        for trace in self.reasoning_trace:
            summary.append(f"\nStep {trace.step_number}: {trace.step_type.value.upper()}")
            summary.append(f"  {trace.content[:100]}...")

            if trace.tool_name:
                summary.append(f"  Tool: {trace.tool_name}")

            if trace.tool_result:
                summary.append(f"  Result: {str(trace.tool_result)[:100]}...")

        summary.append("\n" + "=" * 60)
        return "\n".join(summary)
