"""
Dense Tool Retrieval System

Implements semantic search over large-scale API repositories
for dynamic tool discovery during reasoning.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json


@dataclass
class ToolDefinition:
    """Represents a tool/API that can be discovered and used"""
    name: str
    description: str
    category: str
    parameters: List[Dict[str, Any]]
    returns: str
    examples: List[str]
    api_endpoint: Optional[str] = None
    auth_required: bool = False
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
            "api_endpoint": self.api_endpoint,
            "auth_required": self.auth_required
        }

    def get_signature(self) -> str:
        """Get function signature string"""
        params = ", ".join([f"{p['name']}: {p['type']}" for p in self.parameters])
        return f"{self.name}({params}) -> {self.returns}"


class DenseToolRetriever:
    """
    Semantic search over tool repository using dense embeddings
    Simulates searching 10,000+ tools like RapidAPI or ToolHop
    """

    def __init__(self, embedding_dim: int = 384):
        self.tools: List[ToolDefinition] = []
        self.embedding_dim = embedding_dim
        self.tool_embeddings: Optional[np.ndarray] = None
        self.category_index: Dict[str, List[int]] = defaultdict(list)

    def add_tool(self, tool: ToolDefinition) -> None:
        """Add a tool to the registry"""
        # Generate embedding if not provided
        if tool.embedding is None:
            tool.embedding = self._generate_embedding(tool)

        self.tools.append(tool)
        idx = len(self.tools) - 1
        self.category_index[tool.category].append(idx)

        # Update embedding matrix
        self._update_embedding_matrix()

    def add_tools_batch(self, tools: List[ToolDefinition]) -> None:
        """Add multiple tools efficiently"""
        for tool in tools:
            if tool.embedding is None:
                tool.embedding = self._generate_embedding(tool)

            self.tools.append(tool)
            idx = len(self.tools) - 1
            self.category_index[tool.category].append(idx)

        self._update_embedding_matrix()

    def _generate_embedding(self, tool: ToolDefinition) -> np.ndarray:
        """
        Generate semantic embedding for a tool
        In production, use sentence-transformers or OpenAI embeddings
        Here we use a simple hash-based embedding for demo
        """
        # Combine tool information into searchable text
        text = f"{tool.name} {tool.description} {tool.category}"

        # Simple hash-based embedding (replace with real embeddings in production)
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        return embedding

    def _update_embedding_matrix(self) -> None:
        """Update the matrix of all tool embeddings"""
        if self.tools:
            self.tool_embeddings = np.vstack([tool.embedding for tool in self.tools])

    def search(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.0
    ) -> List[Tuple[ToolDefinition, float]]:
        """
        Search for relevant tools using semantic similarity

        Args:
            query: Natural language description of needed functionality
            top_k: Number of top results to return
            category_filter: Optional category to filter by
            min_similarity: Minimum similarity threshold

        Returns:
            List of (tool, similarity_score) tuples
        """
        if not self.tools:
            return []

        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)

        # Compute similarities
        if category_filter and category_filter in self.category_index:
            # Filter by category
            indices = self.category_index[category_filter]
            embeddings = self.tool_embeddings[indices]
            similarities = embeddings @ query_embedding

            results = [
                (self.tools[indices[i]], float(similarities[i]))
                for i in range(len(indices))
                if similarities[i] >= min_similarity
            ]
        else:
            # Search all tools
            similarities = self.tool_embeddings @ query_embedding

            results = [
                (self.tools[i], float(similarities[i]))
                for i in range(len(self.tools))
                if similarities[i] >= min_similarity
            ]

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query"""
        np.random.seed(hash(query) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def search_by_category(self, category: str) -> List[ToolDefinition]:
        """Get all tools in a category"""
        if category not in self.category_index:
            return []

        indices = self.category_index[category]
        return [self.tools[i] for i in indices]

    def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve specific tool by name"""
        for tool in self.tools:
            if tool.name.lower() == name.lower():
                return tool
        return None

    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.category_index.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": len(self.category_index),
            "category_distribution": {
                cat: len(indices) for cat, indices in self.category_index.items()
            }
        }

    def save_registry(self, filepath: str) -> None:
        """Save tool registry to file"""
        data = {
            "tools": [tool.to_dict() for tool in self.tools],
            "embedding_dim": self.embedding_dim
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_registry(self, filepath: str) -> None:
        """Load tool registry from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tools = []
        for tool_data in data["tools"]:
            tool = ToolDefinition(**tool_data)
            tool.embedding = self._generate_embedding(tool)
            tools.append(tool)

        self.add_tools_batch(tools)


class ToolRegistry:
    """
    Central registry for managing available tools
    Integrates with DenseToolRetriever for discovery
    """

    def __init__(self):
        self.retriever = DenseToolRetriever()
        self.loaded_tools: Dict[str, ToolDefinition] = {}

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a new tool"""
        self.retriever.add_tool(tool)
        self.loaded_tools[tool.name] = tool

    def register_tools_batch(self, tools: List[ToolDefinition]) -> None:
        """Register multiple tools"""
        self.retriever.add_tools_batch(tools)
        for tool in tools:
            self.loaded_tools[tool.name] = tool

    def discover_tools(
        self,
        task_description: str,
        max_tools: int = 5,
        category: Optional[str] = None
    ) -> List[ToolDefinition]:
        """
        Discover relevant tools for a task

        This is the key method that enables dynamic tool discovery
        during the reasoning loop.
        """
        results = self.retriever.search(
            query=task_description,
            top_k=max_tools,
            category_filter=category,
            min_similarity=0.3
        )

        return [tool for tool, _ in results]

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name"""
        return self.loaded_tools.get(name)

    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self.loaded_tools.values())


def create_sample_tool_registry() -> ToolRegistry:
    """
    Create a sample registry with mock tools
    In production, this would load from RapidAPI, ToolHop, etc.
    """
    registry = ToolRegistry()

    # Bioinformatics tools
    bio_tools = [
        ToolDefinition(
            name="get_protein_structure",
            description="Retrieve 3D protein structure from PDB database using protein ID or name",
            category="bioinformatics",
            parameters=[
                {"name": "protein_id", "type": "str", "description": "PDB ID or protein name"}
            ],
            returns="Dict[str, Any]",
            examples=["get_protein_structure('BRCA1')", "get_protein_structure('1ABC')"],
            api_endpoint="https://www.rcsb.org/structure/"
        ),
        ToolDefinition(
            name="analyze_binding_sites",
            description="Analyze and identify binding sites in protein structure for drug interactions",
            category="bioinformatics",
            parameters=[
                {"name": "structure_data", "type": "Dict", "description": "Protein structure data"},
                {"name": "ligand", "type": "str", "description": "Ligand molecule to test"}
            ],
            returns="List[Dict]",
            examples=["analyze_binding_sites(structure, 'ATP')"],
            api_endpoint="https://api.bindingdb.org/"
        ),
        ToolDefinition(
            name="sequence_alignment",
            description="Perform multiple sequence alignment using BLAST or similar algorithm",
            category="bioinformatics",
            parameters=[
                {"name": "sequences", "type": "List[str]", "description": "List of sequences to align"},
                {"name": "algorithm", "type": "str", "description": "Algorithm (blast, clustal, muscle)"}
            ],
            returns="str",
            examples=["sequence_alignment(['ATCG...', 'ATGC...'], 'blast')"]
        ),
        ToolDefinition(
            name="predict_gene_function",
            description="Predict gene function from sequence using machine learning models",
            category="bioinformatics",
            parameters=[
                {"name": "sequence", "type": "str", "description": "DNA or protein sequence"}
            ],
            returns="Dict[str, float]",
            examples=["predict_gene_function('ATCGATCG...')"]
        ),
    ]

    # Drug discovery tools
    drug_tools = [
        ToolDefinition(
            name="search_drugbank",
            description="Search DrugBank database for drug information, interactions, and targets",
            category="drug_discovery",
            parameters=[
                {"name": "query", "type": "str", "description": "Drug name or compound"},
                {"name": "search_type", "type": "str", "description": "Type: name, target, indication"}
            ],
            returns="List[Dict]",
            examples=["search_drugbank('aspirin', 'name')"],
            api_endpoint="https://api.drugbank.com/"
        ),
        ToolDefinition(
            name="calculate_drug_properties",
            description="Calculate molecular properties (Lipinski's rule, ADME properties) for drug candidates",
            category="drug_discovery",
            parameters=[
                {"name": "smiles", "type": "str", "description": "SMILES notation of molecule"}
            ],
            returns="Dict[str, float]",
            examples=["calculate_drug_properties('CC(=O)OC1=CC=CC=C1C(=O)O')"]
        ),
        ToolDefinition(
            name="predict_toxicity",
            description="Predict toxicity and side effects of drug compounds using ML models",
            category="drug_discovery",
            parameters=[
                {"name": "compound", "type": "str", "description": "Compound identifier or SMILES"}
            ],
            returns="Dict[str, Any]",
            examples=["predict_toxicity('CC(=O)OC1=CC=CC=C1C(=O)O')"]
        ),
    ]

    # Data analysis tools
    data_tools = [
        ToolDefinition(
            name="statistical_analysis",
            description="Perform statistical analysis on dataset (t-test, ANOVA, correlation)",
            category="data_analysis",
            parameters=[
                {"name": "data", "type": "List[float]", "description": "Numerical data"},
                {"name": "test_type", "type": "str", "description": "Type of test to perform"}
            ],
            returns="Dict[str, float]",
            examples=["statistical_analysis([1,2,3,4,5], 't-test')"]
        ),
        ToolDefinition(
            name="plot_visualization",
            description="Create scientific plots and visualizations from data",
            category="data_analysis",
            parameters=[
                {"name": "data", "type": "Dict", "description": "Data to visualize"},
                {"name": "plot_type", "type": "str", "description": "Plot type: line, bar, scatter, heatmap"}
            ],
            returns="str",
            examples=["plot_visualization(data, 'heatmap')"]
        ),
    ]

    # Web search and information tools
    web_tools = [
        ToolDefinition(
            name="search_pubmed",
            description="Search PubMed for scientific literature and research papers",
            category="information",
            parameters=[
                {"name": "query", "type": "str", "description": "Search query"},
                {"name": "max_results", "type": "int", "description": "Maximum results to return"}
            ],
            returns="List[Dict]",
            examples=["search_pubmed('CRISPR gene editing', 10)"],
            api_endpoint="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        ),
        ToolDefinition(
            name="fetch_wikipedia",
            description="Fetch summary and content from Wikipedia articles",
            category="information",
            parameters=[
                {"name": "topic", "type": "str", "description": "Topic to search"}
            ],
            returns="str",
            examples=["fetch_wikipedia('DNA replication')"]
        ),
    ]

    # Utility tools
    utility_tools = [
        ToolDefinition(
            name="convert_file_format",
            description="Convert between different file formats (CSV, JSON, XML, FASTA)",
            category="utility",
            parameters=[
                {"name": "data", "type": "str", "description": "Input data"},
                {"name": "from_format", "type": "str", "description": "Source format"},
                {"name": "to_format", "type": "str", "description": "Target format"}
            ],
            returns="str",
            examples=["convert_file_format(data, 'csv', 'json')"]
        ),
        ToolDefinition(
            name="send_notification",
            description="Send notification or alert when task completes",
            category="utility",
            parameters=[
                {"name": "message", "type": "str", "description": "Notification message"}
            ],
            returns="bool",
            examples=["send_notification('Analysis complete')"]
        ),
    ]

    all_tools = bio_tools + drug_tools + data_tools + web_tools + utility_tools
    registry.register_tools_batch(all_tools)

    return registry
