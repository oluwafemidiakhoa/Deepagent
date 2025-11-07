"""
GENESIS-AI Integration Example

Demonstrates how to use DeepAgent for synthetic biology and
bioinformatics workflows, as mentioned in the original discussion.
"""

import sys
sys.path.insert(0, '..')

from deepagent import DeepAgent, AgentConfig, ToolDefinition


def create_genesis_ai_agent():
    """Create specialized agent for synthetic biology"""

    config = AgentConfig(
        max_steps=100,
        max_tools_per_search=10,
        verbose=True
    )

    agent = DeepAgent(config)

    # Add specialized synthetic biology tools
    synbio_tools = [
        ToolDefinition(
            name="design_genetic_circuit",
            description="Design genetic circuits for synthetic biology applications",
            category="synthetic_biology",
            parameters=[
                {"name": "function", "type": "str", "description": "Desired circuit function"},
                {"name": "organism", "type": "str", "description": "Target organism"}
            ],
            returns="Dict[str, Any]",
            examples=["design_genetic_circuit('oscillator', 'E.coli')"]
        ),
        ToolDefinition(
            name="simulate_metabolic_pathway",
            description="Simulate metabolic pathways and predict metabolite production",
            category="synthetic_biology",
            parameters=[
                {"name": "pathway", "type": "str", "description": "Pathway name or genes"},
                {"name": "conditions", "type": "Dict", "description": "Growth conditions"}
            ],
            returns="Dict[str, float]",
            examples=["simulate_metabolic_pathway('glycolysis', {'glucose': 10})"]
        ),
        ToolDefinition(
            name="optimize_codon_usage",
            description="Optimize codon usage for expression in target organism",
            category="synthetic_biology",
            parameters=[
                {"name": "sequence", "type": "str", "description": "Protein or DNA sequence"},
                {"name": "organism", "type": "str", "description": "Target organism"}
            ],
            returns="str",
            examples=["optimize_codon_usage('ATGCGT...', 'E.coli')"]
        ),
        ToolDefinition(
            name="predict_protein_folding",
            description="Predict protein structure using AlphaFold-like models",
            category="synthetic_biology",
            parameters=[
                {"name": "sequence", "type": "str", "description": "Amino acid sequence"}
            ],
            returns="Dict[str, Any]",
            examples=["predict_protein_folding('MKTAYIAKQRQ...')"]
        ),
        ToolDefinition(
            name="design_crispr_guide",
            description="Design CRISPR guide RNAs for gene editing",
            category="synthetic_biology",
            parameters=[
                {"name": "target_gene", "type": "str", "description": "Gene to target"},
                {"name": "genome", "type": "str", "description": "Organism genome"}
            ],
            returns="List[str]",
            examples=["design_crispr_guide('BRCA1', 'human')"]
        ),
    ]

    # Implement synthetic biology tools
    def design_genetic_circuit(function: str, organism: str) -> dict:
        return {
            "circuit_type": function,
            "organism": organism,
            "components": [
                {"type": "promoter", "name": "pTet"},
                {"type": "rbs", "name": "B0034"},
                {"type": "cds", "name": f"{function}_gene"},
                {"type": "terminator", "name": "T1"}
            ],
            "estimated_performance": 0.85
        }

    def simulate_metabolic_pathway(pathway: str, conditions: dict) -> dict:
        return {
            "pathway": pathway,
            "metabolite_production": {
                "pyruvate": 45.2,
                "ATP": 30.5,
                "NADH": 12.8
            },
            "growth_rate": 0.65,
            "yield": 0.78
        }

    def optimize_codon_usage(sequence: str, organism: str) -> str:
        return f"OPTIMIZED_{sequence[:20]}..._{organism}"

    def predict_protein_folding(sequence: str) -> dict:
        return {
            "confidence": 0.92,
            "structure": "alpha_helix_beta_sheet",
            "pLDDT_score": 88.5,
            "coordinates": "PDB_format_data"
        }

    def design_crispr_guide(target_gene: str, genome: str) -> list:
        return [
            f"GUIDE1_{target_gene}_5UTR",
            f"GUIDE2_{target_gene}_CDS",
            f"GUIDE3_{target_gene}_3UTR"
        ]

    # Register all tools
    for tool in synbio_tools:
        impl = locals()[tool.name]
        agent.add_custom_tool(tool, impl)

    return agent


def example_1_protein_engineering():
    """Example: Protein engineering workflow"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 1: Protein Engineering")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    result = agent.run(
        task="""Design a new enzyme with improved catalytic activity:
        1. Predict the folding of the target protein sequence
        2. Analyze binding sites
        3. Optimize codon usage for E.coli expression
        """
    )

    print(f"\n{result.answer}")


def example_2_metabolic_engineering():
    """Example: Metabolic pathway optimization"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 2: Metabolic Engineering")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    result = agent.run(
        task="""Optimize production of a biofuel compound:
        1. Simulate the metabolic pathway for biofuel production
        2. Design genetic circuits to regulate the pathway
        3. Predict protein structures of key enzymes
        """
    )

    print(f"\n{result.answer}")


def example_3_crispr_design():
    """Example: CRISPR gene editing design"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 3: CRISPR Gene Editing")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    result = agent.run(
        task="""Design a CRISPR system to edit the BRCA1 gene:
        1. Get the protein structure of BRCA1
        2. Design guide RNAs for precise targeting
        3. Search literature for similar editing approaches
        """
    )

    print(f"\n{result.answer}")


def example_4_drug_target_discovery():
    """Example: Drug target discovery and validation"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 4: Drug Target Discovery")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    result = agent.run(
        task="""Identify and validate a new drug target:
        1. Search PubMed for disease-related proteins
        2. Analyze binding sites in candidate proteins
        3. Search DrugBank for similar drug interactions
        4. Predict toxicity of potential compounds
        """
    )

    print(f"\n{result.answer}")
    print(f"\nTools used: {', '.join(result.tools_used)}")
    print(f"Total steps: {result.total_steps}")


def example_5_automated_lab_workflow():
    """Example: Automated laboratory workflow planning"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 5: Automated Lab Workflow")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    result = agent.run(
        task="""Plan an automated workflow for protein expression:
        1. Optimize codon usage for the target protein
        2. Design the expression vector with genetic circuits
        3. Simulate expected protein yield
        4. Predict the final protein structure
        """,
        max_steps=50
    )

    print(f"\n{result.answer}")

    # Show memory of the workflow
    print("\n--- Agent Memory Summary ---")
    print(agent.get_memory_summary())


def example_6_multi_step_synthesis():
    """Example: Multi-step synthetic biology experiment"""
    print("\n" + "=" * 80)
    print("GENESIS-AI Example 6: Multi-Step Synthesis")
    print("=" * 80)

    agent = create_genesis_ai_agent()

    # Complex multi-step task
    result = agent.run(
        task="""Design and validate a complete synthetic biology construct:
        1. Design a genetic circuit for producing a therapeutic protein
        2. Optimize all coding sequences for E.coli
        3. Simulate the metabolic impact
        4. Predict the protein structure
        5. Analyze potential binding sites for drug interactions
        6. Search literature for safety considerations
        """,
        max_steps=80
    )

    print(f"\n{result.answer}")

    # Get detailed statistics
    stats = agent.get_tool_stats()
    print("\n--- Tool Usage Statistics ---")
    for key, value in stats.items():
        print(f"{key}: {value}")


def main():
    """Run all GENESIS-AI integration examples"""
    print("\n" + "=" * 80)
    print("GENESIS-AI INTEGRATION EXAMPLES")
    print("Synthetic Biology and Bioinformatics Workflows")
    print("=" * 80)

    try:
        example_1_protein_engineering()
        example_2_metabolic_engineering()
        example_3_crispr_design()
        example_4_drug_target_discovery()
        example_5_automated_lab_workflow()
        example_6_multi_step_synthesis()

        print("\n" + "=" * 80)
        print("ALL GENESIS-AI EXAMPLES COMPLETED")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
