# -*- coding: utf-8 -*-
"""UVM Analysis System with Graph Database and GPU Acceleration

Original from Colab notebook:
https://colab.research.google.com/drive/1BMkryTiTdh1jfGf675imgBnEc-LgXp1W
"""

import pandas as pd
import json
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
import networkx as nx
from arango.client import ArangoClient
from langchain.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from pyngrok import ngrok
import pdfplumber
import re
from pypdf import PdfReader
import fitz
import os
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
import matplotlib.pyplot as plt

# Database Configuration
ARANGO_HOST = "https://38676237cba5.arangodb.cloud:8529"
USERNAME = "root"
PASSWORD = "UV9bhJ37alFbpNU8aeP5"
DB_NAME = "uvm_db"
GRAPH_NAME = "CodeGraph"

# Environment setup
os.environ["DATABASE_HOST"] = ARANGO_HOST
os.environ["DATABASE_NAME"] = DB_NAME
os.environ["DATABASE_USERNAME"] = USERNAME
os.environ["DATABASE_PASSWORD"] = PASSWORD
os.environ["GRAPH_NAME"] = GRAPH_NAME

def setup_database():
    """Initialize ArangoDB database and collections"""
    if not os.getenv("DATABASE_HOST") or not os.getenv("DATABASE_NAME"):
        raise ValueError("❌ ArangoDB database environment variables are not set!")

    client = ArangoClient(hosts=os.getenv("DATABASE_HOST"), request_timeout=None)
    sys_db = client.db("_system", username=USERNAME, password=PASSWORD)

    if not sys_db.has_database(DB_NAME):
        sys_db.create_database(DB_NAME)

    db = client.db(DB_NAME, username=USERNAME, password=PASSWORD, verify=True)

    # Create Graph
    if not db.has_graph(GRAPH_NAME):
        graph = db.create_graph(GRAPH_NAME)
        print("✅ Created Graph: SOC_Graph")
    else:
        graph = db.graph(GRAPH_NAME)

    # Create collections if they don't exist
    if not db.has_collection("SOC_Components"):
        db.create_collection("SOC_Components")
        print("✅ Created Collection: SOC_Components")

    if not db.has_collection("Contains"):
        db.create_collection("Contains", edge=True)
        print("✅ Created Edge Collection: Contains")

    # Define or update the edge definition
    existing_edges = [edge["edge_collection"] for edge in graph.edge_definitions()]

    if "Contains" in existing_edges:
        graph.replace_edge_definition(
            edge_collection="Contains",
            from_vertex_collections=["SOC_Components"],
            to_vertex_collections=["SOC_Components"]
        )
        print("✅ Updated Edge Definition: Contains")
    else:
        graph.create_edge_definition(
            edge_collection="Contains",
            from_vertex_collections=["SOC_Components"],
            to_vertex_collections=["SOC_Components"]
        )
        print("✅ Created Edge Definition: Contains")

    # SoC-Level UVM Components
    soc_components = [
        # SoC Level
        {"_key": "soc_test", "name": "SoC Test", "level": "SoC"},
        {"_key": "soc_env", "name": "SoC Environment", "level": "SoC"},
        {"_key": "soc_config_db", "name": "SoC Config DB", "level": "SoC"},

        # Unit Level (Subsystems)
        {"_key": "cpu_subsystem", "name": "CPU Subsystem", "level": "Unit"},
        {"_key": "memory_subsystem", "name": "Memory Subsystem", "level": "Unit"},
        {"_key": "peripheral_subsystem", "name": "Peripheral Subsystem", "level": "Unit"},
        {"_key": "noc_subsystem", "name": "Network-on-Chip Subsystem", "level": "Unit"},

        # IP Level
        {"_key": "cpu_agent", "name": "CPU Agent", "level": "IP"},
        {"_key": "memory_agent", "name": "Memory Agent", "level": "IP"},
        {"_key": "peripheral_agent", "name": "Peripheral Agent", "level": "IP"},
        {"_key": "noc_agent", "name": "NoC Agent", "level": "IP"},

        # Protocol-Specific Components
        {"_key": "uart_driver", "name": "UART Driver", "level": "IP"},
        {"_key": "spi_driver", "name": "SPI Driver", "level": "IP"},
        {"_key": "axi_driver", "name": "AXI Driver", "level": "IP"},
        {"_key": "uart_monitor", "name": "UART Monitor", "level": "IP"},
        {"_key": "spi_monitor", "name": "SPI Monitor", "level": "IP"},
        {"_key": "axi_monitor", "name": "AXI Monitor", "level": "IP"},
        {"_key": "scoreboard", "name": "Scoreboard", "level": "IP"},
    ]

    # Insert Components into Database
    for component in soc_components:
        if not db["SOC_Components"].has(component["_key"]):
            db["SOC_Components"].insert(component)
            print(f"✅ Inserted Component: {component['name']} ({component['level']})")

    # SoC Hierarchy Relationships (Edges)
    soc_edges = [
        # SoC-Level Connections
        {"_from": "SOC_Components/soc_test", "_to": "SOC_Components/soc_env"},
        {"_from": "SOC_Components/soc_env", "_to": "SOC_Components/soc_config_db"},

        # Unit-Level Connections
        {"_from": "SOC_Components/soc_env", "_to": "SOC_Components/cpu_subsystem"},
        {"_from": "SOC_Components/soc_env", "_to": "SOC_Components/memory_subsystem"},
        {"_from": "SOC_Components/soc_env", "_to": "SOC_Components/peripheral_subsystem"},
        {"_from": "SOC_Components/soc_env", "_to": "SOC_Components/noc_subsystem"},

        # IP-Level Connections
        {"_from": "SOC_Components/cpu_subsystem", "_to": "SOC_Components/cpu_agent"},
        {"_from": "SOC_Components/memory_subsystem", "_to": "SOC_Components/memory_agent"},
        {"_from": "SOC_Components/peripheral_subsystem", "_to": "SOC_Components/peripheral_agent"},
        {"_from": "SOC_Components/noc_subsystem", "_to": "SOC_Components/noc_agent"},

        # Protocol-Level Connections
        {"_from": "SOC_Components/peripheral_agent", "_to": "SOC_Components/uart_driver"},
        {"_from": "SOC_Components/peripheral_agent", "_to": "SOC_Components/spi_driver"},
        {"_from": "SOC_Components/noc_agent", "_to": "SOC_Components/axi_driver"},
        {"_from": "SOC_Components/uart_driver", "_to": "SOC_Components/uart_monitor"},
        {"_from": "SOC_Components/spi_driver", "_to": "SOC_Components/spi_monitor"},
        {"_from": "SOC_Components/axi_driver", "_to": "SOC_Components/axi_monitor"},
        {"_from": "SOC_Components/axi_monitor", "_to": "SOC_Components/scoreboard"},
    ]

    # Insert Relationships
    for edge in soc_edges:
        edge_key = edge["_from"].split("/")[-1] + "_" + edge["_to"].split("/")[-1]
        if not db["Contains"].has(edge_key):
            edge["_key"] = edge_key
            db["Contains"].insert(edge)
            print(f"✅ Inserted Edge: {edge['_from']} → {edge['_to']}")

    print("🎯 **SoC-Level Hierarchy Setup Complete!** 🎯")
    return db, graph

def debug_query(query, db):
    """Debug AQL queries"""
    try:
        cursor = db.aql.execute(query)
        response_data = list(cursor)
        print("✅ Query Result:", response_data)
        return response_data
    except Exception as e:
        print("❌ ERROR:", str(e))
        return None

def check_missing_component(component, db):
    """Check if a required UVM component exists in the graph."""
    if not db["UVM_Components"].has(component):
        print(f"❌ ERROR: Missing Component - {component}")
    else:
        print(f"✅ Component Exists - {component}")

def analyze_graph_with_networkx():
    """Analyze the SoC graph using NetworkX."""
    db, _ = setup_database()
    G_nx = nx.DiGraph()

    for doc in db["SOC_Components"].all():
        G_nx.add_node(doc["_key"], name=doc["name"], level=doc["level"])

    for edge in db["Contains"].all():
        G_nx.add_edge(edge["_from"].split("/")[-1], edge["_to"].split("/")[-1])

    # Compute centrality for important nodes
    degree_centrality = nx.degree_centrality(G_nx)
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:5]

    # Check for cycles
    try:
        cycles = list(nx.find_cycle(G_nx, orientation="original"))
    except nx.NetworkXNoCycle:
        cycles = []

    return top_nodes, cycles

def check_uvm_code_with_graph_and_llm(uvm_code_snippet, openai_api_key):
    """Use graph analysis and LLM to debug UVM testbench code."""
    # Analyze with NetworkX
    top_nodes_nx, cycles = analyze_graph_with_networkx()

    # Construct the LLM debugging prompt
    prompt = f"""
    You are an expert in UVM-based SoC verification.
    The following testbench components are critical for debugging: {top_nodes_nx}.
    Detected cycles in the SoC hierarchy graph: {cycles}.

    Given this UVM testbench code:
    ```
    {uvm_code_snippet}
    ```

    Identify potential issues and provide debugging recommendations.
    """

    # Initialize the LLM with the provided API key
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key
    )

    # Query the LLM for code analysis
    response = llm.predict(prompt)
    return response

def visualize_graph_with_networkx():
    """Visualize the SoC hierarchy graph using NetworkX and Matplotlib."""
    G = nx.DiGraph()

    # Adding nodes to the graph
    nodes = [
        ("SoC Test", "SoC"),
        ("SoC Environment", "SoC"),
        ("SoC Config DB", "SoC"),
        ("CPU Subsystem", "Unit"),
        ("Memory Subsystem", "Unit"),
        ("Peripheral Subsystem", "Unit"),
        ("Network-on-Chip Subsystem", "Unit"),
        ("CPU Agent", "IP"),
        ("Memory Agent", "IP"),
        ("Peripheral Agent", "IP"),
        ("NoC Agent", "IP")
    ]
    
    for node, level in nodes:
        G.add_node(node, level=level)

    # Adding edges between nodes (relationships)
    edges = [
        ("SoC Test", "SoC Environment"),
        ("SoC Environment", "SoC Config DB"),
        ("SoC Environment", "CPU Subsystem"),
        ("SoC Environment", "Memory Subsystem"),
        ("SoC Environment", "Peripheral Subsystem"),
        ("SoC Environment", "Network-on-Chip Subsystem"),
        ("CPU Subsystem", "CPU Agent"),
        ("Memory Subsystem", "Memory Agent"),
        ("Peripheral Subsystem", "Peripheral Agent"),
        ("Network-on-Chip Subsystem", "NoC Agent")
    ]
    
    G.add_edges_from(edges)

    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos, with_labels=True, node_size=3000, node_color="lightblue", 
        font_size=8, font_weight="bold", edge_color="gray", width=2, 
        alpha=0.6, arrowsize=20
    )

    # Add title and show the plot
    plt.title("SoC Hierarchy Graph", fontsize=16)
    plt.tight_layout()
    plt.savefig("soc_hierarchy.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage
    uvm_test_code = """
    class soc_env extends uvm_env;
        `uvm_component_utils(soc_env)

        function new(string name, uvm_component parent);
            super.new(name, parent);
        endfunction

        virtual function void build_phase(uvm_phase phase);
            soc_config_db = soc_config_db::type_id::create("soc_config_db", this);
            cpu_subsystem = cpu_agent::type_id::create("cpu_agent", this);
        endfunction
    endclass
    """

    # Run graph analysis
    print("🔍 **NetworkX Analysis Results:**")
    top_nodes_nx, cycles = analyze_graph_with_networkx()
    print(f"Critical Nodes: {top_nodes_nx}")
    print(f"Detected Cycles: {cycles}")

    # Visualize the graph
    visualize_graph_with_networkx()
    
    print("\n✅ Project setup complete!")
