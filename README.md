# UVM Analysis System

AI-powered UVM (Universal Verification Methodology) analysis system with graph database integration and GPU acceleration for SoC verification.

## Features

- **Graph Database Integration**: Uses ArangoDB to model SoC hierarchies
- **AI-Powered Debugging**: LangChain + OpenAI for intelligent UVM code analysis
- **GPU Acceleration**: Optional cuGraph support for large-scale analysis
- **Network Analysis**: NetworkX for graph analysis and visualization
- **PDF Processing**: Extract UVM documentation and debug information

## Architecture

```
SoC Level
├── SoC Test
├── SoC Environment  
└── SoC Config DB

Unit Level (Subsystems)
├── CPU Subsystem → CPU Agent
├── Memory Subsystem → Memory Agent
├── Peripheral Subsystem → Peripheral Agent
└── Network-on-Chip Subsystem → NoC Agent

IP Level (Protocol Components)
├── UART Driver/Monitor
├── SPI Driver/Monitor
├── AXI Driver/Monitor
└── Scoreboard
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration:
```bash
pip install --extra-index-url=https://pypi.nvidia.com cuGraph-cu12 cudf-cu12
```

## Configuration

Set your OpenAI API key and ArangoDB credentials in `main.py`:

```python
ARANGO_HOST = "your-arangodb-host"
USERNAME = "your-username"  
PASSWORD = "your-password"
openai_api_key = "your-openai-key"
```

## Usage

```python
from main import *

# Setup database
db, graph = setup_database()

# Analyze UVM code
uvm_code = """
class soc_env extends uvm_env;
    // Your UVM code here
endclass
"""

# Get AI-powered debugging recommendations
debug_report = check_uvm_code_with_graph_and_llm(uvm_code, openai_api_key)
print(debug_report)

# Visualize SoC hierarchy
visualize_graph_with_networkx()
```

## Original Source

Adapted from Google Colab notebook:
https://colab.research.google.com/drive/1BMkryTiTdh1jfGf675imgBnEc-LgXp1W

## File Structure

```
uvm_analysis_project/
├── main.py              # Core analysis system
├── requirements.txt     # Python dependencies  
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── data/               # Sample UVM files (optional)
```
