#!/usr/bin/env python3
"""
Script to convert Jupyter Notebook (.ipynb) to HTML
Automatically generates HTML from the latest notebook version
"""

import os
import json
from pathlib import Path
from nbconvert import HTMLExporter
import nbformat

def convert_notebook_to_html(notebook_path, output_path=None):
    """
    Convert a Jupyter Notebook to HTML
    
    Args:
        notebook_path (str): Path to the .ipynb file
        output_path (str, optional): Path to save HTML. Defaults to same name with .html extension
    
    Returns:
        str: Path to the generated HTML file
    """
    
    # Validate input file exists
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    # Set output path
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.html')
    
    print(f"[*] Reading notebook: {notebook_path}")
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    print(f"[*] Converting notebook to HTML...")
    
    # Create the HTML exporter with custom configuration
    html_exporter = HTMLExporter(
        template_name='classic',  # Use classic template
        exclude_input=False,  # Include code cells
        exclude_output=False,  # Include output cells
    )
    
    # Export the notebook
    (body, resources) = html_exporter.from_notebook_node(notebook)
    
    # Save the HTML output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"[+] Successfully created: {output_path}")
    print(f"[+] File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return output_path

if __name__ == "__main__":
    # Convert the notebook
    notebook_file = "lichess_ml_analysis.ipynb"
    html_file = "lichess_ml_analysis.html"
    
    try:
        convert_notebook_to_html(notebook_file, html_file)
        print("\n[✓] Conversion completed successfully!")
    except Exception as e:
        print(f"\n[✗] Error during conversion: {str(e)}")
        exit(1)
