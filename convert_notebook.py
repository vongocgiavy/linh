#!/usr/bin/env python3
"""
Script để chuyển đổi Jupyter Notebook (.ipynb) sang HTML
Sử dụng nbconvert API programmatically
"""

from nbconvert import HTMLExporter
import nbformat
import os

def convert_ipynb_to_html(notebook_path, output_path):
    """
    Chuyển đổi file .ipynb sang HTML
    
    Args:
        notebook_path: Đường dẫn tới file .ipynb
        output_path: Đường dẫn output file .html
    """
    try:
        # Load notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Tạo HTMLExporter
        html_exporter = HTMLExporter(
            template_name='classic',
            exclude_input_prompt=False,
            exclude_output_prompt=False
        )
        
        # Export to HTML
        body, resources = html_exporter.from_notebook_node(notebook)
        
        # Lưu file HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        
        print(f"✅ Chuyển đổi thành công!")
        print(f"📁 Input:  {notebook_path}")
        print(f"📁 Output: {output_path}")
        print(f"📊 Kích thước HTML: {len(body) / 1024:.2f} KB")
        
        return True
    
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return False


if __name__ == "__main__":
    # Chuyển đổi lichess_ml_analysis.ipynb sang HTML
    notebook_file = "lichess_ml_analysis.ipynb"
    html_file = "lichess_ml_analysis.html"
    
    if os.path.exists(notebook_file):
        convert_ipynb_to_html(notebook_file, html_file)
    else:
        print(f"❌ Không tìm thấy file: {notebook_file}")
