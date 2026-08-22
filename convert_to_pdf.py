#!/usr/bin/env python3
"""
Script để chuyển đổi Jupyter Notebook (.ipynb) sang PDF
Sử dụng nbconvert với LaTeX backend
"""

import subprocess
import os
import sys

def convert_ipynb_to_pdf(notebook_path, output_path):
    """
    Chuyển đổi file .ipynb sang PDF
    
    Args:
        notebook_path: Đường dẫn tới file .ipynb
        output_path: Đường dẫn output file .pdf
    """
    try:
        print(f"🔄 Đang chuyển đổi: {notebook_path} → {output_path}")
        print("⏳ Vui lòng chờ (quá trình này có thể mất 1-2 phút)...")
        
        # Chạy nbconvert command
        cmd = [
            "jupyter", "nbconvert",
            "--to", "pdf",
            "--output", output_path,
            notebook_path,
            "--template", "classic"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
                print(f"✅ Chuyển đổi thành công!")
                print(f"📁 Input:  {notebook_path}")
                print(f"📁 Output: {output_path}")
                print(f"📊 Kích thước PDF: {file_size:.2f} MB")
                return True
            else:
                print("❌ File PDF không được tạo")
                return False
        else:
            print(f"❌ Lỗi: {result.stderr}")
            print("\n💡 Gợi ý: Đảm bảo bạn đã cài đặt:")
            print("   - pip install nbconvert")
            print("   - pip install pandoc")
            print("   - pip install latex (hoặc texlive trên Linux/Mac)")
            return False
    
    except FileNotFoundError:
        print("❌ Không tìm thấy 'jupyter'. Vui lòng cài đặt:")
        print("   pip install jupyter nbconvert")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return False


def convert_with_python_api(notebook_path, output_path):
    """
    Phương pháp thay thế sử dụng nbconvert API trực tiếp
    """
    try:
        from nbconvert import PDFExporter
        import nbformat
        
        print(f"🔄 Đang chuyển đổi (Python API): {notebook_path} → {output_path}")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        pdf_exporter = PDFExporter()
        pdf_data, resources = pdf_exporter.from_notebook_node(notebook)
        
        with open(output_path, 'wb') as f:
            f.write(pdf_data)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Chuyển đổi thành công!")
            print(f"📁 Output: {output_path}")
            print(f"📊 Kích thước PDF: {file_size:.2f} MB")
            return True
        
    except Exception as e:
        print(f"❌ Lỗi (Python API): {str(e)}")
        return False


if __name__ == "__main__":
    notebook_file = "lichess_ml_analysis.ipynb"
    pdf_file = "lichess_ml_analysis.pdf"
    
    if not os.path.exists(notebook_file):
        print(f"❌ Không tìm thấy file: {notebook_file}")
        sys.exit(1)
    
    # Thử phương pháp 1: Command line
    success = convert_ipynb_to_pdf(notebook_file, pdf_file)
    
    # Nếu thất bại, thử phương pháp 2: Python API
    if not success:
        print("\n🔄 Thử phương pháp thay thế...")
        success = convert_with_python_api(notebook_file, pdf_file)
    
    sys.exit(0 if success else 1)
