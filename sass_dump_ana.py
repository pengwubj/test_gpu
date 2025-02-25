# sass_analyzer.py
import os
import re
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SassAnalyzer:
    def __init__(self, output_dir: str = "sass_ana_res"):
        """初始化SASS分析器"""
        self.output_dir = Path(output_dir)
        self.db_path = self.output_dir / "sass_analysis.db"
        self.files_with_sass: List[str] = []
        self.files_without_sass: List[str] = []
        self.total_instructions: int = 0
        self.total_functions: int = 0
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """初始化SQLite数据库结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建所需的表
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    total_instructions INTEGER,
                    has_sass BOOLEAN
                );

                CREATE TABLE IF NOT EXISTS functions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    function_name TEXT,
                    mangled_name TEXT,
                    instruction_count INTEGER,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                );

                CREATE TABLE IF NOT EXISTS instructions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_id INTEGER,
                    instruction TEXT,
                    count INTEGER,
                    percentage REAL,
                    FOREIGN KEY (function_id) REFERENCES functions (id)
                );
            """)

    def find_object_files(self) -> List[str]:
        """查找当前目录及子目录中的所有.o文件"""
        return [str(p) for p in Path('.').rglob('*.o')]

    def _demangle_name(self, mangled_name: str) -> str:
        """使用c++filt解码C++符号名"""
        try:
            result = subprocess.run(
                ['c++filt', mangled_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return mangled_name

    def _get_sass_instructions(self, obj_file: str) -> str:
        """使用cuobjdump获取SASS指令"""
        try:
            result = subprocess.run(
                ['cuobjdump', '-sass', obj_file],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def normalize_instruction(self, instruction):
        """
        标准化指令，移除谓词部分
        例如: @P0 FFMA.FTZ.SAT R0, R0, R13, RZ -> FFMA.FTZ.SAT R0, R0, R13, RZ
        """
        # 匹配以 @P 开头的谓词部分
        predicate_pattern = r'^@P\d+\s+'
        return re.sub(predicate_pattern, '', instruction)

    def _parse_sass_output(self, sass_output: str) -> Dict:
        """解析SASS输出，提取函数和指令信息"""
        functions = {}
        current_function = None
        
        func_pattern = re.compile(r'Function : (.+)')
        #instr_pattern = re.compile(r'\/\*\w+\*\/\s+([A-Z0-9\.]+)')
        instr_pattern = re.compile(r'\/\*\w+\*\/\s+(?:@P\d+\s+)?([A-Z][A-Z0-9\.]+(?:\.[A-Z][A-Z0-9\.]+)*)')
        
        for line in sass_output.splitlines():
            if func_match := func_pattern.search(line):
                mangled_name = func_match.group(1).strip()
                demangled_name = self._demangle_name(mangled_name)
                current_function = {
                    'mangled_name': mangled_name,
                    'demangled_name': demangled_name,
                    'instructions': []
                }
                functions[mangled_name] = current_function
                self.total_functions += 1
                continue
                
            if current_function and '//' not in line:
                if instr_match := instr_pattern.search(line):
                    instruction = instr_match.group(1)
                    current_function['instructions'].append(instruction)
                    self.total_instructions += 1
        
        return functions

    def _save_to_database(self, obj_file: str, functions: Dict, file_total_instructions: int) -> None:
        """将分析结果保存到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 保存文件信息
            cursor.execute(
                "INSERT INTO files (file_path, total_instructions, has_sass) VALUES (?, ?, 1)",
                (obj_file, file_total_instructions)
            )
            file_id = cursor.lastrowid
            
            # 保存函数和指令信息
            for mangled_name, func_data in functions.items():
                instruction_counts = Counter(func_data['instructions'])
                
                cursor.execute(
                    """INSERT INTO functions 
                       (file_id, function_name, mangled_name, instruction_count) 
                       VALUES (?, ?, ?, ?)""",
                    (file_id, func_data['demangled_name'], mangled_name, 
                     len(func_data['instructions']))
                )
                function_id = cursor.lastrowid
                
                # 保存指令统计
                for instruction, count in instruction_counts.items():
                    percentage = count / len(func_data['instructions']) * 100
                    cursor.execute(
                        """INSERT INTO instructions 
                           (function_id, instruction, count, percentage) 
                           VALUES (?, ?, ?, ?)""",
                        (function_id, instruction, count, percentage)
                    )

    def analyze_file(self, obj_file: str) -> Optional[Dict]:
        """分析单个对象文件的SASS指令"""
        sass_output = self._get_sass_instructions(obj_file)
        
        # 检查是否包含SASS代码
        if "Function : " not in sass_output:
            self.files_without_sass.append(obj_file)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO files (file_path, total_instructions, has_sass) VALUES (?, 0, 0)",
                    (obj_file,)
                )
            return None
        
        self.files_with_sass.append(obj_file)
        functions = self._parse_sass_output(sass_output)
        file_total_instructions = sum(len(func['instructions']) for func in functions.values())
        
        # 保存到数据库
        self._save_to_database(obj_file, functions, file_total_instructions)
        
        # 生成文件报告
        self._generate_file_report(obj_file, functions, file_total_instructions)
        
        return {
            'file_path': obj_file,
            'total_instructions': file_total_instructions,
            'functions': functions
        }

    def _generate_file_report(self, obj_file: str, functions: Dict, total_instructions: int) -> None:
        """生成单个文件的分析报告"""
        report_path = self.output_dir / f"{obj_file.replace('/', '_').replace('\\', '_').lstrip('._')}.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"SASS Analysis Report for {obj_file}\n{'='*80}\n\n")
            f.write(f"Total instructions: {total_instructions}\n\n")
            
            # 按函数输出统计信息
            for mangled_name, func_data in functions.items():
                instruction_counts = Counter(func_data['instructions'])
                f.write(f"Function: {func_data['demangled_name']}\n")
                f.write(f"Original symbol: {mangled_name}\n")
                f.write(f"Instructions: {len(func_data['instructions'])}\n\n")
                
                # 输出指令统计
                f.write(f"{'Instruction':<20} {'Count':<10} {'Percentage':<10}\n{'-'*40}\n")
                for instr, count in instruction_counts.most_common():
                    percentage = count / len(func_data['instructions']) * 100
                    f.write(f"{instr:<20} {count:<10} {percentage:6.2f}%\n")
                f.write("\n")

    def generate_summary_report(self) -> None:
        """生成总体汇总报告"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 获取统计数据
            cursor.execute("""
                SELECT instruction, SUM(count) as total_count
                FROM instructions
                GROUP BY instruction
                ORDER BY total_count DESC
                LIMIT 20
            """)
            top_instructions = cursor.fetchall()
            
            # 获取最多指令的文件
            cursor.execute("""
                SELECT file_path, total_instructions
                FROM files
                WHERE has_sass = 1
                ORDER BY total_instructions DESC
                LIMIT 5
            """)
            top_files = cursor.fetchall()
            
            # 获取最多指令的函数
            cursor.execute("""
                SELECT f.function_name, f.instruction_count, fl.file_path
                FROM functions f
                JOIN files fl ON f.file_id = fl.id
                ORDER BY f.instruction_count DESC
                LIMIT 10
            """)
            top_functions = cursor.fetchall()
        
        # 生成汇总报告
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write("CUDA SASS Analysis Summary Report\n")
            f.write(f"{'='*80}\n\n")
            
            # 基本统计信息
            f.write(f"Total files analyzed: {len(self.files_with_sass) + len(self.files_without_sass)}\n")
            f.write(f"Files with SASS code: {len(self.files_with_sass)}\n")
            f.write(f"Files without SASS code: {len(self.files_without_sass)}\n")
            f.write(f"Total functions analyzed: {self.total_functions}\n")
            f.write(f"Total instructions analyzed: {self.total_instructions}\n\n")
            
            # 输出详细统计信息
            self._write_summary_sections(f, top_instructions, top_files, top_functions)

    def _write_summary_sections(self, f, top_instructions, top_files, top_functions):
        """写入汇总报告的各个部分"""
        # 没有SASS代码的文件列表
        if self.files_without_sass:
            f.write("Files without SASS code:\n")
            f.write(f"{'-'*80}\n")
            for file in self.files_without_sass:
                f.write(f"- {file}\n")
            f.write("\n")
        
        # 最常见指令
        f.write("Top 20 most common instructions:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Instruction':<20} {'Count':<10}\n")
        f.write(f"{'-'*30}\n")
        for instr, count in top_instructions:
            f.write(f"{instr:<20} {count:<10}\n")
        f.write("\n")
        
        # 最多指令的文件
        f.write("Top 5 files with most instructions:\n")
        f.write(f"{'-'*80}\n")
        for file_path, count in top_files:
            f.write(f"{file_path:<60} {count:>10}\n")
        f.write("\n")
        
        # 最多指令的函数
        f.write("Top 10 functions with most instructions:\n")
        f.write(f"{'-'*80}\n")
        for func_name, count, file_path in top_functions:
            f.write(f"{func_name:<50} {count:>10} {file_path}\n")

    def run(self) -> None:
        """运行分析工具的主函数"""
        print("CUDA SASS Instruction Analyzer")
        print("=" * 50)
        
        object_files = self.find_object_files()
        print(f"Found {len(object_files)} object files to analyze.")
        
        for i, obj_file in enumerate(object_files, 1):
            print(f"[{i}/{len(object_files)}] Analyzing {obj_file}...")
            self.analyze_file(obj_file)
        
        print("Generating summary report...")
        self.generate_summary_report()
        
        print("\nAnalysis complete!")
        print(f"Results saved in: {self.output_dir}")


if __name__ == "__main__":
    analyzer = SassAnalyzer()
    analyzer.run()
