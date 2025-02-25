#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sqlite3
import argparse
from collections import defaultdict
from tabulate import tabulate  # 需要安装: pip install tabulate


class SassQueryTool:
    def __init__(self, db_path="sass_ana_res/sass_analysis.db"):
        """初始化查询工具"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
    
    def find_instruction_variants(self, prefix):
        """查找指定前缀的指令变种"""
        cursor = self.conn.cursor()
        
        # 查询指令变种及其使用频率
        query = """
        SELECT instruction, SUM(count) as total_count
        FROM instructions
        WHERE instruction LIKE ?
        GROUP BY instruction
        ORDER BY total_count DESC
        """
        
        cursor.execute(query, (f"{prefix}%",))
        variants = cursor.fetchall()
        
        return variants
    
    def find_functions_using_instruction(self, instruction_pattern):
        """查找使用指定指令的函数"""
        cursor = self.conn.cursor()
        
        # 查询使用该指令的函数及文件信息
        query = """
        SELECT 
            f.function_name, 
            f.mangled_name, 
            i.instruction,
            i.count,
            i.percentage,
            files.file_path
        FROM instructions i
        JOIN functions f ON i.function_id = f.id
        JOIN files ON f.file_id = files.id
        WHERE i.instruction LIKE ?
        ORDER BY i.count DESC
        """
        
        cursor.execute(query, (instruction_pattern,))
        functions = cursor.fetchall()
        
        return functions
    
    def find_instruction_usage_by_file(self, instruction_pattern):
        """按文件统计指令使用情况"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT 
            files.file_path,
            SUM(i.count) as total_count
        FROM instructions i
        JOIN functions f ON i.function_id = f.id
        JOIN files ON f.file_id = files.id
        WHERE i.instruction LIKE ?
        GROUP BY files.id
        ORDER BY total_count DESC
        """
        
        cursor.execute(query, (instruction_pattern,))
        file_stats = cursor.fetchall()
        
        return file_stats
    
    def group_functions_by_instruction_variant(self, prefix):
        """按指令变种对函数进行分组"""
        variants = self.find_instruction_variants(prefix)
        result = {}
        
        for variant in variants:
            instruction = variant['instruction']
            functions = self.find_functions_using_instruction(instruction)
            result[instruction] = {
                'count': variant['total_count'],
                'functions': functions
            }
        
        return result
    
    def analyze_instruction_distribution(self, prefix):
        """分析指令分布情况"""
        # 获取指令变种
        variants = self.find_instruction_variants(prefix)
        
        if not variants:
            print(f"未找到前缀为 '{prefix}' 的指令")
            return
        
        # 输出指令变种统计
        print(f"\n=== {prefix} 指令变种统计 ===")
        variant_data = [(v['instruction'], v['total_count']) for v in variants]
        print(tabulate(variant_data, headers=['指令变种', '使用次数'], tablefmt='grid'))
        
        # 对每个变种，查找使用最多的函数
        print(f"\n=== 各 {prefix} 指令变种的主要使用函数 ===")
        for variant in variants:
            instruction = variant['instruction']
            functions = self.find_functions_using_instruction(instruction)
            
            if functions:
                print(f"\n>> {instruction} (总使用次数: {variant['total_count']})")
                
                # 只显示前5个使用最多的函数
                top_functions = functions[:5]
                func_data = []
                for func in top_functions:
                    func_name = func['function_name']
                    # 如果函数名太长，截断显示
                    if len(func_name) > 40:
                        func_name = func_name[:37] + "..."
                    
                    file_path = func['file_path']
                    # 如果路径太长，只显示最后部分
                    if len(file_path) > 30:
                        file_path = "..." + file_path[-27:]
                    
                    func_data.append([
                        func_name,
                        func['count'],
                        f"{func['percentage']:.2f}%",
                        file_path
                    ])
                
                print(tabulate(func_data, 
                               headers=['函数名', '使用次数', '函数内占比', '文件'],
                               tablefmt='grid'))
                
                if len(functions) > 5:
                    print(f"... 以及其他 {len(functions) - 5} 个函数")
        
        # 查找使用该前缀指令最多的文件
        file_stats = self.find_instruction_usage_by_file(f"{prefix}%")
        
        if file_stats:
            print(f"\n=== 使用 {prefix} 指令最多的文件 (前10) ===")
            file_data = [(stats['file_path'], stats['total_count']) for stats in file_stats[:10]]
            print(tabulate(file_data, headers=['文件路径', '使用次数'], tablefmt='grid'))
    
    def search_by_pattern(self, pattern):
        """根据模式搜索指令"""
        cursor = self.conn.cursor()
        
        # 查询匹配模式的指令
        query = """
        SELECT DISTINCT instruction
        FROM instructions
        WHERE instruction LIKE ?
        ORDER BY instruction
        """
        
        cursor.execute(query, (f"%{pattern}%",))
        instructions = cursor.fetchall()
        
        if not instructions:
            print(f"未找到包含 '{pattern}' 的指令")
            return
        
        print(f"\n=== 包含 '{pattern}' 的指令 ===")
        for idx, instr in enumerate(instructions, 1):
            print(f"{idx}. {instr['instruction']}")
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description='CUDA SASS 指令数据库查询工具')
    parser.add_argument('--db', default='sass_ana_res/sass_analysis.db',
                        help='SASS 分析数据库路径 (默认: sass_ana_res/sass_analysis.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='查询命令')
    
    # 查找指令变种命令
    variant_parser = subparsers.add_parser('variant', help='查找指令变种')
    variant_parser.add_argument('prefix', help='指令前缀 (如 LD, MOV, ST 等)')
    
    # 搜索指令模式命令
    search_parser = subparsers.add_parser('search', help='搜索指令模式')
    search_parser.add_argument('pattern', help='搜索模式 (如 SYS, CI 等)')
    
    args = parser.parse_args()
    
    try:
        query_tool = SassQueryTool(args.db)
        
        if args.command == 'variant':
            query_tool.analyze_instruction_distribution(args.prefix)
        elif args.command == 'search':
            query_tool.search_by_pattern(args.pattern)
        else:
            parser.print_help()
        
        query_tool.close()
    
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()

