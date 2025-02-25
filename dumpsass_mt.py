#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class DisassemblerTool:
    def __init__(self, max_workers=None):
        """
        初始化反汇编工具
        
        参数:
            max_workers: 最大线程数，默认为 None (使用系统默认值，通常是 CPU 核心数 x 5)
        """
        self.max_workers = max_workers
        self.results = []
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.processed_files = 0
        self.total_files = 0
        self.start_time = 0
        self.errors = []
    
    def find_object_files(self, start_dir='.'):
        """
        遍历指定目录及其子目录，查找所有 .o 文件
        
        参数:
            start_dir: 起始目录，默认为当前目录
        
        返回:
            包含所有 .o 文件路径的列表
        """
        object_files = []
        for root, _, files in os.walk(start_dir):
            for file in files:
                if file.endswith('.o'):
                    object_files.append(os.path.join(root, file))
        
        return object_files
    
    def disassemble_object_file(self, obj_file, file_index):
        """
        使用 cuobjdump -sass 反汇编指定的对象文件
        
        参数:
            obj_file: 对象文件的路径
            file_index: 文件索引，用于排序结果
        
        返回:
            包含文件索引、文件路径和反汇编结果的元组
        """
        try:
            result = subprocess.run(
                ['cuobjdump', '-sass', obj_file],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout
            
            # 更新进度
            with self.progress_lock:
                self.processed_files += 1
                current = self.processed_files
                total = self.total_files
                elapsed = time.time() - self.start_time
                files_per_second = current / elapsed if elapsed > 0 else 0
                remaining = (total - current) / files_per_second if files_per_second > 0 else 0
                
                print(f"\r处理进度: [{current}/{total}] {current/total*100:.1f}% "
                      f"- {obj_file} - "
                      f"速度: {files_per_second:.1f}文件/秒 "
                      f"剩余时间: {remaining:.1f}秒", end='')
            
            return (file_index, obj_file, output)
        
        except subprocess.CalledProcessError as e:
            error_msg = f"错误: 反汇编 {obj_file} 失败\n{e.stderr}"
            with self.lock:
                self.errors.append((obj_file, error_msg))
            return (file_index, obj_file, error_msg)
        
        except Exception as e:
            error_msg = f"错误: 处理 {obj_file} 时发生异常\n{str(e)}"
            with self.lock:
                self.errors.append((obj_file, error_msg))
            return (file_index, obj_file, error_msg)
    
    def process_all_files(self, object_files):
        """
        使用线程池并行处理所有对象文件
        
        参数:
            object_files: 要处理的对象文件列表
        """
        self.total_files = len(object_files)
        self.processed_files = 0
        self.start_time = time.time()
        self.results = []
        self.errors = []
        
        print(f"使用 {self.max_workers or '自动选择的'} 个线程进行处理...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.disassemble_object_file, obj_file, i): (i, obj_file)
                for i, obj_file in enumerate(object_files)
            }
            
            # 收集结果
            for future in future_to_file:
                result = future.result()
                with self.lock:
                    self.results.append(result)
        
        # 按文件索引排序结果
        self.results.sort(key=lambda x: x[0])
        
        print("\n所有文件处理完成!")
    
    def write_results_to_file(self, output_file="allsass.txt"):
        """
        将处理结果写入输出文件
        
        参数:
            output_file: 输出文件名
        """
        print(f"正在将结果写入 {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write("CUDA 对象文件反汇编结果汇总\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文件数: {self.total_files}\n")
            f.write("=" * 80 + "\n\n")
            
            # 写入每个文件的反汇编结果
            for i, (_, obj_file, output) in enumerate(self.results, 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"文件 {i}/{self.total_files}: {obj_file}\n")
                f.write("=" * 80 + "\n\n")
                f.write(output)
                f.write("\n")
            
            # 如果有错误，写入错误摘要
            if self.errors:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("错误摘要\n")
                f.write("=" * 80 + "\n\n")
                
                for obj_file, error in self.errors:
                    f.write(f"文件: {obj_file}\n")
                    f.write(f"{error}\n\n")
        
        print(f"结果已保存到: {os.path.abspath(output_file)}")

def main():
    print("多线程 CUDA 对象文件反汇编工具")
    print("=" * 50)
    
    # 可选: 从命令行参数获取线程数
    import argparse
    parser = argparse.ArgumentParser(description='多线程 CUDA 对象文件反汇编工具')
    parser.add_argument('-j', '--jobs', type=int, default=None, 
                      help='并行线程数 (默认: CPU核心数)')
    parser.add_argument('-o', '--output', type=str, default='allsass.txt',
                      help='输出文件名 (默认: allsass.txt)')
    parser.add_argument('-d', '--directory', type=str, default='.',
                      help='起始目录 (默认: 当前目录)')
    args = parser.parse_args()
    
    # 初始化工具
    tool = DisassemblerTool(max_workers=args.jobs)
    
    # 查找所有 .o 文件
    print(f"正在查找 {args.directory} 目录中的 .o 文件...")
    object_files = tool.find_object_files(args.directory)
    total_files = len(object_files)
    
    if total_files == 0:
        print("未找到任何 .o 文件。")
        return
    
    print(f"找到 {total_files} 个对象文件。")
    
    # 记录开始时间
    start_time = time.time()
    
    # 处理所有文件
    tool.process_all_files(object_files)
    
    # 写入结果
    tool.write_results_to_file(args.output)
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    # 显示错误摘要
    if tool.errors:
        print(f"\n处理过程中发生了 {len(tool.errors)} 个错误:")
        for i, (obj_file, _) in enumerate(tool.errors, 1):
            print(f"  {i}. {obj_file}")
        print(f"详细错误信息已写入 {args.output}")

if __name__ == "__main__":
    main()

