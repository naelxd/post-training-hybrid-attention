#!/usr/bin/env python3
"""
vLLM Throughput Benchmark Script
Запускает бенчмарк с конфигурациями input_len 4k-32k и output_len 128-2048
Использует --output-json для сохранения результатов
"""

import subprocess
import json
import time
import os
import tempfile
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
from tabulate import tabulate
import re
import threading
import queue

@dataclass
class BenchmarkConfig:
    """Конфигурация для бенчмарка"""
    model: str
    input_len: int
    output_len: int
    num_prompts: int
    tensor_parallel_size: int = 1
    dtype: str = "float16"
    runs: int = 3
    max_model_len: int = 65536  # Увеличиваем для поддержки длинных контекстов

class GPUMonitor:
    """Мониторинг GPU памяти в отдельном потоке"""
    
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_memory = 0
        self.running = False
        self.thread = None
        
    def start(self):
        """Запускает мониторинг в отдельном потоке"""
        self.running = True
        self.peak_memory = 0
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self) -> float:
        """Останавливает мониторинг и возвращает пиковое значение памяти в GB"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        return self.peak_memory
        
    def _monitor(self):
        """Мониторит использование памяти GPU через nvidia-smi"""
        while self.running:
            try:
                # Запускаем nvidia-smi для получения информации о памяти
                result = subprocess.run(
                    [
                        "nvidia-smi", 
                        "--query-gpu=memory.used", 
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    # Парсим вывод (может быть несколько GPU)
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                memory_mb = float(line.strip())
                                memory_gb = memory_mb / 1024
                                self.peak_memory = max(self.peak_memory, memory_gb)
                            except ValueError:
                                pass
                                
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                # Если nvidia-smi недоступен, пробуем использовать torch
                try:
                    import torch
                    if torch.cuda.is_available():
                        memory_gb = torch.cuda.memory_allocated() / 1024**3
                        self.peak_memory = max(self.peak_memory, memory_gb)
                except:
                    pass
                    
            time.sleep(self.interval)
    
class VLLMBenchmark:
    def __init__(self):
        self.results = []
        
    def run_benchmark(self, config: BenchmarkConfig) -> Optional[Dict[str, Any]]:
        """Запускает один прогон бенчмарка с сохранением результатов в JSON"""
        
        # Динамически регулируем количество промптов в зависимости от длины
        if config.input_len >= 32000:
            num_prompts = 8  # Для очень длинных контекстов меньше промптов
        elif config.input_len >= 16000:
            num_prompts = 16
        elif config.input_len >= 8000:
            num_prompts = 32
        else:
            num_prompts = 50
        
        # Создаем временный файл для JSON результатов
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json_output_path = tmp_file.name
        
        # Формируем команду для vLLM throughput bench
        cmd = [
            "vllm", "bench", "throughput",
            "--model", config.model,
            "--backend", "vllm",
            "--input-len", str(config.input_len),
            "--output-len", str(config.output_len),
            "--num-prompts", str(num_prompts),
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--dtype", config.dtype,
            "--trust-remote-code",
            "--max-model-len", str(config.max_model_len),
            "--seed", str(np.random.randint(0, 10000)),
            "--output-json", json_output_path  # Сохраняем результаты в JSON
        ]
        
        print(f"\n{'='*70}")
        print(f"🔄 ЗАПУСК КОНФИГУРАЦИИ:")
        print(f"   Model: {config.model}")
        print(f"   Input len: {config.input_len/1000:.0f}k ({config.input_len} токенов)")
        print(f"   Output len: {config.output_len} токенов")
        print(f"   Prompts: {num_prompts}")
        print(f"   Tensor parallel size: {config.tensor_parallel_size}")
        print(f"   JSON output: {json_output_path}")
        print(f"{'='*70}")
        
        # Запускаем мониторинг GPU памяти
        gpu_monitor = GPUMonitor(interval=0.5)
        gpu_monitor.start()
        
        try:
            start_time = time.time()
            
            # Запускаем процесс
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=False  # Не выкидываем исключение при ошибке
            )
            
            end_time = time.time()
            
            # Останавливаем мониторинг и получаем пиковую память
            peak_gpu_mem_gb = gpu_monitor.stop()
            
            # Проверяем на OOM в stderr
            oom_detected = False
            if result.stderr:
                oom_patterns = [
                    r'out of memory', r'oom', r'cuda out of memory',
                    r'CUDA_OUT_OF_MEMORY', r'allocator.*out of memory'
                ]
                for pattern in oom_patterns:
                    if re.search(pattern, result.stderr, re.IGNORECASE):
                        oom_detected = True
                        print(f"❌ Обнаружена OOM ошибка!")
                        break
            
            # Парсим JSON файл если он существует и нет OOM
            metrics = self.parse_json_output(json_output_path, oom_detected)
            
            # Добавляем мета-информацию
            metrics.update({
                'config': {
                    'model': config.model,
                    'input_len': config.input_len,
                    'output_len': config.output_len,
                    'num_prompts': num_prompts,
                    'tensor_parallel_size': config.tensor_parallel_size,
                    'dtype': config.dtype
                },
                'run_time': end_time - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'seed': cmd[cmd.index('--seed') + 1] if '--seed' in cmd else None,
                'oom': oom_detected,
                'stderr': result.stderr[:500] if result.stderr and oom_detected else None,
                'peak_gpu_mem_gb': peak_gpu_mem_gb  # <-- ЗДЕСЬ БЕРЕТСЯ ПАМЯТЬ!
            })
            
            # Если была OOM, обнуляем метрики throughput
            if oom_detected:
                metrics['decode_throughput'] = 0
                metrics['total_throughput'] = 0
                metrics['requests_per_second'] = 0
            
            print(f"\n✅ Результаты:")
            print(f"   decode tok/s: {metrics.get('decode_throughput', 0):.2f}")
            print(f"   total tok/s: {metrics.get('total_throughput', 0):.2f}")
            if not oom_detected:
                print(f"   requests/sec: {metrics.get('requests_per_second', 0):.2f}")
            print(f"   peak GPU mem: {metrics.get('peak_gpu_mem_gb', 0):.2f} GB")  # <-- ТЕПЕРЬ БУДЕТ РЕАЛЬНОЕ ЗНАЧЕНИЕ
            print(f"   время выполнения: {metrics['run_time']:.2f} сек")
            print(f"   OOM: {'Да' if oom_detected else 'Нет'}")
            
            # Удаляем временный файл
            try:
                os.unlink(json_output_path)
            except:
                pass
            
            return metrics
            
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
            gpu_monitor.stop()
            # Удаляем временный файл в случае ошибки
            try:
                os.unlink(json_output_path)
            except:
                pass
            return None
    
    def parse_json_output(self, json_path: str, oom_detected: bool) -> Dict[str, Any]:
        """Парсит JSON файл с результатами бенчмарка"""
        
        metrics = {
            'decode_throughput': 0,
            'total_throughput': 0,
            'requests_per_second': 0,
            'total_num_tokens': 0,
            'num_requests': 0,
            'elapsed_time': 0
        }
        
        if oom_detected:
            return metrics
        
        try:
            if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Извлекаем метрики из JSON
                metrics['elapsed_time'] = data.get('elapsed_time', 0)
                metrics['num_requests'] = data.get('num_requests', 0)
                metrics['total_num_tokens'] = data.get('total_num_tokens', 0)
                metrics['requests_per_second'] = data.get('requests_per_second', 0)
                
                # total_throughput = tokens_per_second
                metrics['total_throughput'] = data.get('tokens_per_second', 0)
                
                # Для decode throughput используем общий throughput
                # (vLLM не разделяет decode и prefill в JSON)
                metrics['decode_throughput'] = data.get('tokens_per_second', 0)
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Ошибка при парсинге JSON: {e}")
        
        return metrics
    
    def run_all_configs(self, configs: List[BenchmarkConfig]):
        """Запускает все конфигурации с несколькими прогонами"""
        
        total_configs = len(configs)
        for idx, config in enumerate(configs, 1):
            print(f"\n{'#'*80}")
            print(f"# КОНФИГУРАЦИЯ {idx}/{total_configs}")
            print(f"# {config.model} - {config.input_len/1000:.0f}k/{config.output_len}")
            print(f"{'#'*80}")
            
            config_results = []
            
            for run in range(config.runs):
                print(f"\n📊 Запуск {run + 1}/{config.runs}")
                result = self.run_benchmark(config)
                
                if result:
                    config_results.append(result)
                    self.results.append(result)
                
                # Пауза между запусками для очистки памяти
                if run < config.runs - 1:
                    print("⏳ Пауза 5 секунд перед следующим запуском...")
                    time.sleep(5)
            
            # Вычисляем и выводим медианные значения для этой конфигурации
            if config_results:
                self.print_median_results(config, config_results)
            
            # Пауза между разными конфигурациями
            if idx < total_configs:
                print("\n⏳ Пауза 10 секунд перед следующей конфигурацией...")
                time.sleep(10)
    
    def print_median_results(self, config: BenchmarkConfig, results: List[Dict]):
        """Выводит медианные результаты для конфигурации"""
        
        print(f"\n{'='*80}")
        print(f"📊 МЕДИАННЫЕ РЕЗУЛЬТАТЫ ДЛЯ {config.input_len/1000:.0f}k/{config.output_len}")
        print(f"{'='*80}")
        
        # Фильтруем успешные запуски (без OOM)
        successful_results = [r for r in results if not r.get('oom', False)]
        oom_count = len(results) - len(successful_results)
        
        if successful_results:
            # Собираем метрики только из успешных запусков
            decode_tokens = [r.get('decode_throughput', 0) for r in successful_results]
            total_tokens = [r.get('total_throughput', 0) for r in successful_results]
            requests_per_second = [r.get('requests_per_second', 0) for r in successful_results]
            gpu_memory = [r.get('peak_gpu_mem_gb', 0) for r in successful_results]  # <-- ДОБАВЛЕНО
            
            # Вычисляем медианы
            median_decode = np.median(decode_tokens)
            median_total = np.median(total_tokens)
            median_requests = np.median(requests_per_second)
            median_gpu_mem = np.median(gpu_memory)  # <-- ДОБАВЛЕНО
            
            print(f"\n✅ Decode tok/s:")
            print(f"   Медиана: {median_decode:.2f}")
            if len(decode_tokens) > 1:
                print(f"   Мин/Макс: {min(decode_tokens):.2f} / {max(decode_tokens):.2f}")
                print(f"   Стд откл: {np.std(decode_tokens):.2f}")
            
            print(f"\n✅ Total tok/s:")
            print(f"   Медиана: {median_total:.2f}")
            if len(total_tokens) > 1:
                print(f"   Мин/Макс: {min(total_tokens):.2f} / {max(total_tokens):.2f}")
            
            print(f"\n✅ Requests/sec:")
            print(f"   Медиана: {median_requests:.2f}")
            
            print(f"\n✅ Peak GPU Memory:")  # <-- ДОБАВЛЕНО
            print(f"   Медиана: {median_gpu_mem:.2f} GB")
            if len(gpu_memory) > 1:
                print(f"   Мин/Макс: {min(gpu_memory):.2f} GB / {max(gpu_memory):.2f} GB")
        else:
            print(f"\n❌ Нет успешных запусков (все OOM)")
        
        print(f"\n✅ OOM:")
        print(f"   Произошло: {oom_count}/{len(results)} запусков")
        
        print(f"{'='*80}\n")
    
    def save_results(self, filename: str = "vllm_benchmark_results.json"):
        """Сохраняет все результаты в JSON файл"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Полные результаты сохранены в {filename}")
    
    def print_summary_table(self):
        """Выводит сводную таблицу всех результатов (медианные значения)"""
        
        if not self.results:
            print("Нет результатов для отображения")
            return
        
        # Группируем результаты по конфигурациям
        config_groups = {}
        for r in self.results:
            config = r.get('config', {})
            key = (config.get('input_len'), config.get('output_len'), config.get('model'))
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(r)
        
        # Вычисляем медианы для каждой группы
        table_data = []
        for (input_len, output_len, model), results in sorted(config_groups.items()):
            # Фильтруем успешные запуски
            successful = [r for r in results if not r.get('oom', False)]
            oom_count = len(results) - len(successful)
            
            if successful:
                decode_tokens = [r.get('decode_throughput', 0) for r in successful]
                total_tokens = [r.get('total_throughput', 0) for r in successful]
                requests_sec = [r.get('requests_per_second', 0) for r in successful]
                gpu_memory = [r.get('peak_gpu_mem_gb', 0) for r in successful]  # <-- ДОБАВЛЕНО
                
                median_decode = np.median(decode_tokens)
                median_total = np.median(total_tokens)
                median_requests = np.median(requests_sec)
                median_gpu_mem = np.median(gpu_memory)  # <-- ДОБАВЛЕНО
                status = f"✅ {len(successful)}/{len(results)}"
            else:
                median_decode = 0
                median_total = 0
                median_requests = 0
                median_gpu_mem = 0  # <-- ДОБАВЛЕНО
                status = f"❌ OOM ({oom_count}/{len(results)})"
            
            # Короткое имя модели
            model_short = model.split('/')[-1] if '/' in model else model
            if len(model_short) > 15:
                model_short = model_short[:12] + "..."
            
            table_data.append([
                model_short,
                f"{input_len/1000:.0f}k",
                f"{output_len}",
                f"{median_decode:.1f}",
                f"{median_total:.1f}",
                f"{median_requests:.2f}",
                f"{median_gpu_mem:.2f}",  # <-- ДОБАВЛЕНО
                status
            ])
        
        # Сортируем по модели и длине входа
        table_data.sort(key=lambda x: (x[0], int(x[1].replace('k', ''))))
        
        headers = ["Model", "Input", "Output", "Decode tok/s", "Total tok/s", "Req/sec", "GPU Mem (GB)", "Status"]  # <-- ИЗМЕНЕНО
        print("\n" + "="*110)
        print("📊 СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (МЕДИАННЫЕ ЗНАЧЕНИЯ)")
        print("="*110)
        print(tabulate(table_data, headers=headers, tablefmt="grid", numalign="right"))
        
        # Сохраняем также в CSV
        df = pd.DataFrame(table_data, columns=headers)
        csv_filename = "vllm_benchmark_summary.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n💾 Сводная таблица сохранена в {csv_filename}")

def generate_configs(model_name: str, 
                    tensor_parallel_size: int = 1,
                    runs: int = 3) -> List[BenchmarkConfig]:
    """Генерирует все конфигурации из таблицы"""
    
    input_lengths = [4096, 8192, 16384, 32768]  # 4k, 8k, 16k, 32k
    output_lengths = [128, 512, 2048]
    
    configs = []
    for input_len in input_lengths:
        for output_len in output_lengths:
            configs.append(BenchmarkConfig(
                model=model_name,
                input_len=input_len,
                output_len=output_len,
                num_prompts=50,  # Будет скорректировано в run_benchmark
                tensor_parallel_size=tensor_parallel_size,
                runs=runs,
                max_model_len=input_len + output_len + 1024  # Немного запаса
            ))
    
    return configs

def main():
    """Основная функция"""
    
    print("="*110)
    print("🚀 VLLM THROUGHPUT BENCHMARK - LONG CONTEXT (JSON OUTPUT + GPU MONITORING)")
    print("="*110)
    print("\n📋 ТЕСТИРУЕМЫЕ КОНФИГУРАЦИИ:")
    print("   Input lengths: 4k, 8k, 16k, 32k токенов")
    print("   Output lengths: 128, 512, 2048 токенов")
    print("   Всего конфигураций: 12")
    print("   Запусков на конфигурацию: 3")
    print("   Результаты сохраняются в JSON и CSV")
    print("   Мониторинг GPU памяти через nvidia-smi")
    print("="*110)
    
    # Создаем экземпляр бенчмарка
    benchmark = VLLMBenchmark()
    
    # Генерируем конфигурации для Qwen/Qwen3-4B
    configs = generate_configs(
        model_name="Qwen/Qwen3-4B",
        tensor_parallel_size=1,
        runs=3
    )
    
    # Запускаем бенчмарк
    print("\n" + "="*110)
    print("🔥 ТЕСТИРОВАНИЕ Qwen/Qwen3-4B (1 GPU)")
    print("="*110)
    benchmark.run_all_configs(configs)
    
    # Сохраняем результаты
    benchmark.save_results()
    
    # Выводим сводную таблицу
    benchmark.print_summary_table()
    
    print("\n" + "="*110)
    print("✅ БЕНЧМАРК ЗАВЕРШЕН!")
    print("="*110)

if __name__ == "__main__":
    main()
