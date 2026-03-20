import os
import json
from collections import defaultdict

def calculate_metrics_by_domain():
    files = os.listdir('results')
    
    # Структура для хранения результатов по доменам
    # domain_stats[domain][difficulty][length] = {'correct': 0, 'total': 0}
    domain_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})))
    
    # Структура для хранения общих результатов по всем данным
    overall_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    # overall_stats[difficulty][length] = {'correct': 0, 'total': 0}
    
    compensated = False
    
    for file in files:
        filename = os.path.join('results', file)
        try:
            pred_data = json.load(open(filename, encoding='utf-8'))
        except Exception as e:
            pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
        
        for pred in pred_data:
            domain = pred.get('domain', 'Unknown')
            difficulty = pred.get('difficulty', 'unknown')
            length = pred.get('length', 'unknown')
            
            acc = int(pred['judge'])
            if compensated and pred["pred"] is None:
                acc = 0.25
            
            # Сохраняем по доменам
            domain_stats[domain][difficulty][length]['correct'] += acc
            domain_stats[domain][difficulty][length]['total'] += 1
            
            # Сохраняем общую статистику
            overall_stats[difficulty][length]['correct'] += acc
            overall_stats[difficulty][length]['total'] += 1
    
    return domain_stats, overall_stats

def print_metrics(domain_stats, overall_stats):
    """Вывод метрик для каждого домена и общих метрик"""
    
    # Сначала выводим общие метрики по всем данным
    print("\n" + "="*80)
    print("ОБЩИЕ МЕТРИКИ (ALL DOMAINS)")
    print("="*80)
    
    # Общая статистика по всем данным
    total_correct = 0
    total_samples = 0
    
    # Выводим по сложности для всех данных
    print(f"\n{'Difficulty':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-"*50)
    
    for difficulty in ['easy', 'hard']:
        if difficulty in overall_stats:
            diff_stats = overall_stats[difficulty]
            diff_correct = sum(stat['correct'] for stat in diff_stats.values())
            diff_total = sum(stat['total'] for stat in diff_stats.values())
            
            if diff_total > 0:
                accuracy = (diff_correct / diff_total) * 100
                print(f"{difficulty:<12} {accuracy:>6.1f}%     {diff_correct:<10} {diff_total:<10}")
                
                total_correct += diff_correct
                total_samples += diff_total
    
    # Общая точность по всем данным
    if total_samples > 0:
        overall_accuracy = (total_correct / total_samples) * 100
        print(f"\n{'Overall':<12} {overall_accuracy:>6.1f}%     {total_correct:<10} {total_samples:<10}")
    
    # Выводим по длине для всех данных
    print(f"\n{'Length':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-"*50)
    
    length_order = ['short', 'medium', 'long']
    for length in length_order:
        length_correct = 0
        length_total = 0
        for difficulty in overall_stats.values():
            if length in difficulty:
                length_correct += difficulty[length]['correct']
                length_total += difficulty[length]['total']
        
        if length_total > 0:
            accuracy = (length_correct / length_total) * 100
            print(f"{length:<12} {accuracy:>6.1f}%     {length_correct:<10} {length_total:<10}")
    
    # Затем выводим метрики по доменам
    print("\n" + "="*80)
    print("МЕТРИКИ ПО ДОМЕНАМ")
    print("="*80)
    
    for domain in sorted(domain_stats.keys()):
        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain}")
        print(f"{'='*80}")
        
        # Собираем все данные по домену
        total_correct = 0
        total_samples = 0
        
        # Выводим по сложности
        print(f"\n{'Difficulty':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
        print("-"*50)
        
        for difficulty in ['easy', 'hard']:
            if difficulty in domain_stats[domain]:
                diff_stats = domain_stats[domain][difficulty]
                diff_correct = sum(stat['correct'] for stat in diff_stats.values())
                diff_total = sum(stat['total'] for stat in diff_stats.values())
                
                if diff_total > 0:
                    accuracy = (diff_correct / diff_total) * 100
                    print(f"{difficulty:<12} {accuracy:>6.1f}%     {diff_correct:<10} {diff_total:<10}")
                    
                    total_correct += diff_correct
                    total_samples += diff_total
        
        # Общая точность по домену
        if total_samples > 0:
            overall_accuracy = (total_correct / total_samples) * 100
            print(f"\n{'Overall':<12} {overall_accuracy:>6.1f}%     {total_correct:<10} {total_samples:<10}")
        
        # Выводим по длине
        print(f"\n{'Length':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
        print("-"*50)
        
        length_order = ['short', 'medium', 'long']
        for length in length_order:
            length_correct = 0
            length_total = 0
            for difficulty in domain_stats[domain].values():
                if length in difficulty:
                    length_correct += difficulty[length]['correct']
                    length_total += difficulty[length]['total']
            
            if length_total > 0:
                accuracy = (length_correct / length_total) * 100
                print(f"{length:<12} {accuracy:>6.1f}%     {length_correct:<10} {length_total:<10}")

def save_metrics_to_file(domain_stats, overall_stats, output_file='metrics_by_domain.txt'):
    """Сохранение метрик в файл"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Сохраняем общие метрики
        f.write("ОБЩИЕ МЕТРИКИ (ALL DOMAINS)\n")
        f.write("="*80 + "\n")
        
        total_correct = 0
        total_samples = 0
        
        # По сложности
        f.write(f"\n{'Difficulty':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
        f.write("-"*50 + "\n")
        
        for difficulty in ['easy', 'hard']:
            if difficulty in overall_stats:
                diff_stats = overall_stats[difficulty]
                diff_correct = sum(stat['correct'] for stat in diff_stats.values())
                diff_total = sum(stat['total'] for stat in diff_stats.values())
                
                if diff_total > 0:
                    accuracy = (diff_correct / diff_total) * 100
                    f.write(f"{difficulty:<12} {accuracy:>6.1f}%     {diff_correct:<10} {diff_total:<10}\n")
                    
                    total_correct += diff_correct
                    total_samples += diff_total
        
        # Общая точность
        if total_samples > 0:
            overall_accuracy = (total_correct / total_samples) * 100
            f.write(f"\n{'Overall':<12} {overall_accuracy:>6.1f}%     {total_correct:<10} {total_samples:<10}\n")
        
        # По длине
        f.write(f"\n{'Length':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
        f.write("-"*50 + "\n")
        
        length_order = ['short', 'medium', 'long']
        for length in length_order:
            length_correct = 0
            length_total = 0
            for difficulty in overall_stats.values():
                if length in difficulty:
                    length_correct += difficulty[length]['correct']
                    length_total += difficulty[length]['total']
            
            if length_total > 0:
                accuracy = (length_correct / length_total) * 100
                f.write(f"{length:<12} {accuracy:>6.1f}%     {length_correct:<10} {length_total:<10}\n")
        
        # Сохраняем метрики по доменам
        f.write("\n\n" + "МЕТРИКИ ПО ДОМЕНАМ\n")
        f.write("="*80 + "\n")
        
        for domain in sorted(domain_stats.keys()):
            f.write(f"\nDOMAIN: {domain}\n")
            f.write("="*80 + "\n")
            
            total_correct = 0
            total_samples = 0
            
            # По сложности
            f.write(f"\n{'Difficulty':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
            f.write("-"*50 + "\n")
            
            for difficulty in ['easy', 'hard']:
                if difficulty in domain_stats[domain]:
                    diff_stats = domain_stats[domain][difficulty]
                    diff_correct = sum(stat['correct'] for stat in diff_stats.values())
                    diff_total = sum(stat['total'] for stat in diff_stats.values())
                    
                    if diff_total > 0:
                        accuracy = (diff_correct / diff_total) * 100
                        f.write(f"{difficulty:<12} {accuracy:>6.1f}%     {diff_correct:<10} {diff_total:<10}\n")
                        
                        total_correct += diff_correct
                        total_samples += diff_total
            
            # Общая точность по домену
            if total_samples > 0:
                overall_accuracy = (total_correct / total_samples) * 100
                f.write(f"\n{'Overall':<12} {overall_accuracy:>6.1f}%     {total_correct:<10} {total_samples:<10}\n")
            
            # По длине
            f.write(f"\n{'Length':<12} {'Accuracy':<10} {'Correct':<10} {'Total':<10}\n")
            f.write("-"*50 + "\n")
            
            length_order = ['short', 'medium', 'long']
            for length in length_order:
                length_correct = 0
                length_total = 0
                for difficulty in domain_stats[domain].values():
                    if length in difficulty:
                        length_correct += difficulty[length]['correct']
                        length_total += difficulty[length]['total']
                
                if length_total > 0:
                    accuracy = (length_correct / length_total) * 100
                    f.write(f"{length:<12} {accuracy:>6.1f}%     {length_correct:<10} {length_total:<10}\n")

def create_summary_table(domain_stats, overall_stats):
    """Создание сводной таблицы по всем доменам и общей"""
    
    # Сначала выводим общую строку
    print("\n" + "="*100)
    print("СВОДНАЯ ТАБЛИЦА (ВКЛЮЧАЯ ОБЩУЮ МЕТРИКУ)")
    print("="*100)
    
    # Заголовок
    print(f"{'Domain':<25} {'Overall':<10} {'Easy':<10} {'Hard':<10} {'Short':<10} {'Medium':<10} {'Long':<10}")
    print("-"*100)
    
    # Вычисляем общую метрику по всем данным
    total_correct_all = 0
    total_samples_all = 0
    easy_correct_all = 0
    easy_total_all = 0
    hard_correct_all = 0
    hard_total_all = 0
    short_correct_all = 0
    short_total_all = 0
    medium_correct_all = 0
    medium_total_all = 0
    long_correct_all = 0
    long_total_all = 0
    
    for difficulty in overall_stats:
        for length in overall_stats[difficulty]:
            correct = overall_stats[difficulty][length]['correct']
            total = overall_stats[difficulty][length]['total']
            
            total_correct_all += correct
            total_samples_all += total
            
            if difficulty == 'easy':
                easy_correct_all += correct
                easy_total_all += total
            else:
                hard_correct_all += correct
                hard_total_all += total
            
            if length == 'short':
                short_correct_all += correct
                short_total_all += total
            elif length == 'medium':
                medium_correct_all += correct
                medium_total_all += total
            else:
                long_correct_all += correct
                long_total_all += total
    
    # Выводим общую строку
    overall_all = (total_correct_all / total_samples_all * 100) if total_samples_all > 0 else 0
    easy_all = (easy_correct_all / easy_total_all * 100) if easy_total_all > 0 else 0
    hard_all = (hard_correct_all / hard_total_all * 100) if hard_total_all > 0 else 0
    short_all = (short_correct_all / short_total_all * 100) if short_total_all > 0 else 0
    medium_all = (medium_correct_all / medium_total_all * 100) if medium_total_all > 0 else 0
    long_all = (long_correct_all / long_total_all * 100) if long_total_all > 0 else 0
    
    print(f"{'ALL DOMAINS':<25} {overall_all:>6.1f}%   {easy_all:>6.1f}%   {hard_all:>6.1f}%   {short_all:>6.1f}%   {medium_all:>6.1f}%   {long_all:>6.1f}%")
    print("-"*100)
    
    # Выводим по каждому домену
    for domain in sorted(domain_stats.keys()):
        # Общая статистика по домену
        total_correct = 0
        total_samples = 0
        
        # По сложности
        easy_correct = 0
        easy_total = 0
        hard_correct = 0
        hard_total = 0
        
        # По длине
        short_correct = 0
        short_total = 0
        medium_correct = 0
        medium_total = 0
        long_correct = 0
        long_total = 0
        
        for difficulty in domain_stats[domain]:
            for length in domain_stats[domain][difficulty]:
                correct = domain_stats[domain][difficulty][length]['correct']
                total = domain_stats[domain][difficulty][length]['total']
                
                total_correct += correct
                total_samples += total
                
                # По сложности
                if difficulty == 'easy':
                    easy_correct += correct
                    easy_total += total
                else:
                    hard_correct += correct
                    hard_total += total
                
                # По длине
                if length == 'short':
                    short_correct += correct
                    short_total += total
                elif length == 'medium':
                    medium_correct += correct
                    medium_total += total
                else:
                    long_correct += correct
                    long_total += total
        
        # Рассчитываем проценты
        overall = (total_correct / total_samples * 100) if total_samples > 0 else 0
        easy_acc = (easy_correct / easy_total * 100) if easy_total > 0 else 0
        hard_acc = (hard_correct / hard_total * 100) if hard_total > 0 else 0
        short_acc = (short_correct / short_total * 100) if short_total > 0 else 0
        medium_acc = (medium_correct / medium_total * 100) if medium_total > 0 else 0
        long_acc = (long_correct / long_total * 100) if long_total > 0 else 0
        
        print(f"{domain:<25} {overall:>6.1f}%   {easy_acc:>6.1f}%   {hard_acc:>6.1f}%   {short_acc:>6.1f}%   {medium_acc:>6.1f}%   {long_acc:>6.1f}%")

if __name__ == "__main__":
    # Анализируем данные
    stats, overall = calculate_metrics_by_domain()
    
    # Выводим детальные метрики (включая общие)
    print_metrics(stats, overall)
    
    # Создаем сводную таблицу (включая общую метрику)
    create_summary_table(stats, overall)
    
    # Сохраняем в файл
    save_metrics_to_file(stats, overall)
    print(f"\n\nРезультаты сохранены в файл 'metrics_by_domain.txt'")
