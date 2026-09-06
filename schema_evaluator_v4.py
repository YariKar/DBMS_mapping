import os
import sys
import pandas as pd

# Add current directory and scratch to sys.path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schema_benchmark_loader_v5 import MIMICtoOMOPLoader
from schema_mapping_v5 import StructuralMetadataUnifier, AdvancedHybridMatcher, DynamicVisualizer, RobustSentenceTransformerFallback

class DualLogger:
    """
    Класс для дублирования потока вывода (stdout) как в консоль, так и в лог-файл.
    Позволяет прозрачно вести лог без изменения существующих вызовов print().
    """
    def __init__(self, filename="evaluation_run.log"):
        self.terminal = sys.stdout
        # Открываем файл в режиме добавления (append) с кодировкой utf-8
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def evaluate_predictions(predicted_df, ground_truth):
    """
    Математически строгая оценка качества выравнивания схем.
    Рассчитывает Precision, Recall и F1-Score на основе Ground Truth бенчмарка.
    """
    # Нормализуем Ground Truth для точного посимвольного сопоставления
    gt_set = set()
    for src, tgt in ground_truth:
        # Приводим к нижнему регистру и очищаем от пробелов
        src_norm = str(src).strip().lower()
        tgt_norm = str(tgt).strip().lower()
        # Сохраняем в виде неупорядоченного множества (чтобы не зависеть от порядка A <-> B)
        gt_set.add(frozenset([src_norm, tgt_norm]))

    if predicted_df.empty:
        return 0.0, 0.0, 0.0, [], [], list(ground_truth)

    tp_list = []
    fp_list = []
    predicted_set = set()

    # Оцениваем каждое предсказание
    for _, row in predicted_df.iterrows():
        a_loc = str(row['A_Loc']).strip().lower()
        b_loc = str(row['B_Loc']).strip().lower()
        pair_frozenset = frozenset([a_loc, b_loc])
        
        # Защита от дубликатов в предсказаниях
        if pair_frozenset in predicted_set:
            continue
        predicted_set.add(pair_frozenset)

        # Проверяем, есть ли эта пара в эталонной разметке
        if pair_frozenset in gt_set:
            tp_list.append((row['A_Loc'], row['B_Loc'], row['Score']))
        else:
            fp_list.append((row['A_Loc'], row['B_Loc'], row['Score']))

    tp = len(tp_list)
    fp = len(fp_list)
    
    # Находим пропущенные эталонные связи (False Negatives)
    fn_list = []
    for src, tgt in ground_truth:
        src_norm = str(src).strip().lower()
        tgt_norm = str(tgt).strip().lower()
        gt_frozenset = frozenset([src_norm, tgt_norm])
        if gt_frozenset not in predicted_set:
            fn_list.append((src, tgt))
            
    fn = len(fn_list)

    # Расчет метрик по ГОСТ-совместимым формулам
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score, tp_list, fp_list, fn_list


def main():
    # Инициализация дублирования вывода в файл лога
    log_file_path = "./evaluation_run.log"
    sys.stdout = DualLogger(log_file_path)

    print("=" * 75)
    print("НИР3: СКВОЗНАЯ ВЕРИФИКАЦИЯ СЕМАНТИКО-СТРУКТУРНОГО ВЫРАВНИВАНИЯ СХЕМ")
    print("=" * 75)

    # 1. Загрузка реального/псевдореального медицинского бенчмарка MIMIC-to-OMOP
    print("\nШаг 1. Загрузка бенчмарка MIMIC-to-OMOP...")
    loader = MIMICtoOMOPLoader()
    raw_schemas = loader.load_schemas()
    ground_truth = loader.load_ground_truth()

    print(f"[Загружено] Элементов схемы: {len(raw_schemas)}")
    print(f"[Загружено] Эталонных связей (Ground Truth): {len(ground_truth)}")

    # 2. Унификация и структурное обогащение метаданных (контекстуализация)
    print("\nШаг 2. Унификация метаданных и автогенерация пропущенных описаний...")
    unifier = StructuralMetadataUnifier()
    unified_df = unifier.unify(raw_schemas)
    print(f"[Выполнено] Обогащено элементов схем: {len(unified_df)}")

    # 3. Инициализация и запуск AdvancedHybridMatcher
    print("\nШаг 3. Запуск семантико-структурного сопоставления пороговым методом...")
    matcher = AdvancedHybridMatcher()
    
    # Адаптивный порог на основе используемого векторизатора
    if isinstance(matcher.embedder, RobustSentenceTransformerFallback):
        # TF-IDF синтаксически более жесткий, снижаем порог для полноты охвата
        threshold = 0.35
    else:
        # SBERT улавливает глубокую семантику, используем строгий порог 0.55 из НИР2
        threshold = 0.55
        
    print(f"[Конфигурация] Порог косинусного сходства (Threshold): {threshold}")
    matches_df = matcher.find_matches(unified_df, threshold=threshold)
    print(f"[Выполнено] Алгоритм выявил потенциальных связей: {len(matches_df)}")

    # 4. Численный академический расчет метрик качества
    print("\nШаг 4. Расчет научных метрик выравнивания...")
    precision, recall, f1_score, tp_list, fp_list, fn_list = evaluate_predictions(matches_df, ground_truth)

    print("-" * 75)
    print(f"МЕТРИКИ КАЧЕСТВА АЛГОРИТМА (MIMIC-to-OMOP):")
    print(f"  Точность (Precision): {precision:.4f} (доля истинных связей среди предсказанных)")
    print(f"  Полнота (Recall):     {recall:.4f} (доля найденных связей из золотого стандарта)")
    print(f"  F1-мера (F1-Score):   {f1_score:.4f} (гармоническое среднее Precision и Recall)")
    print("-" * 75)

    print(f"\n[Подробно] Успешно найденные эталонные связи (True Positives) - [{len(tp_list)}]:")
    for src, tgt, score in tp_list:
        print(f"  ✔ {src} <=======> {tgt} (Score: {score:.4f})")

    if fp_list:
        print(f"\n[Подробно] Ложноположительные связи (False Positives) - [{len(fp_list)}]:")
        for src, tgt, score in fp_list:
            print(f"  ✖ {src} <=======> {tgt} (Score: {score:.4f} - требует ручной разметки)")

    if fn_list:
        print(f"\n[Подробно] Пропущенные эталонные связи (False Negatives) - [{len(fn_list)}]:")
        for src, tgt in fn_list:
            print(f"  ⏳ {src} <=======> {tgt}")

    # 5. Динамическая генерация графиков и отчетов НИР3
    print("\nШаг 5. Генерация графических научных артефактов...")
    output_heatmap_path = "./semantic_heatmap_v4.png"
    output_graph_path = "./semantic_graph_v4.png"
    
    DynamicVisualizer.draw_heatmap(matches_df, save_path=output_heatmap_path)
    DynamicVisualizer.draw_relation_graph(matches_df, save_path=output_graph_path)

    print("\n[Успешно] Все верификационные этапы НИР3 завершены!")
    print("=" * 75)


if __name__ == "__main__":
    main()
