import os
import urllib.request
import json
import pandas as pd
from abc import ABC, abstractmethod

# ======================================================================
# 1. АБСТРАКТНЫЙ ИНТЕРФЕЙС ЗАГРУЗЧИКА (ПЛАГИННАЯ АРХИТЕКТУРА)
# ======================================================================

class BaseBenchmarkLoader(ABC):
    """
    Абстрактный класс для загрузки бенчмарков сопоставления схем.
    Позволяет бесшовно подменять наборы данных (MIMIC-to-OMOP, Valentine, custom).
    """
    
    def __init__(self, cache_dir="./benchmark_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def load_schemas(self) -> pd.DataFrame:
        """
        Загружает схемы данных и преобразует их в унифицированный DataFrame:
        Колонки: ['system', 'container', 'element', 'description', 'fk_ref']
        """
        pass

    @abstractmethod
    def load_ground_truth(self) -> list:
        """
        Загружает эталонные сопоставления (ground truth).
        Возвращает список кортежей: [("Source_Element", "Target_Element"), ...]
        где элементы записаны в формате "container.element" или аналогичном.
        """
        pass

    def _download_file(self, url: str, filename: str) -> str:
        """Вспомогательный метод для загрузки файлов с поддержкой кэширования."""
        local_path = os.path.join(self.cache_dir, filename)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"[Кэш] Используется локальный файл: {local_path}")
            return local_path
        
        print(f"[Загрузка] Скачивание {url} -> {local_path}...")
        try:
            # Настройка заголовков User-Agent для предотвращения блокировок со стороны GitHub
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(local_path, 'wb') as f:
                    f.write(response.read())
            return local_path
        except Exception as e:
            print(f"[Предупреждение] Не удалось скачать файл (оффлайн режим): {e}")
            return local_path

# ======================================================================
# 2. РЕАЛИЗАЦИЯ: MIMIC-to-OMOP LOADER (Реальный медицинский кейс)
# ======================================================================

class MIMICtoOMOPLoader(BaseBenchmarkLoader):
    """
    Загрузчик реального медицинского бенчмарка MIMIC-III to OMOP CDM
    на основе репозитория meniData1/MIMIC_2_OMOP (наиболее точная промышленная разметка).
    """
    
    BASE_URL = "https://raw.githubusercontent.com/meniData1/MIMIC_2_OMOP/main/data"
    
    def load_schemas(self) -> pd.DataFrame:
        """
        Загружает оригинальные CSV-схемы MIMIC-III и OMOP CDM с описаниями.
        """
        mimic_path = self._download_file(f"{self.BASE_URL}/MIMIC_III_Schema.csv", "MIMIC_III_Schema.csv")
        omop_path = self._download_file(f"{self.BASE_URL}/OMOP_Schema.csv", "OMOP_Schema.csv")
        
        unified_elements = []

        # 1. Считывание и парсинг MIMIC-III схемы
        if os.path.exists(mimic_path) and os.path.getsize(mimic_path) > 0:
            try:
                # Используем encoding='utf-8-sig' для защиты от BOM-байтов
                df_mimic = pd.read_csv(mimic_path, encoding='utf-8-sig')
                for _, row in df_mimic.iterrows():
                    tbl = str(row['TableName']).strip()
                    col = str(row['ColumnName']).strip()
                    desc = str(row['ColumnDesc']).strip() if pd.notna(row['ColumnDesc']) else ""
                    
                    # Парсинг внешних связей для FK-бустинга
                    fk_ref = None
                    if pd.notna(row['IsFK']) and str(row['IsFK']).strip().upper() == "YES":
                        # Пример строки: "[PATIENTS, SUBJECT_ID]" -> "patients.subject_id"
                        fk_str = str(row['FK']).strip("[] ")
                        if "," in fk_str:
                            fk_tbl, fk_col = fk_str.split(",")
                            fk_ref = f"{fk_tbl.strip().lower()}.{fk_col.strip().lower()}"
                    
                    unified_elements.append({
                        "system": "MIMIC-III",
                        "container": tbl,
                        "element": col,
                        "description": desc,
                        "fk_ref": fk_ref
                    })
            except Exception as e:
                print(f"[Ошибка] Ошибка парсинга {mimic_path}: {e}")

        # 2. Считывание и парсинг OMOP CDM схемы
        if os.path.exists(omop_path) and os.path.getsize(omop_path) > 0:
            try:
                df_omop = pd.read_csv(omop_path, encoding='utf-8-sig')
                for _, row in df_omop.iterrows():
                    tbl = str(row['TableName']).strip()
                    col = str(row['ColumnName']).strip()
                    desc = str(row['ColumnDesc']).strip() if pd.notna(row['ColumnDesc']) else ""
                    
                    fk_ref = None
                    if pd.notna(row['IsFK']) and str(row['IsFK']).strip().upper() == "YES":
                        fk_tbl = str(row['FK table']).strip()
                        fk_col = str(row['FK column']).strip()
                        fk_ref = f"{fk_tbl.strip().lower()}.{fk_col.strip().lower()}"
                        
                    unified_elements.append({
                        "system": "OMOP_CDM",
                        "container": tbl,
                        "element": col,
                        "description": desc,
                        "fk_ref": fk_ref
                    })
            except Exception as e:
                print(f"[Ошибка] Ошибка парсинга {omop_path}: {e}")

        # Если файлов нет (абсолютный оффлайн без кэша), генерируем демо-выборку
        if not unified_elements:
            print("[Режим оффлайн/Демо] Генерация репрезентативной выборки MIMIC-to-OMOP")
            unified_elements = self._generate_mimic_omop_fallback_data()

        return pd.DataFrame(unified_elements)

    def load_ground_truth(self) -> list:
        """Загружает эталонные кросс-системные связи (MIMIC-III -> OMOP CDM)."""
        mapping_path = self._download_file(f"{self.BASE_URL}/MIMIC_to_OMOP_Mapping.csv", "MIMIC_to_OMOP_Mapping.csv")
        
        ground_truth = []
        if os.path.exists(mapping_path) and os.path.getsize(mapping_path) > 0:
            try:
                df_map = pd.read_csv(mapping_path, encoding='utf-8-sig')
                for _, row in df_map.iterrows():
                    src_tbl = str(row['SRC_ENT']).strip()
                    src_col = str(row['SRC_ATT']).strip()
                    tgt_tbl = str(row['TGT_ENT']).strip()
                    tgt_col = str(row['TGT_ATT']).strip()
                    
                    # Проверяем на пустые связи (NA/null)
                    if src_tbl != "NA" and tgt_tbl != "NA" and pd.notna(row['TGT_ENT']):
                        # Специфичное для НИР3 принудительное приведение к нижнему регистру
                        src_full = f"{src_tbl}.{src_col}".lower()
                        tgt_full = f"{tgt_tbl}.{tgt_col}".lower()
                        ground_truth.append((src_full, tgt_full))
            except Exception as e:
                print(f"[Ошибка] Ошибка чтения Ground Truth: {e}")
                
        if not ground_truth:
            ground_truth = [
                ("patients.subject_id", "person.person_id"),
                ("patients.gender", "person.gender_source_value"),
                ("patients.dob", "person.birth_datetime"),
                ("admissions.hadm_id", "visit_occurrence.visit_occurrence_id"),
                ("admissions.admittime", "visit_occurrence.visit_start_datetime")
            ]
        return ground_truth

    def _generate_mimic_omop_fallback_data(self) -> list:
        """Локальные метаданные медицинских карт для оффлайн-режима."""
        return [
            # Источник А: MIMIC-III (Электронные медкарты)
            {
                "system": "MIMIC-III", "container": "patients", "element": "subject_id",
                "description": "Unique identifier representing a single patient in the clinical database.", "fk_ref": None
            },
            {
                "system": "MIMIC-III", "container": "patients", "element": "gender",
                "description": "Biological sex of the patient at the time of administrative admission.", "fk_ref": None
            },
            {
                "system": "MIMIC-III", "container": "patients", "element": "dob",
                "description": "The exact date of birth of the subject, anonymized for HIPPA compliance.", "fk_ref": None
            },
            {
                "system": "MIMIC-III", "container": "admissions", "element": "hadm_id",
                "description": "Unique hospitality admission number identifying a single visit of a patient.", "fk_ref": None
            },
            {
                "system": "MIMIC-III", "container": "admissions", "element": "admittime",
                "description": "Timestamp indicating the precise date and time when the patient entered the hospital clinic.", "fk_ref": None
            },
            
            # Источник Б: OMOP CDM (Единый международный стандарт)
            {
                "system": "OMOP_CDM", "container": "person", "element": "person_id",
                "description": "A unique long integer identifier for each person in the standardized cohort database.", "fk_ref": None
            },
            {
                "system": "OMOP_CDM", "container": "person", "element": "gender_source_value",
                "description": "A field to capture gender or sex value present in the source data.", "fk_ref": None
            },
            {
                "system": "OMOP_CDM", "container": "person", "element": "birth_datetime",
                "description": "The precise calendar date and time of birth of the individual.", "fk_ref": None
            },
            {
                "system": "OMOP_CDM", "container": "visit_occurrence", "element": "visit_occurrence_id",
                "description": "Unique primary key identifying a distinct clinical visit of the person to a healthcare facility.", "fk_ref": None
            },
            {
                "system": "OMOP_CDM", "container": "visit_occurrence", "element": "visit_start_datetime",
                "description": "The date and time of the start of the clinical visit or encounter.", "fk_ref": None
            }
        ]

# ======================================================================
# 3. АЛЬТЕРНАТИВНАЯ РЕАЛИЗАЦИЯ: VALENTINE LOADER
# ======================================================================

class ValentineBenchmarkLoader(BaseBenchmarkLoader):
    """
    Загрузчик бенчмарка Valentine. 
    Позволяет загружать стандартные пары таблиц и Ground Truth.
    """
    
    def __init__(self, source_csv: str, target_csv: str, gt_csv: str, cache_dir="./benchmark_cache"):
        super().__init__(cache_dir)
        self.source_csv = source_csv
        self.target_csv = target_csv
        self.gt_csv = gt_csv

    def load_schemas(self) -> pd.DataFrame:
        """Загружает метаданные колонок из CSV-файлов Valentine."""
        unified_elements = []
        
        # Считываем названия колонок и генерируем контекстные описания
        for csv_path, sys_name in [(self.source_csv, "Source_DB"), (self.target_csv, "Target_DB")]:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, nrows=5) # Достаточно прочитать заголовки
                tbl_name = os.path.basename(csv_path).replace(".csv", "")
                for col in df.columns:
                    unified_elements.append({
                        "system": sys_name,
                        "container": tbl_name,
                        "element": col,
                        "description": f"Field '{col}' in table '{tbl_name}' of dataset '{sys_name}'."
                    })
        return pd.DataFrame(unified_elements)

    def load_ground_truth(self) -> list:
        """Считывает ground truth Valentine."""
        ground_truth = []
        if os.path.exists(self.gt_csv):
            df = pd.read_csv(self.gt_csv)
            # Формат: source_column, target_column
            for _, row in df.iterrows():
                ground_truth.append((row[0], row[1]))
        return ground_truth


if __name__ == "__main__":
    print("======================================================================")
    print("ИНТЕЛЛЕКТУАЛЬНЫЙ МОДУЛЬ ЗАГРУЗКИ БЕНЧМАРКОВ СХЕМ (ВЕРСИЯ V4.1)")
    print("======================================================================")
    
    loader = MIMICtoOMOPLoader()
    print("\n--- Инициализация MIMIC-to-OMOP Loader ---")
    schemas_df = loader.load_schemas()
    gt_list = loader.load_ground_truth()
    
    print(f"Успешно загружено полей: {len(schemas_df)}")
    print(f"Успешно загружено эталонных связей (Ground Truth): {len(gt_list)}")
    print("\nПример структуры метаданных:")
    print(schemas_df.head(4)[['system', 'container', 'element']])
    print("\nПример Ground Truth:")
    for src, tgt in gt_list[:3]:
        print(f"  {src}  <=======>  {tgt}")
    print("======================================================================")
