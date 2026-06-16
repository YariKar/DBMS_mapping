from semantic_matcher import SemanticMatcher
from data_parse import raw_db_metadata, MetadataUnifier

# --- ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА ---
if __name__ == "__main__":
    print("1. Получение сырых гетерогенных метаданных...")
    unifier = MetadataUnifier()
    unified_df = unifier.unified_df = unifier.unify(raw_db_metadata)
    
    print("\nУнифицированный датасет, готовый к анализу:")
    print(unified_df[['system', 'container', 'element']])
    
    print("\n2. Запуск семантического анализа...")
    matcher = SemanticMatcher()
    matches = matcher.find_matches(unified_df, threshold=0.55)
    
    print("\n3. Найдено связей:", len(matches))
    for idx, r in matches.head(5).iterrows():
        print(f"\n[Связь #{idx+1}] Уверенность: {r['Score']}")
        print(f"  Сущность А: ({r['A_Sys']}) {r['A_Loc']} -> '{r['A_Desc']}'")
        print(f"  Сущность Б: ({r['B_Sys']}) {r['B_Loc']} -> '{r['B_Desc']}'")
