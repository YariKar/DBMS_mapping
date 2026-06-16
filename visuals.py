import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Список уникальных полей, участвующих в обнаруженных связях
labels = [
    "PG: users.created_at",
    "Mongo: customers.profile.registration_date",
    "PG: orders.user_fk",
    "PG: order_items.order_ref",
    "Neo4j: UserSessionLog.logged_at"
]

def draw_heatmap():
    """
    Матрица косинусного сходства на основе твоих реальных результатов.
    Диагональ равна 1.0 (поле схоже само с собой).
    Остальные ячейки заполнены точными весами из твоего лога.
    """
    n = len(labels)
    similarity_matrix = np.zeros((n, n))
    
    # Заполняем диагональ (селф-матчинг)
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Индексы полей для точного маппинга матрицы:
    # 0 - PG: users.created_at
    # 1 - Mongo: customers.profile.registration_date
    # 2 - PG: orders.user_fk
    # 3 - PG: order_items.order_ref
    # 4 - Neo4j: UserSessionLog.logged_at
    
    # Связь #1: PG: users.created_at <-> Mongo: registration_date (0.6364)
    similarity_matrix[0, 1] = 0.6364
    similarity_matrix[1, 0] = 0.6364
    
    # Связь #2: PG: orders.user_fk <-> PG: order_items.order_ref (0.6166)
    similarity_matrix[2, 3] = 0.6166
    similarity_matrix[3, 2] = 0.6166
    
    # Связь #3: PG: users.created_at <-> Neo4j: UserSessionLog.logged_at (0.5595)
    similarity_matrix[0, 4] = 0.5595
    similarity_matrix[4, 0] = 0.5595

    plt.figure(figsize=(9, 7))
    im = plt.imshow(similarity_matrix, cmap="YlGnBu", vmin=0, vmax=1)
    
    plt.xticks(np.arange(n), labels, rotation=35, ha="right", fontname="DejaVu Sans")
    plt.yticks(np.arange(n), labels, fontname="DejaVu Sans")
    
    # Отрисовка числовых значений внутри матрицы
    for i in range(n):
        for j in range(n):
            val = similarity_matrix[i, j]
            if val > 0: # Показываем только значимые связи, чтобы не перегружать график
                color = "white" if val > 0.6 else "black"
                plt.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontweight="bold")

    plt.title("Экспериментальная матрица косинусного сходства полей\n(Реальные результаты работы SLM-алгоритма)", fontsize=12, pad=15)
    plt.colorbar(im, label="Коэффициент семантического сходства")
    plt.tight_layout()
    plt.savefig("semantic_heatmap_real.png", dpi=300)
    print("[Успешно] График 'semantic_heatmap_real.png' сохранен.")
    plt.show()

def draw_relation_graph():
    """
    Построение графа реальных кросс-модельных связей.
    Толщина линий зависит от уровня уверенности алгоритма.
    """
    G = nx.Graph()

    # Маппинг узлов по СУБД для цветового кодирования (визуальный акцент на мультимодальности)
    node_systems = {
        "PG: users.created_at": "PostgreSQL",
        "Mongo: customers.profile.registration_date": "MongoDB",
        "PG: orders.user_fk": "PostgreSQL",
        "PG: order_items.order_ref": "PostgreSQL",
        "Neo4j: UserSessionLog.logged_at": "Neo4j"
    }
    
    for node, system in node_systems.items():
        G.add_node(node, system=system)

    # Реальные ребра (твои 3 связи)
    edges = [
        ("PG: users.created_at", "Mongo: customers.profile.registration_date", 0.6364),
        ("PG: orders.user_fk", "PG: order_items.order_ref", 0.6166),
        ("PG: users.created_at", "Neo4j: UserSessionLog.logged_at", 0.5595)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    plt.figure(figsize=(10, 7))
    
    # Раскладка графа
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    # Цветовая палитра для разных моделей данных
    color_map = []
    for node in G:
        sys = G.nodes[node]['system']
        if sys == "PostgreSQL": color_map.append("#336791") # Насыщенный синий
        elif sys == "MongoDB": color_map.append("#4DB33D")   # Зеленый NoSQL
        else: color_map.append("#008CC1")                   # Фирменный голубой Neo4j

    # Рисуем узлы и подписи
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=3200, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", font_weight="bold")
    
    # Рисуем связи (толщина пропорциональна уверенности)
    weights = [G[u][v]['weight'] * 6 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="#7f8c8d", alpha=0.7)
    
    # Добавляем веса (метки) на ребра
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_weight="bold")

    plt.title("Граф извлеченных семантических отношений (Мультимодальная топология)\nРезультат этапа автоматического выравнивания схем данных", fontsize=12, pad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("semantic_graph_real.png", dpi=300)
    print("[Успешно] График 'semantic_graph_real.png' сохранен.")
    plt.show()

if __name__ == "__main__":
    print("Запуск генерации точных научных графиков на основе логов НИР...")
    draw_heatmap()
    print("-" * 50)
    draw_relation_graph()
