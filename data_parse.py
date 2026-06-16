import pandas as pd


# --- 1. СЫРЫЕ ДАННЫЕ ИЗ РАЗНЫХ СУБД (РЕАЛИСТИЧНЫЙ ВИД) ---

raw_db_metadata = [
    # РЕЛИЗ ИЗ POSTGRESQL: Результат SQL-запроса к information_schema
    {
        "source_type": "PostgreSQL",
        "raw_payload": {
            "columns": ["table_name", "column_name", "data_type", "column_comment"],
            "rows": [
                ["users", "usr_id", "integer", "Primary key for the registered user account inside the main application."],
                ["users", "created_at", "timestamp", "The exact date and time when user profile was created."],
                ["orders", "order_id", "integer", "Primary key representing a customer purchase transaction record."],
                ["orders", "user_fk", "integer", "Foreign key linking the purchase to the specific buyer account."],
                ["order_items", "item_id", "integer", "Unique identifier for a line item within a purchase order."],
                ["order_items", "order_ref", "integer", "Reference identifier linking back to the parent purchase transaction."]
            ]
        }
    },
    
    # ОТВЕТ ИЗ MONGODB: Стандартная JSON-схема валидации (коллекция customers)
    {
        "source_type": "MongoDB",
        "raw_payload": {
            "collection": "customers",
            "validator": {
                "$jsonSchema": {
                    "bsonType": "object",
                    "required": ["_id", "profile"],
                    "properties": {
                        "_id": {
                            "bsonType": "objectId",
                            "description": "Unique BSON identifier for the client document record."
                        },
                        "profile": {
                            "bsonType": "object",
                            "properties": {
                                "client_name": {
                                    "bsonType": "string",
                                    "description": "Full name of the registered buyer or individual customer."
                                },
                                "registration_date": {
                                    "bsonType": "date",
                                    "description": "Timestamp indicating when the client profile was created in the system."
                                }
                            }
                        }
                    }
                }
            }
        }
    },

    # ОТВЕТ ИЗ NEO4J: Результат вызова процедуры apoc.meta.schema()
    {
        "source_type": "Neo4j",
        "raw_payload": {
            "labels": {
                "UserSessionLog": {
                    "type": "node",
                    "properties": {
                        "session_id": {"type": "STRING", "comment": "Unique token identifier for the active user login session."},
                        "account_id_ref": {"type": "INTEGER", "comment": "Reference pointer to the registered user account who generated this event."},
                        "logged_at": {"type": "LOCAL_DATETIME", "comment": "The exact date and time when the system activity occurred."}
                    }
                }
            }
        }
    }
]

# --- 2. МОДУЛЬ УНИФИКАЦИИ (ПАРСЕР МЕТАДАННЫХ) ---

class MetadataUnifier:
    """
    Этот класс имитирует коннекторы. Он парсит специфичные форматы СУБД 
    и приводит их к единому плоскому списку элементов для NLP-анализа.
    """
    def unify(self, raw_data_list):
        unified_elements = []
        
        for db in raw_data_list:
            stype = db["source_type"]
            payload = db["raw_payload"]
            
            if stype == "PostgreSQL":
                # Парсим строки таблицы
                for row in payload["rows"]:
                    unified_elements.append({
                        "system": "PostgreSQL",
                        "container": row[0],     # Имя таблицы
                        "element": row[1],       # Имя колонки
                        "description": row[3]    # Комментарий
                    })
                    
            elif stype == "MongoDB":
                # Парсим JSON-схему (для демо — рекурсивный обход заменен на прямой доступ к полям)
                coll = payload["collection"]
                props = payload["validator"]["$jsonSchema"]["properties"]
                
                # Корень _id
                unified_elements.append({
                    "system": "MongoDB", "container": coll, "element": "_id",
                    "description": props["_id"]["description"]
                })
                # Вложенные поля profile
                sub_props = props["profile"]["properties"]
                for sub_key, sub_val in sub_props.items():
                    unified_elements.append({
                        "system": "MongoDB", "container": coll, "element": f"profile.{sub_key}",
                        "description": sub_val["description"]
                    })
                    
            elif stype == "Neo4j":
                # Парсим свойства узлов графа
                for label, data in payload["labels"].items():
                    for prop_name, prop_info in data["properties"].items():
                        unified_elements.append({
                            "system": "Neo4j", "container": label, "element": prop_name,
                            "description": prop_info["comment"]
                        })
                        
        return pd.DataFrame(unified_elements)


