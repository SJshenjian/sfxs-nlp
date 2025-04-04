from elasticsearch import helpers, Elasticsearch


def init_es_client(es_host):
    es = Elasticsearch(hosts=[es_host], verify_certs=False)
    return es


es_client = init_es_client('http://127.0.0.1:9200')


def query_enterprises_scroll(index_name="enterprise_info", province="江苏省"):
    """
    使用 Scroll API 查询所有企业。
    """
    # 初始查询
    query = {
        "query": {
            "term": {
                "province": province
            }
        },
        "_source": ["company_name", "registered_address", "province", "city", "district"],
        "size": 10000  # 每批次大小
    }

    enterprises = []
    try:
        # 第一次查询，获取 scroll_id
        response = es_client.search(index=index_name, body=query, scroll="2m")
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        count = 0
        while hits:
            for hit in hits:
                source = hit["_source"]
                enterprises.append({
                    "company_name": source.get("company_name", "未知企业"),
                    "registered_address": source.get("registered_address", "未知地址"),
                    "province": source.get("province", "未知省份"),
                    "city": source.get("city", "未知城市"),
                    "district": source.get("district", "未知区县")
                })
                count += 1
                if count % 100000 == 0:
                    print(f"已加载 {count} 条记录...")

            # 使用 scroll_id 获取下一批数据
            response = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
        print(f"共加载 {count} 条记录")

        # 清理 scroll
        es_client.clear_scroll(scroll_id=scroll_id)
        return enterprises

    except Exception as e:
        print(f"查询失败: {e}")
        return []
