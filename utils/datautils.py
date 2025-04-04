import hashlib


def get_unique_id(val):
    try:
        md5_hash = hashlib.md5(val.encode('utf-8'))
        unique_id = md5_hash.hexdigest()
        return unique_id
    except Exception as e:
        print(f"生成ID错误: {e}")
        return None
