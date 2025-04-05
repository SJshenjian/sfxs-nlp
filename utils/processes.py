import traceback
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 全局定义 process_func
def process_func(item):
    return {"result": item["value"] * 2} if "value" in item else None

# 全局定义线程处理装饰器
def thread_process_func_decorator(item):
    order = item["THREAD_ORDER"]
    try:
        if order % 100000 == 0:
            print(f"处理进度 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:{order}")
        result = process_func(item)
        return item if result is None else {**item, **result}
    except Exception as ex:
        traceback_info = traceback.format_exc()
        print(f"处理异常:{item} \n异常信息:{traceback_info}")
        item["PROC_STATUS"] = "ERR"
    return item

# 全局定义处理块的函数
def process_chunk(chunk):
    # 注意：这里不能直接访问 num_processes，需要通过参数传递或其他方式
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(thread_process_func_decorator, dict(item))
            for _, item in chunk.iterrows()
        ]
        return pd.DataFrame([future.result() for future in futures])

def process_dataframe_multiprocess(data: pd.DataFrame, process_func_param, num_processes: int = None, chunk_size: int = 1000):
    """
    多进程处理 DataFrame
    """
    global process_func
    process_func = process_func_param  # 将传入的 process_func 赋值给全局变量

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    data['THREAD_ORDER'] = range(len(data))

    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    with ProcessPoolExecutor(max_workers=min(num_processes, len(chunks))) as executor:
        processed_chunks = list(executor.map(process_chunk, chunks))  # 直接调用全局函数

    processed_df = pd.concat(processed_chunks, ignore_index=False)
    del processed_df['THREAD_ORDER']
    return processed_df

if __name__ == "__main__":
    df = pd.DataFrame({"value": range(10000)})
    result_df = process_dataframe_multiprocess(df, process_func, num_processes=4, chunk_size=100000)
    print(result_df.head())