import traceback
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def thread_process_func_decorator(func, item):
    order = item["THREAD_ORDER"]
    try:
        if order % 10000 == 0:
            print(f"处理进度{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:{order}")
        # 调用原始函数
        result = func(item)
        item["PROC_STATUS"] = "SUC"
        # 返回原始函数的结果
        return result
    except Exception as ex:
        traceback_info = traceback.format_exc()
        print(f"处理异常:{item} \n异常信息:{traceback_info}")
        item["PROC_STATUS"] = "ERR"
    return item


def process_dataframe_multithread(data: pd.DataFrame, process_func, num_threads: int):
    """
    多线程处理DataFrame
    :param data:
    :param process_func:
    :param num_threads:
    :return:
    """
    data['THREAD_ORDER'] = range(len(data))
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for index, item in data.iterrows():
            futures.append(executor.submit(thread_process_func_decorator, process_func, dict(item)))
        # 获取任务执行结果
        results = [future.result() for future in futures]
        # 合并处理结果
        processed_df = pd.DataFrame(results)
        processed_df = processed_df.sort_values(by="THREAD_ORDER")
        del processed_df["THREAD_ORDER"]
        # 返回处理后的 DataFrame
        return processed_df