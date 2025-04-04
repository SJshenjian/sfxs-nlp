from fastapi import FastAPI
from scripts.train import train
from scripts.predict import predict
import pandas as pd
app = FastAPI()


@app.get("/")
async def root():
    # 训练模型
    print("Training model...")
    train()

    # 推理
    print("\nPredicting...")
    address = "湖北省孝感市云梦县曾店镇刘吴村132号"
    result = predict(address)
    print(result)

def split():
    # 推理
    print("\nPredicting...")
    addr = pd.read_csv("company/江苏省企业.csv", encoding="utf-8")
    for address in addr.head(100)["registered_address"]:
        print(f"\nAddress: {address}")
        result = predict(address)
        print(result)

if __name__ == "__main__":
    train()
    split()