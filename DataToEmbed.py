import json
import pandas as pd
from dataParsing import text
from dataParsing import labels
from embedding import embedding


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


# api호출시 대략 3000개정도가 최대임 2000개씩 나눠서 요청
chunk_size = 2000
text_chunks = list(chunk_list(text, chunk_size))


all_embeddings = []


for chunk in text_chunks:
    embeddings = embedding(chunk)
    all_embeddings.extend(embeddings)


df = pd.DataFrame(all_embeddings)

# 데이터프레임을 엑셀 파일로 저장
df.to_excel('embeddingl.xlsx', index=False, header=False)
print("임베딩 데이터가 'embeddingl.xlsx' 파일에 저장되었습니다.")
