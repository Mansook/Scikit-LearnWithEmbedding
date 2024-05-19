import json
import pandas as pd
from dataParsing import text
from dataParsing import labels
from embedding import embedding
# 문자열 배열
strings = ["시발", "how cool"]
# JSON 형식으로 변환
# 텍스트 배열을 2000개씩 나누기


def chunk_list(lst, chunk_size):
    """리스트를 일정 크기로 나누어 반환합니다."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


chunk_size = 2000
text_chunks = list(chunk_list(text, chunk_size))

# 모든 임베딩 데이터를 저장할 리스트
all_embeddings = []

# 각 텍스트 청크에 대해 임베딩 요청하기
for chunk in text_chunks:
    embeddings = embedding(chunk)
    all_embeddings.extend(embeddings)

# 임베딩 데이터를 데이터프레임으로 변환
df = pd.DataFrame(all_embeddings)

# 데이터프레임을 엑셀 파일로 저장
df.to_excel('embeddingl.xlsx', index=False, header=False)
print("임베딩 데이터가 'embeddingl.xlsx' 파일에 저장되었습니다.")
