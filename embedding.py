import requests
import json


def embedding(txt):
    # OpenAI API 키 설정
    api_key = "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"

    # 요청 헤더 설정
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # 요청 데이터 설정
    data = {
        "input": txt,
        "model": "text-embedding-3-small"
    }

    response = requests.post(
        'https://api.openai.com/v1/embeddings', headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_data = response.json()
        embeddings = [item['embedding'] for item in response_data['data']]
        return embeddings

    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None
