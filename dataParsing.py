import pandas as pd

file_path = 'dataset.txt'
file_path2 = 'dataset2.xlsx'


def parse_data():
    data = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거
            if '|' in line:
                text, label = line.split('|', 1)  # '|'을 기준으로 텍스트와 라벨을 분리
                data.append(text)
                labels.append(label)

    df = pd.read_excel(file_path2, header=None)
    lines = df.values
    for line in lines:
        line = line[0]
        text, label = line.split('\t', 1)
        data.append(text)
        if label == '0':
            labels.append(1)
        else:
            labels.append(0)

    return data, labels


text, labels = parse_data()

# 라벨 데이터를 DataFrame으로 변환
labels_df = pd.DataFrame(labels, columns=['label'])

# DataFrame을 엑셀 파일로 저장
labels_df.to_excel('labels.xlsx', index=False)

print("라벨 데이터가 labels.xlsx 파일에 저장되었습니다.")
