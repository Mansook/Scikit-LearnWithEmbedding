def parse_data(file_path):
    data = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 줄 끝의 줄 바꿈 문자를 제거
            if '|' in line:
                text, label = line.split('|', 1)  # '|'을 기준으로 텍스트와 라벨을 분리
                data.append(text)
                labels.append(label)

    return data, labels


file_path = 'dataset.txt'

text, labels = parse_data(file_path)
print(len(labels))
