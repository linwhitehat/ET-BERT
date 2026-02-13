import json
import csv

def verify_predictions():
    # 1. dataset.json 로드 및 Hex -> Label 맵핑 생성
    with open('datasets/tls_version/dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    hex_to_label = {}
    for label, content in dataset.items():
        if 'payload' in content:
            for _, hex_data in content['payload'].items():
                # 공백 제거 등 전처리가 필요하다면 여기서 수행 (현재는 strip()만 적용)
                hex_to_label[hex_data.strip()] = str(label)

    # 2. 파일 비교 검증
    # nolabel_test_dataset.tsv (Hex 데이터)와 prediction.tsv (예측 라벨) 로드
    with open('datasets/tls_version/nolabel_test_dataset.tsv', 'r', encoding='utf-8') as f_hex, \
         open('datasets/tls_version/prediction.tsv', 'r', encoding='utf-8') as f_pred:
        
        # TSV 파일이므로 delimiter='\t'
        hex_reader = csv.reader(f_hex, delimiter='\t')
        pred_reader = csv.reader(f_pred, delimiter='\t')
        
        # 헤더 건너뛰기
        next(hex_reader)
        next(pred_reader)
        
        total = 0
        correct = 0
        mismatches = []
        
        # zip을 사용하여 두 파일을 동시에 순회
        for i, (hex_row, pred_row) in enumerate(zip(hex_reader, pred_reader)):
                        
            target_hex = hex_row[0].strip()
            pred_val = pred_row[0].strip()
            
            # 정답 조회
            true_val = hex_to_label.get(target_hex)
            
            if true_val:
                total += 1
                if true_val == pred_val:
                    correct += 1
                else:
                    mismatches.append((i, true_val, pred_val))
    
    # 결과 출력
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")

    idx = [0] * 84
    for a in mismatches:
        idx[int(a[1])] += 1

    k = 0
    for a in idx:
        print(f"{k}: {a}")
        k += 1

if __name__ == "__main__":
    verify_predictions()
