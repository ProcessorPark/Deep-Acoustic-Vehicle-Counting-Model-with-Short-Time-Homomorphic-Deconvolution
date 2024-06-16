import os

# 폴더 만들기 함수
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print(f'create {folder_name}')        
        os.makedirs(folder_name)

# 가장 마지막 폴더 다음 번호 이름 출력
def new_num_folder_name(folder_name):
    for nn in range(10000):  
        new_folder_name = f'{folder_name}{nn}'
        
        if not os.path.isdir(new_folder_name):
            print(new_folder_name)            
            break
    
    return new_folder_name

# 존재하는 폴더 이름 중에 가장 마지막 폴더 이름 출력
def last_num_folder_name(folder_name):
    last_folder_name = f'{folder_name}0'
    
    if not os.path.isdir(last_folder_name):
        print(last_folder_name)
        return last_folder_name
    else:
        for nn in range(10000):
            last_folder_name = f'{folder_name}{nn+1}' # 다음 폴더 번호를 확인
            if not os.path.isdir(last_folder_name):
                last_folder_name = f'{folder_name}{nn}'
                print(last_folder_name)
                break            

    return last_folder_name

# 폴더에 있는 모든 (하위 포함) 파일들 경로를 리스트로 출력
def list_all_file_path(dir_path):
    file_paths = []

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)            
            file_paths.append(file_path)

    return file_paths


def ckpt_path(ckpt_num):
    ckpt_dir = f'lightning_logs/version_{ckpt_num}/checkpoints'
    ckpt_name = os.listdir(ckpt_dir)

    return f'{ckpt_dir}/{ckpt_name[0]}'
