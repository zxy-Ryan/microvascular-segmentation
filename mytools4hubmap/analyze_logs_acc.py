import matplotlib.pyplot as plt
import re


def extract_data_from_log(file_path):
    # 正则表达式，用于匹配和提取所需的数据
    pattern = r'\|\s*(blood_vessel|glomerulus)\s*\|\s*(\d+\.\d+)\s*\|\s*(\d+\.\d+)\s*\|'

    # 初始化列表来保存提取的数据
    blood_vessel_first = []
    blood_vessel_second = []
    glomerulus_first = []
    glomerulus_second = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                category = match.group(1)
                first_value = float(match.group(2))
                second_value = float(match.group(3))

                if category == 'blood_vessel':
                    blood_vessel_first.append(first_value)
                    blood_vessel_second.append(second_value)
                elif category == 'glomerulus':
                    glomerulus_first.append(first_value)
                    glomerulus_second.append(second_value)

    return blood_vessel_first, blood_vessel_second, glomerulus_first, glomerulus_second


def plot_data(blood_vessel_first, blood_vessel_second, glomerulus_first, glomerulus_second):
    plt.plot(blood_vessel_first, label='Blood Vessel First Number')
    plt.plot(blood_vessel_second, label='Blood Vessel Second Number')
    plt.plot(glomerulus_first, label='Glomerulus First Number')
    plt.plot(glomerulus_second, label='Glomerulus Second Number')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Data from Log File')
    plt.legend()
    plt.show()


log_file_path = '../result/mmseg_result/hubmap/deeplabv3/20231201_013349/20231201_013349.log'

bv_first, bv_second, gl_first, gl_second = extract_data_from_log(log_file_path)

plot_data(bv_first, bv_second, gl_first, gl_second)
