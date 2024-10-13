import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(dir_data):
    dir_data = dir_data

    name_label = 'train-labels.tif'
    name_input = 'train-volume.tif'

    """
    os.path.join을 통해 dir_data에 name_label, 혹은 name_input을 경로 형태로 결합한 후에
    Image.open을 통해 해당 경로에 있는 이미지 파일을 identification한다.
    (이미지의 데이터를 읽어드리는 것이 아니라 단지 이미지 파일에 대한 몇 가지 정보만을 확인)
    """

    img_label = Image.open(os.path.join(dir_data, name_label))
    img_input = Image.open(os.path.join(dir_data, name_input))

    ny, nx = img_label.size
    nframe = img_label.n_frames

    
    print(f"Data Size: {ny} X {nx}")
    print(f"Number of frames(Data): {nframe}")
    print('-----------------------------------------------------------------------')



    ## 디렉토리 생성

    # Train, Validation, Test 데이터셋 개수 지정
    nframe_train = 24
    nframe_val = 3
    nframe_test = 3

    dir_save_train = os.path.join(dir_data, 'train')  # dir_data + train => ./datasets/train
    dir_save_val = os.path.join(dir_data, 'val')
    dir_save_test = os.path.join(dir_data, 'test')

    """
    os.path.join을 통해 결합된 경로를 바탕으로
    os.makedirs를 사용하여 해당 경로를 생성한다.
    """
    os.makedirs(dir_save_train)
    os.makedirs(dir_save_val)
    os.makedirs(dir_save_test)

    print("Train, Test, Validation directory generation complete!")
    print('-----------------------------------------------------------------------')




    ## Train, Val, Test 데이터로 나누기

    id_frame = np.arange(nframe)  # nframe = 30 -> id_frame = [0 1 2 3 4 ... 28 29]
    np.random.shuffle(id_frame)  # id_frame을 random하게 shuffle

    offset_nframe = 0

    for i in range(nframe_train):  # nframe_train = 24

        """
        이미지 데이터를 load 할 때는 보통 read() 함수를 사용하지만
        이미지들이 frame 단위로 묶여있는 경우 seek() 함수를 이용해 load한다.
        train-volume.tif, train-labels.tif의 shape: 512X512X30

        img_label.seek(id_frame[i + offset_nframe])에서 img_label 경로 상에 있는 이미지 파일을 load 하는데,
        (id_frame[i + offset_nframe]을 통해 몇 번째 frame을 load 할 지 결정한다.
        이때 for문을 통해 nframe_train 수 만큼 반복한다.
        """

        img_label.seek(id_frame[i + offset_nframe])  # ex id_frame = [0, 2, 10, 5 ...] 일때, id_frame[3] = 5
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)  # numpy array 형태로 변환
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)  # ./datasets/train 경로에 array 저장
        np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)



    offset_nframe += nframe_train  # offset_nframe = 0 + 24 = 24

    for i in range(nframe_val):  # nframe_val = 3, validation data를 3개로 설정했으므로 
        img_label.seek(id_frame[i + offset_nframe])  # i + offset_nframe 의 범위는 24~26(정수)
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)



    offset_nframe += nframe_val  # offset_nframe = 24 + 3 = 27

    for i in range(nframe_test):  # nframe_test = 3, test data를 3개로 설정했으므로
        img_label.seek(id_frame[i + offset_nframe])  # i + offset_nframe의 범위는 27~29(정수수)
        img_input.seek(id_frame[i + offset_nframe])

        label_ = np.asarray(img_label)
        input_ = np.asarray(img_input)

        np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
        np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)


    print('Data is sotred in each directory!')
    print('-----------------------------------------------------------------------')


        # 이미지 데이터 plot

    """
    두 개의 데이터를 나란히 plot 할 수 있다.
    plt.subplot()의 입력값은 행의 수, 열의 수, index 순이다.
    행의 수가 1, 열의 수가 2 이므로 가로로 2개의 데이터가 plot 되었다.
    이때 label의 index가 1, input의 index가 2 이므로 label은 왼쪽, input은 오른쪽에 plot 된다.
    """
    plt.subplot(121)
    plt.imshow(label_, cmap='gray')  # label_: 불러온 frame이 numpy array로 변환된 형태
    plt.title('label')

    plt.subplot(122)
    plt.imshow(input_, cmap='gray')
    plt.title('input')

    plt.show()