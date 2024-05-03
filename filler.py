import torch

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.is_available())
    # # print()
    # # testShapeNN = Critic(1).to(device)

    # img = cv2.imread('testImage.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RG
    #     transforms.ToTensor()
    # ])
    # tensor = transform(img).to(device)
    # print(tensor.shape)
    # # output = testShapeNN.forward(tensor)

    # # print(output.shape)

    # print(device)
    # test_shape = Actor().to(device)
    # test_shape.forward(tensor)