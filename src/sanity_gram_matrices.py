from models.ResNet import ResNet, BasicBlock


if __name__ == '__main__':
    torch_model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=10)
