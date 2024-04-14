import torch  # noqa
from torchvision.io import read_image, write_png
import meanfilter


x = read_image("Grace_Hopper.jpg").contiguous().cuda()
result = meanfilter.forward(x, 3)
write_png(result.cpu(), "output.png")
print(result)
print(x)
