To deploy, download the model from the code below and place it in this folder

```[python]
import torch
from torchvision.models import resnet

model = resnet.resnet34(pretrained=True)
model.eval()

traced_model = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
traced_model.save('resnet34.pt')
```
